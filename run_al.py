#!/usr/bin/env python3
"""
run_al.py — FusionAL unified active learning CLI.

Two operating modes
-------------------
mve     ALSU surrogates (SingleBackbone, BigFusion, etc.) trained on
        pre-extracted backbone embeddings (.npz files).

molpal  PretrainedAL-VS models (RF, GP, NN, MPN, Transformer, MolCLR)
        trained on molecular fingerprints derived from SMILES strings.

Examples
--------
# MVE single-backbone UCB
python run_al.py --mode mve --dataset Enamine10k \
    --backbone molformer --surrogate single --acq ucb \
    --init-size 200 --batch-size 100 --n-rounds 5

# MVE BigFusion with borda acquisition
python run_al.py --mode mve --dataset Enamine10k \
    --backbones grover molformer unimol --surrogate bigfusion --acq borda \
    --n-rounds 5

# MolPAL RF on fingerprints
python run_al.py --mode molpal --dataset Enamine10k \
    --model rf --acq greedy --n-rounds 5
"""

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent

# ── paths ─────────────────────────────────────────────────────────────────────
EMBED_DIR   = ROOT / "results" / "embed"
DATA_DIR    = ROOT / "data"
LIBRARY_DIR = ROOT / "molpal" / "libraries"
RUNS_DIR    = ROOT / "runs"
RUNS_DIR.mkdir(exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==============================================================================
# DATA LOADING
# ==============================================================================

def load_embeddings(dataset: str, backbones: list[str]) -> tuple[dict, list]:
    """Load pre-extracted .npz embeddings for the given backbones."""
    from molpal.featurizer import EmbeddingFeaturizer
    ef = EmbeddingFeaturizer(
        embed_dir=str(EMBED_DIR / dataset),
        backbones=backbones,
    )
    return ef.load()


def load_oracle(dataset: str) -> dict:
    """Load docking scores as {smiles: score} dict (lower = better)."""
    gz = DATA_DIR / f"{dataset}_scores.csv.gz"
    assert gz.exists(), f"Oracle not found: {gz}"
    df = pd.read_csv(gz)
    df.columns = df.columns.str.strip().str.lower()
    smi_col   = next(c for c in df.columns if "smiles" in c)
    score_col = next(c for c in df.columns if "score"  in c)
    oracle = dict(zip(df[smi_col], df[score_col]))
    print(f"[oracle] {gz.name}: {len(oracle):,} molecules  "
          f"range [{df[score_col].min():.2f}, {df[score_col].max():.2f}]")
    return oracle


def load_library_smiles(dataset: str) -> list[str]:
    """Ordered SMILES from the MolPAL library CSV (supports .csv or .csv.gz)."""
    for suffix in (f"{dataset}.csv.gz", f"{dataset}.csv"):
        lib = LIBRARY_DIR / suffix
        if lib.exists():
            df = pd.read_csv(lib)
            df.columns = df.columns.str.strip().str.lower()
            smi_col = next(c for c in df.columns if "smiles" in c)
            return df[smi_col].dropna().tolist()
    raise FileNotFoundError(
        f"Library not found for dataset '{dataset}' in {LIBRARY_DIR}. "
        f"Expected {dataset}.csv or {dataset}.csv.gz"
    )


# ==============================================================================
# LOOKUP OBJECTIVE (wraps oracle dict for Explorer compatibility)
# ==============================================================================

class DictObjective:
    """Thin wrapper so Explorer can call objective(smis) → {smi: score}."""

    def __init__(self, oracle: dict, minimize: bool = True):
        self.oracle = oracle
        self.c = -1.0 if minimize else 1.0   # flip sign so Explorer maximises

    def __call__(self, smis):
        return {
            s: self.c * self.oracle[s] if s in self.oracle else None
            for s in smis
        }

    @property
    def path(self):
        return None


# ==============================================================================
# MVE ACTIVE LEARNING LOOP  (ALSU-style, embedding-centric)
# ==============================================================================

class MVEExplorer:
    """Runs the active learning loop using ALSU surrogates on embeddings."""

    def __init__(
        self,
        emb_dict: dict,
        pool_smiles: list,
        oracle: dict,
        surrogate_type: str,
        backbone: str,
        acq: str,
        init_size: int = 200,
        batch_size: int = 100,
        n_rounds: int = 10,
        epochs: int = 150,
        dataset_name: str = "Enamine50k",
        run_dir: Path = None,
        seed: int = 42,
    ):
        from molpal.models import mve as build_mve
        from molpal.acquirer.metrics import get_metric, borda

        self.pool_smiles = np.array(pool_smiles)
        self.oracle      = oracle
        self.batch_size  = batch_size
        self.n_rounds    = n_rounds
        self.run_dir     = run_dir or RUNS_DIR / "mve_run"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._sign = -1.0   # oracle scores are negative; negate to maximise

        # Build model
        self.model = build_mve(
            surrogate_type=surrogate_type,
            backbone=backbone,
            emb_dict=emb_dict,
            pool_smiles=pool_smiles,
            dataset_name=dataset_name,
        )

        # Acquisition function — borda is already a rank transform so no UQ needed
        self.acq_fn   = get_metric(acq)
        self.acq_name = acq

        # Initial labeled set
        rng      = np.random.default_rng(seed)
        init_idx = rng.choice(len(self.pool_smiles), init_size, replace=False)

        self.labeled_idx    = set(init_idx.tolist())
        self.labeled_scores = {self.pool_smiles[i]: oracle[self.pool_smiles[i]]
                               for i in init_idx}
        print(f"[init] {init_size} random molecules  best={self._best():.3f} kcal/mol")

    def _best(self) -> float:
        return min(self.labeled_scores.values())

    def _recall(self, k: int) -> float:
        top_k = {s for s, _ in sorted(self.oracle.items(), key=lambda x: x[1])[:k]}
        return sum(1 for s in self.labeled_scores if s in top_k) / k

    def run(self) -> list[dict]:
        history = []
        n = len(self.pool_smiles)

        for rnd in range(self.n_rounds):
            t0 = time.perf_counter()

            # Labeled data (negated scores so surrogate maximises)
            idx  = list(self.labeled_idx)
            xs   = [self.pool_smiles[i] for i in idx]
            ys   = self._sign * np.array(
                [self.labeled_scores[self.pool_smiles[i]] for i in idx], dtype=np.float32
            )

            self.model.train(xs, ys)

            # Predict unlabeled pool
            mask     = np.ones(n, bool)
            for i in self.labeled_idx:
                mask[i] = False
            pool_idx = np.where(mask)[0]
            pool_smi = [self.pool_smiles[i] for i in pool_idx]

            mu, var  = self.model.get_means_and_vars(pool_smi)
            sigma    = np.sqrt(np.maximum(var, 0.0))

            # Acquisition
            if self.acq_name in ("ucb", "lcb", "thompson", "ts", "ei", "pi"):
                scores = self.acq_fn(mu, var)
            elif self.acq_name in ("greedy", "noisy", "borda", "threshold"):
                scores = self.acq_fn(mu)
            else:
                scores = self.acq_fn(mu)

            top_local = np.argsort(scores)[::-1][: self.batch_size]
            selected  = pool_idx[top_local]

            for i in selected:
                smi = self.pool_smiles[i]
                self.labeled_scores[smi] = self.oracle[smi]
                self.labeled_idx.add(int(i))

            k1pct   = max(1, n // 100)
            recall  = self._recall(k1pct)
            elapsed = time.perf_counter() - t0

            print(
                f"  Round {rnd+1:02d}/{self.n_rounds}  "
                f"labeled={len(self.labeled_idx):,}  "
                f"best={self._best():.3f} kcal/mol  "
                f"top-1% recall={recall:.1%}  "
                f"({elapsed:.1f}s)"
            )

            record = dict(
                round=rnd + 1,
                n_labeled=len(self.labeled_idx),
                best_score=float(self._best()),
                top1pct_recall=float(recall),
                elapsed=round(elapsed, 2),
            )
            history.append(record)
            self._checkpoint(rnd + 1, record)

        self._save_final(history)
        return history

    def _checkpoint(self, rnd: int, record: dict):
        d = self.run_dir / f"iter_{rnd}"
        d.mkdir(exist_ok=True)
        (d / "state.json").write_text(json.dumps(record, indent=2))
        with open(d / "scores.pkl", "wb") as f:
            pickle.dump(dict(self.labeled_scores), f)

    def _save_final(self, history: list):
        pd.DataFrame(
            sorted(self.labeled_scores.items(), key=lambda x: x[1]),
            columns=["smiles", "score"],
        ).to_csv(self.run_dir / "all_explored_final.csv", index=False)
        (self.run_dir / "history.json").write_text(json.dumps(history, indent=2))
        print(f"\n[done] results → {self.run_dir}")


# ==============================================================================
# MOLPAL ACTIVE LEARNING LOOP  (fingerprint-based PretrainedAL models)
# ==============================================================================

class MolPALExplorer:
    """Thin driver for PretrainedAL-VS models on molecular fingerprints."""

    def __init__(
        self,
        pool_smiles: list,
        oracle: dict,
        model_type: str,
        acq: str,
        conf_method: str = "none",
        fingerprint: str = "morgan",
        radius: int = 2,
        length: int = 2048,
        init_size: int = 200,
        batch_size: int = 100,
        n_rounds: int = 10,
        run_dir: Path = None,
        seed: int = 42,
    ):
        from molpal.models import model as build_model
        from molpal.featurizer import Featurizer
        from molpal.acquirer.metrics import get_metric

        self.pool_smiles = np.array(pool_smiles)
        self.oracle      = oracle
        self.batch_size  = batch_size
        self.n_rounds    = n_rounds
        self.run_dir     = run_dir or RUNS_DIR / "molpal_run"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._sign = -1.0

        self.featurizer = Featurizer(fingerprint=fingerprint, radius=radius, length=length)
        self.model = build_model(
            model=model_type,
            conf_method=conf_method,
            input_size=length,
            test_batch_size=4096,
        )

        self.acq_fn   = get_metric(acq)
        self.acq_name = acq
        self.needs_var = acq in ("ucb", "lcb", "thompson", "ts", "ei", "pi")

        rng      = np.random.default_rng(seed)
        init_idx = rng.choice(len(self.pool_smiles), init_size, replace=False)

        self.labeled_idx    = set(init_idx.tolist())
        self.labeled_scores = {self.pool_smiles[i]: oracle[self.pool_smiles[i]]
                               for i in init_idx}
        print(f"[init] {init_size} random molecules  best={self._best():.3f} kcal/mol")

    def _best(self) -> float:
        return min(self.labeled_scores.values())

    def _recall(self, k: int) -> float:
        top_k = {s for s, _ in sorted(self.oracle.items(), key=lambda x: x[1])[:k]}
        return sum(1 for s in self.labeled_scores if s in top_k) / k

    def _featurize(self, smis: list) -> np.ndarray:
        feats = [self.featurizer(s) for s in smis]
        valid = [f if f is not None else np.zeros(len(self.featurizer)) for f in feats]
        return np.stack(valid)

    def run(self) -> list[dict]:
        history = []
        n = len(self.pool_smiles)

        for rnd in range(self.n_rounds):
            t0 = time.perf_counter()

            idx  = list(self.labeled_idx)
            xs   = [self.pool_smiles[i] for i in idx]
            ys   = self._sign * np.array(
                [self.labeled_scores[self.pool_smiles[i]] for i in idx], dtype=np.float32
            )

            self.model.train(xs, ys, featurizer=self.featurizer)

            mask     = np.ones(n, bool)
            for i in self.labeled_idx:
                mask[i] = False
            pool_idx = np.where(mask)[0]
            pool_smi = [self.pool_smiles[i] for i in pool_idx]

            if self.needs_var:
                mu, var = self.model.get_means_and_vars(pool_smi)
                scores  = self.acq_fn(mu, var)
            else:
                mu     = self.model.get_means(pool_smi)
                scores = self.acq_fn(mu)

            top_local = np.argsort(scores)[::-1][: self.batch_size]
            selected  = pool_idx[top_local]

            for i in selected:
                smi = self.pool_smiles[i]
                self.labeled_scores[smi] = self.oracle[smi]
                self.labeled_idx.add(int(i))

            k1pct   = max(1, n // 100)
            recall  = self._recall(k1pct)
            elapsed = time.perf_counter() - t0

            print(
                f"  Round {rnd+1:02d}/{self.n_rounds}  "
                f"labeled={len(self.labeled_idx):,}  "
                f"best={self._best():.3f} kcal/mol  "
                f"top-1% recall={recall:.1%}  "
                f"({elapsed:.1f}s)"
            )

            record = dict(
                round=rnd + 1,
                n_labeled=len(self.labeled_idx),
                best_score=float(self._best()),
                top1pct_recall=float(recall),
                elapsed=round(elapsed, 2),
            )
            history.append(record)
            self._checkpoint(rnd + 1, record)

        self._save_final(history)
        return history

    def _checkpoint(self, rnd: int, record: dict):
        d = self.run_dir / f"iter_{rnd}"
        d.mkdir(exist_ok=True)
        (d / "state.json").write_text(json.dumps(record, indent=2))
        with open(d / "scores.pkl", "wb") as f:
            pickle.dump(dict(self.labeled_scores), f)

    def _save_final(self, history: list):
        pd.DataFrame(
            sorted(self.labeled_scores.items(), key=lambda x: x[1]),
            columns=["smiles", "score"],
        ).to_csv(self.run_dir / "all_explored_final.csv", index=False)
        (self.run_dir / "history.json").write_text(json.dumps(history, indent=2))
        print(f"\n[done] results → {self.run_dir}")


# ==============================================================================
# CLI
# ==============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="FusionAL — combined MVE+MolPAL active learning framework",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── shared ────────────────────────────────────────────────────────────────
    p.add_argument("--dataset", default="Enamine10k",
                   choices=["Enamine10k", "Enamine50k", "EnamineHTS"])
    p.add_argument("--mode",    default="mve",
                   choices=["mve", "molpal"],
                   help="mve = ALSU surrogate on embeddings; molpal = fingerprint-based model")
    p.add_argument("--acq",     default="ucb",
                   choices=["ucb", "lcb", "greedy", "noisy", "thompson", "ts",
                            "ei", "pi", "threshold", "random", "borda"],
                   help="Acquisition function")
    p.add_argument("--init-size",  type=int, default=200)
    p.add_argument("--batch-size", type=int, default=100)
    p.add_argument("--n-rounds",   type=int, default=10)
    p.add_argument("--seed",       type=int, default=42)
    p.add_argument("--run-dir",    default=None,
                   help="Output directory (auto-named if not set)")

    # ── MVE-specific ──────────────────────────────────────────────────────────
    mve_grp = p.add_argument_group("MVE surrogate options (--mode mve)")
    mve_grp.add_argument("--surrogate", default="single",
                         choices=["single", "lightweight", "bigfusion", "ensemble",
                                  "learned", "nonlinear", "attention", "oof",
                                  "ft_molformer", "ft_grover", "ft_unimol", "ft_fusion"],
                         help="ALSU surrogate variant")
    mve_grp.add_argument("--backbone",  default="molformer",
                         choices=["molformer", "grover", "unimol"],
                         help="Backbone for --surrogate single")
    mve_grp.add_argument("--backbones", nargs="+",
                         default=["molformer"],
                         choices=["molformer", "grover", "unimol"],
                         help="Backbones to load (multi-backbone surrogates use all of them)")
    mve_grp.add_argument("--epochs", type=int, default=150,
                         help="Surrogate training epochs per round")

    # ── MolPAL-specific ───────────────────────────────────────────────────────
    mp_grp = p.add_argument_group("MolPAL model options (--mode molpal)")
    mp_grp.add_argument("--model", default="rf",
                        choices=["rf", "gp", "nn", "mpn", "lgbm", "transformer", "molclr", "random"],
                        help="Fingerprint-based model type")
    mp_grp.add_argument("--conf-method", default="none",
                        choices=["none", "dropout", "ensemble", "mve", "twooutput"],
                        help="Uncertainty quantification method for NN/MPN/Transformer")
    mp_grp.add_argument("--fingerprint", default="morgan",
                        choices=["morgan", "pair", "rdkit", "maccs"])
    mp_grp.add_argument("--radius", type=int, default=2)
    mp_grp.add_argument("--length", type=int, default=2048)

    return p.parse_args()


def main():
    args = parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    print(f"\n{'='*60}")
    print(f"  FusionAL  |  mode={args.mode}  dataset={args.dataset}  acq={args.acq}")
    print(f"  device={DEVICE}  seed={args.seed}")
    print(f"{'='*60}\n")

    oracle = load_oracle(args.dataset)

    if args.mode == "mve":
        # Determine which backbones to load; for single-backbone surrogate,
        # use --backbone; for multi-backbone, use --backbones.
        if args.surrogate == "single":
            backbones = [args.backbone]
        else:
            backbones = args.backbones

        emb_dict, pool_smiles = load_embeddings(args.dataset, backbones)

        # Restrict pool to molecules that have both an embedding and an oracle score
        usable    = [s for s in pool_smiles if s in oracle]
        usable_set = set(usable)
        emb_dict  = {bb: emb[np.array([i for i, s in enumerate(pool_smiles) if s in usable_set])]
                     for bb, emb in emb_dict.items()}
        pool_smiles = usable
        print(f"[pool] {len(pool_smiles):,} molecules with embeddings + oracle scores")

        suffix = f"mve_{args.dataset}_{'_'.join(backbones)}_{args.surrogate}_{args.acq}"
        run_dir = Path(args.run_dir) if args.run_dir else RUNS_DIR / suffix

        explorer = MVEExplorer(
            emb_dict       = emb_dict,
            pool_smiles    = pool_smiles,
            oracle         = oracle,
            surrogate_type = args.surrogate,
            backbone       = args.backbone,
            acq            = args.acq,
            init_size      = args.init_size,
            batch_size     = args.batch_size,
            n_rounds       = args.n_rounds,
            dataset_name   = args.dataset,
            run_dir        = run_dir,
            seed           = args.seed,
        )

    else:  # molpal mode
        pool_smiles = load_library_smiles(args.dataset)
        usable      = [s for s in pool_smiles if s in oracle]
        pool_smiles = usable
        print(f"[pool] {len(pool_smiles):,} molecules with oracle scores")

        suffix = f"molpal_{args.dataset}_{args.model}_{args.acq}"
        run_dir = Path(args.run_dir) if args.run_dir else RUNS_DIR / suffix

        explorer = MolPALExplorer(
            pool_smiles  = pool_smiles,
            oracle       = oracle,
            model_type   = args.model,
            acq          = args.acq,
            conf_method  = args.conf_method,
            fingerprint  = args.fingerprint,
            radius       = args.radius,
            length       = args.length,
            init_size    = args.init_size,
            batch_size   = args.batch_size,
            n_rounds     = args.n_rounds,
            run_dir      = run_dir,
            seed         = args.seed,
        )

    history = explorer.run()

    print(f"\n{'Round':>6} {'Labeled':>8} {'Best (kcal/mol)':>16} {'Top-1% recall':>14}")
    print("-" * 52)
    for r in history:
        print(f"{r['round']:>6} {r['n_labeled']:>8,} "
              f"{r['best_score']:>16.3f} {r['top1pct_recall']:>13.1%}")


if __name__ == "__main__":
    main()
