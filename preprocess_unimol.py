#!/usr/bin/env python3
"""
preprocess_unimol.py — Generate and save UniMol 3D conformers for a dataset.

This is a one-time step required before running UniMol fine-tuning (--surrogate
ft_unimol).  It generates RDKit 3D conformers for every molecule in the library
and caches them to:

    muben/data/files/<dataset>/processed/unimol-unimol/train.pt

Once this file exists, BackboneFinetuner(backbone='unimol') loads it instantly
instead of regenerating conformers each time.

Usage
-----
    python preprocess_unimol.py --dataset Enamine50k

This takes ~10-40 minutes on CPU (parallelised over --workers cores) or a few
minutes on GPU.  Run once; subsequent AL experiments reuse the cache.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))                      # FusionAL root on path
sys.path.insert(0, str(ROOT / "muben"))            # muben package on path

MOLPAL_LIB = ROOT / "molpal" / "libraries"
DATA_DIR   = ROOT / "muben" / "data" / "files"
MODEL_ZOO  = ROOT / "models"


# ── helpers ────────────────────────────────────────────────────────────────────

def _make_molpal_reader(smiles_list: list):
    """Return a monkey-patch for Dataset.read_csv that injects our SMILES."""
    def _read_csv(self, data_dir, partition):
        self._smiles  = list(smiles_list)
        self._lbs     = np.zeros(len(smiles_list), dtype=np.float32)
        self._masks   = np.ones(len(smiles_list),  dtype=np.float32)
        self._ori_ids = None
        return self
    return _read_csv


def load_library_smiles(dataset: str) -> list:
    for suffix in (f"{dataset}.csv.gz", f"{dataset}.csv"):
        lib = MOLPAL_LIB / suffix
        if lib.exists():
            df = pd.read_csv(lib)
            df.columns = df.columns.str.strip().str.lower()
            smi_col = next(c for c in df.columns if "smiles" in c)
            smiles = df[smi_col].dropna().tolist()
            print(f"[library] {lib.name}: {len(smiles):,} SMILES")
            return smiles
    raise FileNotFoundError(f"Library not found for '{dataset}' in {MOLPAL_LIB}")


# ── main ───────────────────────────────────────────────────────────────────────

def preprocess(dataset: str, workers: int):
    from muben.dataset import DatasetUniMol
    import muben.dataset.dataset as _ds_module

    smiles = load_library_smiles(dataset)

    # Monkey-patch so muben reads our SMILES instead of a CSV file
    _ds_module.Dataset.read_csv = _make_molpal_reader(smiles)

    unimol_ckpt = str(MODEL_ZOO / "unimol" / "mol_pre_all_h_220816.pt")
    results_dir = ROOT / "results" / "embed" / dataset

    class _Cfg:
        model_name                  = "unimol"
        feature_type                = "unimol"
        data_dir                    = str(DATA_DIR / dataset)
        checkpoint_path             = unimol_ckpt
        unimol_feature_dir          = str(results_dir)
        num_preprocess_workers      = workers
        ignore_preprocessed_dataset = False   # use cache if it exists
        disable_dataset_saving      = False   # save after generation
        disable_checkpoint_loading  = False
        # UniMol architecture
        max_atoms               = 64
        max_seq_len             = 80
        only_polar_hydrogens    = False
        remove_hydrogen         = True
        remove_polar_hydrogen   = False
        encoder_embed_dim       = 512
        encoder_layers          = 15
        encoder_attention_heads = 64
        encoder_ffn_embed_dim   = 2048
        activation_fn           = "gelu"
        pooler_stride           = 1
        pooler_dropout          = 0.0
        emb_dropout             = 0.1
        attention_dropout       = 0.1
        activation_dropout      = 0.0
        delta_pair_repr_norm_loss = -1
        masked_coord_loss       = 0.0
        masked_dist_loss        = 0.0
        masked_type_loss        = 0.0
        pooler_activation_fn    = "Tanh"
        uncertainty_method      = "none"
        task_type               = "regression"
        n_lbs = 1
        n_tasks = 1
        bbp_prior_sigma = 0.5

    cfg = _Cfg()

    # Make sure the output directory exists
    out_dir = Path(cfg.data_dir) / "processed" / "unimol-unimol"
    out_dir.mkdir(parents=True, exist_ok=True)

    cached = out_dir / "train.pt"
    if cached.exists():
        import torch
        existing = torch.load(cached, weights_only=False)
        n_cached = len(existing.get("_smiles", []))
        if n_cached == len(smiles):
            print(f"[cache] {cached} already has {n_cached:,} molecules — nothing to do.")
            return
        else:
            print(f"[cache] {cached} has {n_cached:,} molecules but pool has {len(smiles):,}. Regenerating.")
            cached.unlink()

    print(f"\nGenerating UniMol 3D conformers for {len(smiles):,} molecules...")
    print(f"Workers: {workers}  |  Output: {cached}")
    print("This may take 10-40 minutes depending on hardware.\n")

    t0 = time.perf_counter()
    dataset_obj = DatasetUniMol()
    dataset_obj.prepare(config=cfg, partition="train")
    elapsed = time.perf_counter() - t0

    n = len(dataset_obj)
    print(f"\n[done] {n:,} molecules preprocessed in {elapsed:.1f}s  ({elapsed/60:.1f} min)")
    print(f"       Saved to: {cached}")
    print(f"\nYou can now run ft_unimol experiments:")
    print(f"  python run_al.py --mode mve --dataset {dataset} --surrogate ft_unimol \\")
    print(f"    --backbones unimol --acq ucb --n-rounds 5")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default="Enamine50k",
                   choices=["Enamine10k", "Enamine50k", "EnamineHTS"])
    p.add_argument("--workers", type=int, default=4,
                   help="Parallel workers for conformer generation (default: 4)")
    args = p.parse_args()

    preprocess(args.dataset, args.workers)
