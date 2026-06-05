"""
backbone_finetuner.py
Online backbone finetuner for active learning.

At each AL round this wraps a MUBen backbone (GROVER, UniMol, MoLFormer)
with a lightweight regression head, finetunes on the current labeled set,
then re-extracts embeddings for the full pool.

Usage
-----
finetuner = BackboneFinetuner("unimol", "Enamine50k", pool_smiles, model_zoo)

# before each AL round:
finetuner.finetune(labeled_smiles, labeled_scores, n_epochs=10)
new_embeddings = finetuner.extract_pool_embeddings()   # (N, D) float32
"""

import sys
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from torch.utils.data import DataLoader
from torch.optim import AdamW

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent
MODEL_ZOO  = ROOT / "models"
RESULTS_DIR = ROOT / "results" / "embed"

# muben is a subdirectory of the repo, not an installed package.
# The actual Python package lives at ALSU/muben/muben/, so we need
# ALSU/muben/ on sys.path for "import muben" to resolve correctly.
_MUBEN_ROOT = ROOT / "muben"
if str(_MUBEN_ROOT) not in sys.path:
    sys.path.insert(0, str(_MUBEN_ROOT))

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── SMILES injection ───────────────────────────────────────────────────────────

def _inject_smiles(smiles_list: list, scores: np.ndarray = None):
    """
    Monkey-patch muben Dataset.read_csv to inject a specific SMILES list.
    scores, if given, becomes the labels (shape N); otherwise dummy zeros.
    """
    n = len(smiles_list)
    lbs = (scores.reshape(n, 1).astype(np.float32)
           if scores is not None
           else np.zeros((n, 1), dtype=np.float32))

    def _read_csv(self, data_dir, partition):
        self._smiles  = list(smiles_list)
        self._lbs     = lbs.copy()
        self._masks   = np.ones((n, 1), dtype=np.float32)
        self._ori_ids = None
        return self

    import muben.dataset.dataset as _ds
    _ds.Dataset.read_csv = _read_csv


# ── Minimal muben config ───────────────────────────────────────────────────────

class _MubenConfig:
    """Minimal attribute bag that satisfies muben dataset/model constructors."""

    def __init__(self, model_name: str, model_zoo: Path, dataset_name: str):
        self.model_name   = model_name
        self.dataset_name = dataset_name
        self.data_dir     = str(ROOT / "muben" / "data" / "files" / dataset_name)
        self.checkpoint_path         = ""
        self.unimol_feature_dir      = str(RESULTS_DIR / dataset_name)
        self.num_preprocess_workers  = 4
        self.ignore_preprocessed_dataset = False
        self.disable_dataset_saving      = False
        self.disable_checkpoint_loading  = False
        self.feature_type = "none"

        # GROVER
        self.hidden_size           = 128
        self.dropout               = 0.0
        self.bias                  = False
        self.num_mt_block          = 1
        self.num_attn_head         = 4
        self.embedding_output_type = "both"
        self.ffn_num_layers        = 2
        self.ffn_hidden_size       = 128
        self.activation            = "ReLU"

        # UniMol
        self.max_atoms               = 64
        self.max_seq_len             = 80
        self.only_polar_hydrogens    = False
        self.remove_hydrogen         = True
        self.remove_polar_hydrogen   = False
        self.encoder_embed_dim       = 512
        self.encoder_layers          = 15
        self.encoder_attention_heads = 64
        self.encoder_ffn_embed_dim   = 2048
        self.activation_fn           = "gelu"
        self.pooler_stride           = 1
        self.pooler_dropout          = 0.0
        self.emb_dropout             = 0.1
        self.attention_dropout       = 0.1
        self.activation_dropout      = 0.0
        self.delta_pair_repr_norm_loss = -1
        self.masked_coord_loss       = 0.0
        self.masked_dist_loss        = 0.0
        self.masked_type_loss        = 0.0
        self.pooler_activation_fn    = "Tanh"

        # Task (shared)
        self.uncertainty_method  = "none"
        self.task_type           = "regression"
        self.bbp_prior_sigma     = 0.5
        self.n_lbs               = 1
        self.n_tasks             = 1

        # Backbone-specific checkpoint
        if model_name == "grover":
            self.checkpoint_path = str(model_zoo / "grover" / "grover_base.pt")
        elif model_name == "unimol":
            self.checkpoint_path = str(model_zoo / "unimol" / "mol_pre_all_h_220816.pt")
            self.feature_type    = "unimol"


# ── BackboneFinetuner ──────────────────────────────────────────────────────────

class BackboneFinetuner:
    """
    Wraps a MUBen backbone with a regression head for online AL finetuning.

    Parameters
    ----------
    backbone     : "grover" | "unimol" | "molformer"
    dataset_name : e.g. "Enamine50k"
    pool_smiles  : ordered list of all pool SMILES (defines embedding row order)
    model_zoo    : path to pre-trained model weights directory
    """

    def __init__(
        self,
        backbone: str,
        dataset_name: str,
        pool_smiles: list,
        model_zoo: Path = None,
    ):
        self.backbone     = backbone
        self.pool_smiles  = list(pool_smiles)
        self._smi2idx     = {s: i for i, s in enumerate(pool_smiles)}
        self._model_zoo   = model_zoo or MODEL_ZOO
        self._dataset_name = dataset_name

        self._model        = None
        self._head         = None
        self._collator     = None
        self._pool_dataset = None
        self._emb_dim      = None
        self._tokenizer    = None   # MoLFormer only
        self._get_emb      = None   # backbone-specific embedding fn

        print(f"[BackboneFinetuner] Loading {backbone} backbone…")
        setup = getattr(self, f"_setup_{backbone}")
        setup(dataset_name, pool_smiles, self._model_zoo)
        self._head = nn.Linear(self._emb_dim, 1).to(DEVICE)
        print(f"[BackboneFinetuner] Ready.  emb_dim={self._emb_dim}  device={DEVICE}")

    # ── Setup (one per backbone) ───────────────────────────────────────────────

    def _setup_grover(self, dataset_name, pool_smiles, model_zoo):
        from muben.dataset import DatasetGrover
        from muben.dataset.dataset_grover import CollatorGrover
        from muben.model import GROVER

        _inject_smiles(pool_smiles)
        cfg = _MubenConfig("grover", model_zoo, dataset_name)

        self._pool_dataset = DatasetGrover().prepare(config=cfg, partition="train")
        self._collator     = CollatorGrover(cfg)
        self._model        = GROVER(cfg).to(DEVICE)
        self._get_emb      = self._emb_grover

        # Probe actual output dim — checkpoint hidden size may differ from cfg.hidden_size
        with torch.no_grad():
            _probe = self._collator([self._pool_dataset[0]])
            _probe.to(DEVICE)
            self._emb_dim = int(self._emb_grover(_probe).shape[-1])

    def _setup_unimol(self, dataset_name, pool_smiles, model_zoo):
        # Loads only the model weights + collator — no 50k conformer dataset.
        # Conformers for fine-tuning are generated on-the-fly per labeled batch.
        # The full pool conformer cache is lazy-loaded on the first
        # extract_pool_embeddings() call and kept in memory for subsequent rounds.
        from muben.dataset.dataset_unimol import CollatorUniMol
        from muben.dataset.dataset_unimol.dictionary import DictionaryUniMol
        from muben.dataset.dataset_unimol.process import ProcessingPipeline
        from muben.model.unimol.unimol import UniMol

        cfg = _MubenConfig("unimol", model_zoo, dataset_name)

        d = DictionaryUniMol.load()
        d.add_symbol("[MASK]", is_special=True)
        collator = CollatorUniMol(cfg, d)
        pad_idx  = d.pad()
        collator._atom_pad_idx = pad_idx
        collator.pad_idx       = pad_idx
        collator.atom_pad_idx  = pad_idx
        self._collator = collator

        # Processing pipeline — used to convert raw atoms/coords → model input dicts.
        # process_training picks ONE random conformer → shape (1, seq_len) per item.
        # process_inference stacks ALL conformers → shape (n_conf, seq_len) per item.
        # _emb_unimol expects (B, seq_len) so we always use process_training here.
        pipeline = ProcessingPipeline(
            dictionary=d,
            max_atoms=cfg.max_atoms,
            max_seq_len=cfg.max_seq_len,
            remove_hydrogen_flag=cfg.remove_hydrogen,
            remove_polar_hydrogen_flag=cfg.remove_polar_hydrogen,
        )
        self._unimol_process = pipeline.process_training

        self._unimol_cfg  = cfg
        self._pool_dataset = None   # lazy-loaded on first extract_pool_embeddings()

        self._model   = UniMol(config=cfg, dictionary=d).to(DEVICE)
        self._emb_dim = cfg.encoder_embed_dim   # 512
        self._get_emb = self._emb_unimol

    def _setup_molformer(self, dataset_name, pool_smiles, model_zoo):
        from transformers import AutoModel, AutoTokenizer

        path = str(model_zoo / "molformer")
        self._tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        self._model     = AutoModel.from_pretrained(path, trust_remote_code=True).to(DEVICE)

        # Probe embedding dim
        with torch.no_grad():
            enc  = self._tokenizer(["CC"], return_tensors="pt",
                                   truncation=True).to(DEVICE)
            out  = self._model(**enc)
            probe = (out.pooler_output if out.pooler_output is not None
                     else out.last_hidden_state[:, 0])
            self._emb_dim = probe.shape[-1]

    # ── Per-backbone embedding forward passes ──────────────────────────────────

    def _emb_grover(self, batch) -> torch.Tensor:
        components = batch.molecule_graphs.components
        _, _, _, _, _, a_scope, _, _ = components
        out = self._model.grover(components)
        return torch.cat([
            self._model.readout(out["atom_from_bond"], a_scope),
            self._model.readout(out["atom_from_atom"], a_scope),
        ], dim=1)

    def _emb_unimol(self, batch) -> torch.Tensor:
        src_tokens, src_distance, src_edge_type = (
            batch.atoms, batch.distances, batch.edge_types
        )
        pad_mask = src_tokens.eq(self._model.padding_idx)
        if not pad_mask.any():
            pad_mask = None

        x   = self._model.embed_tokens(src_tokens)
        n   = src_distance.size(-1)
        gbf = self._model.gbf_proj(self._model.gbf(src_distance, src_edge_type))
        attn_bias = gbf.permute(0, 3, 1, 2).contiguous().view(-1, n, n)

        rep, *_ = self._model.encoder(x, padding_mask=pad_mask, attn_mask=attn_bias)
        return self._model.hidden_layer(rep[:, 0, :])

    def _emb_molformer(self, smiles_batch: list) -> torch.Tensor:
        enc = self._tokenizer(
            smiles_batch, padding=True, truncation=True, return_tensors="pt"
        ).to(DEVICE)
        out = self._model(**enc)
        return (out.pooler_output if out.pooler_output is not None
                else out.last_hidden_state.mean(dim=1))

    # ── Finetuning ─────────────────────────────────────────────────────────────

    def finetune(
        self,
        labeled_smiles: list,
        labeled_scores: np.ndarray,
        n_epochs: int       = 10,
        batch_size: int     = 32,
        lr_backbone: float  = 1e-5,
        lr_head: float      = 1e-4,
    ):
        """
        Finetune backbone + regression head on the currently labeled molecules.

        Parameters
        ----------
        labeled_smiles : SMILES strings for molecules with known scores
        labeled_scores : corresponding docking scores (lower = better)
        n_epochs       : gradient steps per round (keep small, e.g. 5-20)
        lr_backbone    : learning rate for backbone (small to prevent forgetting)
        lr_head        : learning rate for regression head
        """
        n = len(labeled_smiles)
        print(f"[BackboneFinetuner] Finetuning on {n} labeled molecules, "
              f"{n_epochs} epochs, lr_backbone={lr_backbone}, lr_head={lr_head}")

        y = np.array(labeled_scores, dtype=np.float32)
        y_mean = float(y.mean())
        y_std  = float(y.std()) + 1e-8
        y_norm = (y - y_mean) / y_std

        opt = AdamW([
            {"params": self._model.parameters(), "lr": lr_backbone},
            {"params": self._head.parameters(),  "lr": lr_head},
        ])

        if self.backbone == "molformer":
            self._ft_molformer(labeled_smiles, y_norm, n_epochs, batch_size, opt)
        elif self.backbone == "unimol":
            self._ft_unimol(labeled_smiles, y_norm, n_epochs, batch_size, opt)
        else:
            self._ft_muben(labeled_smiles, y_norm, n_epochs, batch_size, opt)

    def _ft_muben(self, labeled_smiles, y_norm, n_epochs, batch_size, opt):
        """Finetune loop for graph/3D backbones (GROVER). Uses cached pool dataset."""
        labeled_idx = np.array([self._smi2idx[s] for s in labeled_smiles])
        y_t = torch.tensor(y_norm, dtype=torch.float32)

        self._model.train()
        self._head.train()

        for epoch in range(n_epochs):
            perm       = np.random.permutation(len(labeled_idx))
            total_loss = 0.0
            n_steps    = 0

            for start in range(0, len(labeled_idx), batch_size):
                chunk    = perm[start : start + batch_size]
                pool_idx = labeled_idx[chunk]
                targets  = y_t[chunk].to(DEVICE)

                # Collate a batch directly from the cached pool dataset
                items = [self._pool_dataset[int(i)] for i in pool_idx]
                batch = self._collator(items)
                batch.to(DEVICE)

                opt.zero_grad()
                pred = self._head(self._get_emb(batch)).squeeze(-1)
                loss = F.mse_loss(pred, targets)
                loss.backward()
                opt.step()

                total_loss += loss.item()
                n_steps    += 1

            if (epoch + 1) % max(1, n_epochs // 3) == 0:
                logger.info(
                    f"  [finetune] epoch {epoch+1}/{n_epochs}  "
                    f"loss={total_loss / n_steps:.4f}"
                )

    def _ft_unimol(self, labeled_smiles, y_norm, n_epochs, batch_size, opt):
        """Finetune loop for UniMol — generates conformers on-the-fly.

        Pre-generates raw (atoms, coordinates) tuples once per round (fast, ~1-3k
        SMILES via RDKit), then re-samples one conformer per molecule per batch
        step via process_training so each epoch sees different 3D views.
        """
        from muben.utils.chem import smiles_to_coords

        # Generate raw conformers once per round — only 1-3k molecules, fast.
        raw_conformers = []
        for smi in labeled_smiles:
            atoms, coords = smiles_to_coords(smi, n_conformer=10)
            raw_conformers.append((atoms, coords))

        y_t = torch.tensor(y_norm, dtype=torch.float32)
        idx = np.arange(len(labeled_smiles))

        self._model.train()
        self._head.train()

        for epoch in range(n_epochs):
            perm       = np.random.permutation(idx)
            total_loss = 0.0
            n_steps    = 0

            for start in range(0, len(labeled_smiles), batch_size):
                chunk   = perm[start : start + batch_size]
                targets = y_t[chunk].to(DEVICE)

                # Re-sample a conformer per molecule each batch (augmentation).
                # _unimol_process = process_training → conformer_sampling picks
                # a new random conformer each call → shape (1, seq_len) per item.
                items = []
                for i in chunk:
                    atoms, coordinates = raw_conformers[int(i)]
                    a_t, c_t, d_t, e_t = self._unimol_process(
                        atoms=atoms, coordinates=coordinates
                    )
                    items.append({
                        "atoms": a_t, "coordinates": c_t,
                        "distances": d_t, "edge_types": e_t,
                        "lbs":   np.zeros((1,), dtype=np.float32),
                        "masks": np.ones((1,),  dtype=np.float32),
                    })

                batch = self._collator(items)
                batch.to(DEVICE)

                opt.zero_grad()
                pred = self._head(self._get_emb(batch)).squeeze(-1)
                loss = F.mse_loss(pred, targets)
                loss.backward()
                opt.step()

                total_loss += loss.item()
                n_steps    += 1

            if (epoch + 1) % max(1, n_epochs // 3) == 0:
                logger.info(
                    f"  [finetune] epoch {epoch+1}/{n_epochs}  "
                    f"loss={total_loss / n_steps:.4f}"
                )

    def _ft_molformer(self, labeled_smiles, y_norm, n_epochs, batch_size, opt):
        """Finetune loop for MoLFormer (tokenizer-based)."""
        y_t = torch.tensor(y_norm, dtype=torch.float32)
        idx = np.arange(len(labeled_smiles))

        self._model.train()
        self._head.train()

        for epoch in range(n_epochs):
            perm       = np.random.permutation(idx)
            total_loss = 0.0
            n_steps    = 0

            for start in range(0, len(labeled_smiles), batch_size):
                chunk     = perm[start : start + batch_size]
                smi_batch = [labeled_smiles[int(i)] for i in chunk]
                targets   = y_t[chunk].to(DEVICE)

                opt.zero_grad()
                pred = self._head(self._emb_molformer(smi_batch)).squeeze(-1)
                loss = F.mse_loss(pred, targets)
                loss.backward()
                opt.step()

                total_loss += loss.item()
                n_steps    += 1

            if (epoch + 1) % max(1, n_epochs // 3) == 0:
                logger.info(
                    f"  [finetune] epoch {epoch+1}/{n_epochs}  "
                    f"loss={total_loss / n_steps:.4f}"
                )

    # ── Embedding extraction ───────────────────────────────────────────────────

    def extract_pool_embeddings(self, batch_size: int = 64) -> np.ndarray:
        """
        Run the (possibly finetuned) backbone over the full pool.

        Returns
        -------
        numpy array of shape (N, emb_dim), float32, in pool_smiles order.
        """
        print(f"[BackboneFinetuner] Extracting embeddings for "
              f"{len(self.pool_smiles)} molecules…")
        self._model.eval()
        parts = []

        if self.backbone == "molformer":
            with torch.no_grad():
                for i in range(0, len(self.pool_smiles), batch_size):
                    emb = self._emb_molformer(
                        self.pool_smiles[i : i + batch_size]
                    ).float()
                    parts.append(emb.cpu().numpy())
        elif self.backbone == "unimol":
            # Lazy-load the full pool conformer cache on the first extraction.
            # Kept in self._pool_dataset for subsequent rounds (no reload needed).
            if self._pool_dataset is None:
                self._pool_dataset = self._load_unimol_pool_dataset()
            loader = DataLoader(
                self._pool_dataset,
                batch_size  = batch_size,
                shuffle     = False,
                collate_fn  = self._collator,
                num_workers = 0,
            )
            amp_dtype = (torch.bfloat16
                         if (DEVICE.type == "cuda" and torch.cuda.is_bf16_supported())
                         else torch.float16)
            with torch.no_grad():
                for batch in loader:
                    batch.to(DEVICE)
                    with torch.autocast(device_type=DEVICE.type,
                                        dtype=amp_dtype,
                                        enabled=(DEVICE.type == "cuda")):
                        emb = self._get_emb(batch).float()
                    parts.append(emb.cpu().numpy())
        else:
            loader = DataLoader(
                self._pool_dataset,
                batch_size  = batch_size,
                shuffle     = False,
                collate_fn  = self._collator,
                num_workers = 0,
            )
            amp_dtype = (torch.bfloat16
                         if (DEVICE.type == "cuda" and torch.cuda.is_bf16_supported())
                         else torch.float16)
            with torch.no_grad():
                for batch in loader:
                    batch.to(DEVICE)
                    with torch.autocast(device_type=DEVICE.type,
                                        dtype=amp_dtype,
                                        enabled=(DEVICE.type == "cuda")):
                        emb = self._get_emb(batch).float()
                    parts.append(emb.cpu().numpy())

        return np.vstack(parts)

    def _load_unimol_pool_dataset(self):
        """Lazy-load the full pool conformer cache from train.pt for UniMol."""
        import os.path as osp
        from muben.dataset import DatasetUniMol

        cfg = self._unimol_cfg
        preprocessed_path = osp.normpath(osp.join(
            cfg.data_dir, "processed", "unimol-unimol", "train.pt"
        ))

        if not osp.exists(preprocessed_path):
            raise FileNotFoundError(
                f"UniMol conformer cache not found: {preprocessed_path}\n"
                f"Run: python preprocess_unimol.py --dataset {self._dataset_name}"
            )

        print(f"[BackboneFinetuner] Loading UniMol conformer cache ({preprocessed_path})…")
        _inject_smiles(self.pool_smiles)
        dataset = DatasetUniMol()
        # Set up the processing pipeline before loading (it's not serialized in train.pt)
        from muben.dataset.dataset_unimol.process import ProcessingPipeline
        from muben.dataset.dataset_unimol.dictionary import DictionaryUniMol
        d = DictionaryUniMol.load()
        d.add_symbol("[MASK]", is_special=True)
        dataset.processing_pipeline = ProcessingPipeline(
            dictionary=d,
            max_atoms=cfg.max_atoms,
            max_seq_len=cfg.max_seq_len,
            remove_hydrogen_flag=cfg.remove_hydrogen,
            remove_polar_hydrogen_flag=cfg.remove_polar_hydrogen,
        )
        # Must use "training" variant: process_inference stacks ALL n_conf conformers
        # per item → batch shape (B*n_conf, seq_len), but _emb_unimol expects (B, seq_len).
        dataset.set_processor_variant("training")
        dataset._partition = "train"
        dataset.load(preprocessed_path)
        print(f"[BackboneFinetuner] Loaded {len(dataset.atoms):,} conformers.")
        return dataset
