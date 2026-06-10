# FusionAL

A unified active learning framework for molecular docking campaigns, combining:

- **ALSU** — 10 surrogate variants with MVE+Spearman loss and scheduled backbone fine-tuning
- **PretrainedAL-VS** — modular pipeline with pluggable models, rich acquisition functions, and flexible pool management

Runs locally on a single GPU. No SLURM or distributed computing required.

---

## Overview

FusionAL iterates an active learning loop:

```
Initial random batch
       ↓
Surrogate fit (embeddings → μ, σ²)
       ↓
Acquisition function selects next batch
       ↓
Oracle labels batch (docking scores)
       ↓
repeat N rounds
```

**MVE mode** uses pre-extracted backbone embeddings (Grover, UniMol, MoLFormer) fed into one of 10 ALSU surrogate variants. Scheduled fine-tuning surrogates additionally update backbone weights each round on the accumulated labeled set.

**MolPAL mode** uses fingerprint-based models (RF, GP, NN, MPN, Transformer, MolCLR) from PretrainedAL-VS — no pre-extraction step needed.

---

## Repository Layout

```
FusionAL/
├── run_al.py                    # unified CLI — start here
├── extract_embeddings.py        # one-time backbone embedding extraction
├── preprocess_unimol.py         # one-time UniMol 3D conformer generation
├── surrogates.py                # 10 ALSU surrogate classes
├── backbone_finetuner.py        # BackboneFinetuner (Grover / UniMol / MoLFormer)
├── molpal/
│   ├── models/
│   │   ├── mvemodels.py         # EmbeddingMVEModel — bridge between ALSU and molpal
│   │   ├── losses.py            # MVE (Gaussian NLL) + soft Spearman loss
│   │   ├── base.py              # Model ABC
│   │   ├── nnmodels.py          # NN fingerprint models
│   │   ├── mpnmodels.py         # Message-passing network
│   │   ├── sklmodels.py         # RF, GP (sklearn)
│   │   ├── transformermodels.py # MoLFormer fine-tuned model
│   │   └── molclrmodels.py      # MolCLR contrastive model
│   ├── acquirer/
│   │   └── metrics.py           # UCB, EI, PI, TS, greedy, threshold, borda, …
│   ├── featurizer.py            # EmbeddingFeaturizer + Morgan fingerprints
│   ├── explorer.py              # AL loop orchestration
│   ├── pools/                   # molecule pool management
│   └── objectives/              # oracle / scoring functions
├── models/                      # pretrained backbone weights (symlink → ALSU/models)
├── data/                        # oracle score CSVs (symlink → ALSU/data)
├── muben/                       # MUBen backbone library (symlink → ALSU/muben)
├── molpal/libraries/            # molecule libraries (symlink → PretrainedAL-VS/libraries)
└── results/embed/               # pre-extracted backbone embeddings (.npz)
```

---

## Surrogate Variants

| `--surrogate`       | Description                                                                 |
|---------------------|-----------------------------------------------------------------------------|
| `single`            | One backbone, dual MVE head (μ + σ²), Gaussian NLL + Spearman loss          |
| `lightweight`       | Concatenated embeddings → lightweight MVE head                              |
| `bigfusion`         | Three independent single-backbone surrogates → Borda count fusion          |
| `ensemble`          | Ensemble of MVE heads, one per backbone                                     |
| `learned`           | Learnable per-backbone weighting before shared MVE head                     |
| `nonlinear`         | Nonlinear backbone projections before fusion                                |
| `attention`         | Cross-backbone attention fusion                                             |
| `oof`               | Out-of-fold Ridge regression on backbone embeddings                        |
| `ft_molformer`      | MoLFormer: Phase 1 frozen → Phase 2 online backbone fine-tuning            |
| `ft_grover`         | Grover: Phase 1 frozen → Phase 2 online backbone fine-tuning               |
| `ft_unimol`         | UniMol: Phase 1 frozen → Phase 2 online backbone fine-tuning               |
| `ft_fusion`         | All three backbones fine-tuned jointly                                      |

---

## Acquisition Functions

All 9 PretrainedAL-VS metrics plus Borda count:

`greedy`, `ucb`, `ei`, `pi`, `ts` (Thompson sampling), `threshold`, `random`, `cluster`, `diversity`, `borda`

---

## Installation (GPU)

FusionAL requires Python 3.10, PyTorch (GPU), RDKit, and the PyTorch Geometric stack. Install in three steps.

### 1. Create the conda environment

```bash
conda env create -f environment.yml -n py310
conda activate py310
```

This installs Python 3.10, RDKit, PyTorch + CUDA 12.1, and all pip-based dependencies.

### 2. Install PyTorch Geometric (CUDA 12.1 wheels)

PyTorch Geometric must be installed manually because it requires CUDA-specific wheels.

```bash
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv \
    -f https://data.pyg.org/whl/torch-2.1.0+cu121.html

pip install torch_geometric
```

### 3. Verify installation

```bash
python - << 'EOF'
import torch, rdkit, torch_geometric
print("CUDA available:", torch.cuda.is_available())
print("RDKit OK")
print("PyG OK")
EOF
```

Expected output:

```
CUDA available: True
RDKit OK
PyG OK
```

---

## Setup

### Activate the environment

```bash
conda activate py310
```

### Symlinks (already configured)

```bash
# These symlinks should already exist:
ls -la /home/jmeng/repos/FusionAL/data              # → ALSU/data
ls -la /home/jmeng/repos/FusionAL/models            # → ALSU/models
ls -la /home/jmeng/repos/FusionAL/muben             # → ALSU/muben
ls -la /home/jmeng/repos/FusionAL/molpal/libraries  # → PretrainedAL-VS/libraries
```

---

## Quickstart

### Step 1 — Extract backbone embeddings (one time per dataset)

```bash
cd /home/jmeng/repos/FusionAL

# Extract all three backbones for Enamine50k
python extract_embeddings.py --dataset Enamine50k --backbone all

# Or a single backbone
python extract_embeddings.py --dataset Enamine50k --backbone grover
```

Output: `results/embed/Enamine50k/{grover,unimol,molformer}_embeddings.npz`

Each `.npz` contains two arrays:
- `embeddings` — shape `(N, D)` float32
- `smiles` — shape `(N,)` string array for alignment verification

### Step 2 — Preprocess UniMol conformers (required only for `ft_unimol`)

```bash
python preprocess_unimol.py --dataset Enamine50k --workers 4
```

Saves 3D RDKit conformers to `muben/data/files/Enamine50k/processed/unimol-unimol/train.pt`.
Takes ~10–40 min on CPU; run once and reuse.

### Step 3 — Run active learning

```bash
# Single backbone, UCB
python run_al.py --mode mve --dataset Enamine50k \
    --surrogate single --backbones molformer \
    --acq ucb --init-size 500 --batch-size 500 --n-rounds 5

# Multi-backbone BigFusion with Borda count
python run_al.py --mode mve --dataset Enamine50k \
    --surrogate bigfusion --backbones grover molformer unimol \
    --acq borda --init-size 500 --batch-size 500 --n-rounds 5

# Grover online fine-tuning
python run_al.py --mode mve --dataset Enamine50k \
    --surrogate ft_grover --backbones grover \
    --acq ucb --init-size 500 --batch-size 500 --n-rounds 5

# UniMol online fine-tuning (requires Step 2 above)
python run_al.py --mode mve --dataset Enamine50k \
    --surrogate ft_unimol --backbones unimol \
    --acq ucb --init-size 500 --batch-size 500 --n-rounds 5

# Fingerprint RF (no embedding extraction needed)
python run_al.py --mode molpal --dataset Enamine50k \
    --model rf --acq greedy --init-size 500 --batch-size 500 --n-rounds 5
```

---

## CLI Reference

### `run_al.py`

| Flag | Default | Description |
|------|---------|-------------|
| `--mode` | `mve` | `mve` (embedding surrogates) or `molpal` (fingerprint models) |
| `--dataset` | `Enamine50k` | Dataset name; must exist in `data/` and `molpal/libraries/` |
| `--surrogate` | `single` | MVE surrogate type (see table above); MVE mode only |
| `--backbones` | `molformer` | Space-separated backbone names for embedding loading |
| `--model` | `rf` | MolPAL model type (`rf`, `gp`, `nn`, `mpn`, `transformer`, `molclr`); MolPAL mode only |
| `--acq` | `ucb` | Acquisition function |
| `--init-size` | `500` | Number of molecules in the initial random batch |
| `--batch-size` | `500` | Molecules selected per round |
| `--n-rounds` | `5` | Number of AL rounds |
| `--seed` | `42` | Random seed |
| `--run-dir` | auto | Output directory; auto-named if omitted |

### `extract_embeddings.py`

| Flag | Default | Description |
|------|---------|-------------|
| `--dataset` | `Enamine50k` | `Enamine10k`, `Enamine50k`, or `EnamineHTS` |
| `--backbone` | `all` | `all`, `grover`, `unimol`, or `molformer` |

### `preprocess_unimol.py`

| Flag | Default | Description |
|------|---------|-------------|
| `--dataset` | `Enamine50k` | Dataset to preprocess |
| `--workers` | `4` | Parallel workers for RDKit conformer generation |

---

## Output

Each run writes to `runs/<run_name>/`:

```
runs/mve_Enamine50k_grover_ucb/
├── history.json        # per-round metrics (labeled count, best score, recall)
└── scores_rdX.csv      # acquisition scores at each round (optional)
```

Console output per round:
```
Round 03/5  labeled=2,000  best=-9.900 kcal/mol  top-1% recall=36.4%  (301.1s)
```

---

## Online Fine-tuning (Scheduled Surrogates)

`ft_grover`, `ft_unimol`, `ft_molformer` follow a two-phase schedule:

- **Rounds 1–2** (frozen): backbone weights frozen, only the MVE head trains. Fast (~10s/round).
- **Rounds 3+** (fine-tune): `BackboneFinetuner` updates backbone weights on the accumulated labeled set for 10 epochs, then re-extracts embeddings for the entire pool. Slower (~3–5 min/round for 2–3k labeled molecules).

The updated embeddings are written back into the shared `emb_dict` in-place, so `EmbeddingMVEModel._get_X()` automatically uses fresh representations in the next round without any extra wiring.

---

## Verified Results (Enamine50k, 5 rounds, UCB)

| Surrogate | Round 1 | Round 5 | Fine-tune time/round |
|-----------|---------|---------|----------------------|
| `single` (MoLFormer) | 17.3% | ~35% | — |
| `bigfusion` (all 3) | ~18% | ~38% | — |
| `ft_grover` | 17.3% | 39.4% | ~3–5 min |

Top-1% recall: fraction of the true top-1% hits found in the labeled set.

---

## Architecture Notes

**`EmbeddingMVEModel`** ([molpal/models/mvemodels.py](molpal/models/mvemodels.py)) is the central bridge. It:
1. Maps SMILES → row indices into the pre-extracted embedding matrices
2. Calls `surrogate.fit(X, y)` (and passes `labeled_smiles` for scheduled fine-tuning)
3. Propagates the `embeddings_refreshed` flag so Explorer knows when in-place embedding updates happened
4. Implements the `Model` ABC (`train`, `get_means`, `get_means_and_vars`, `save`, `load`)

**Shared `emb_dict` reference**: for `ft_grover`/`ft_unimol`, the surrogate and model share the same `single_emb_dict = {bb: array}` object. When `BackboneFinetuner` replaces the array in-place (`single_emb_dict[bb] = new_emb`), `_get_X()` immediately reads the updated embeddings — no copy or signal needed.
