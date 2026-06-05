#!/usr/bin/env python3
"""
preprocess_unimol.py — Generate and save UniMol 3D conformers for a dataset.

This is a one-time step required before running UniMol fine-tuning (--surrogate
ft_unimol).  It generates RDKit 3D conformers for every molecule in the library
and caches them directly to:

    muben/data/files/<dataset>/processed/unimol-unimol/train.pt

The file is saved in the exact dict format that muben's Dataset.load() expects,
bypassing muben's fragile prepare()/create_features() pipeline entirely.

Usage
-----
    python preprocess_unimol.py --dataset Enamine50k

This takes ~10-40 minutes on CPU (parallelised over --workers cores).
Run once; subsequent AL experiments reuse the cache.
"""

import argparse
import sys
import time
from functools import partial
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "muben"))

MOLPAL_LIB = ROOT / "molpal" / "libraries"
DATA_DIR   = ROOT / "muben" / "data" / "files"


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


def _generate_one(smiles: str):
    """Worker: generate conformers for a single SMILES. Returns (atoms, coordinates)."""
    from muben.utils.chem import smiles_to_coords
    try:
        return smiles_to_coords(smiles, n_conformer=10)
    except Exception as e:
        # Return 2D fallback on failure so the pool never stalls
        from muben.utils.chem import smiles_to_2d_coords
        from rdkit import Chem
        from rdkit.Chem import AllChem
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return [], []
        mol = AllChem.AddHs(mol)
        atoms = [atom.GetSymbol() for atom in mol.GetAtoms()]
        coords_2d = smiles_to_2d_coords(smiles).astype(np.float32)
        return atoms, [coords_2d] * 11


def preprocess(dataset: str, workers: int):
    smiles = load_library_smiles(dataset)
    N = len(smiles)

    out_dir = DATA_DIR / dataset / "processed" / "unimol-unimol"
    out_dir.mkdir(parents=True, exist_ok=True)
    cached = out_dir / "train.pt"

    # Validate existing cache
    if cached.exists():
        try:
            existing = torch.load(cached, weights_only=False)
            n_cached = len(existing.get("_smiles", []))
            n_atoms  = len(existing.get("_atoms",  []))
            if n_cached == N and n_atoms == N:
                print(f"[cache] {cached} already has {n_cached:,} molecules with conformers — nothing to do.")
                return
            else:
                print(f"[cache] {cached}: {n_cached:,} SMILES, {n_atoms:,} atom lists (expected {N:,}). Regenerating.")
        except Exception as e:
            print(f"[cache] {cached} is corrupt ({e}). Regenerating.")
        cached.unlink()

    print(f"\nGenerating UniMol 3D conformers for {N:,} molecules …")
    print(f"Workers: {workers}  |  Output: {cached}")
    print("This may take 10-40 minutes depending on hardware.\n")

    t0 = time.perf_counter()
    all_atoms       = []
    all_coordinates = []

    with get_context("fork").Pool(workers) as pool:
        for atoms, coordinates in tqdm(
            pool.imap(_generate_one, smiles),
            total=N,
            desc="conformers",
        ):
            all_atoms.append(atoms)
            all_coordinates.append(coordinates)

    elapsed = time.perf_counter() - t0

    # Verify before saving
    assert len(all_atoms) == N, f"Expected {N} atom lists, got {len(all_atoms)}"
    empty = sum(1 for a in all_atoms if len(a) == 0)
    if empty > 0:
        print(f"[warn] {empty} molecules produced empty atom lists (will still save)")

    # Save in the exact format muben Dataset.load() expects
    attr_dict = {
        "_smiles":      smiles,
        "_lbs":         np.zeros((N, 1), dtype=np.float32),
        "_masks":       np.ones((N, 1),  dtype=np.float32),
        "_ori_ids":     None,
        "_partition":   "train",
        "_atoms":       all_atoms,
        "_coordinates": all_coordinates,
    }
    torch.save(attr_dict, cached)

    saved_size_mb = cached.stat().st_size / 1e6
    print(f"\n[done] {N:,} molecules in {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"       {empty} empty conformers  |  file size: {saved_size_mb:.1f} MB")
    print(f"       Saved → {cached}")
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
