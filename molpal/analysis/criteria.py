"""AL-centric representation-quality criteria (paper §4.1, criteria C1-C4).

Each function takes plain numpy arrays / SMILES lists rather than a
surrogate or Explorer object, so the same code computes:
  - a static snapshot for Table 1 (criteria averaged over targets/seeds), and
  - a per-round value during the AL loop (e.g. Figure 3's AL-ECE curve).

C1 Local Smoothness      -> local_smoothness
C2 Predictive Calibration -> expected_calibration_error
C3 Exploration Diversity  -> latent_diversity, tanimoto_diversity
C4 Target-aware Organization -> knn_target_precision
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
from scipy.stats import spearmanr
from sklearn.neighbors import NearestNeighbors


def _sample_pairs(n: int, n_pairs: int, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    """Sample up to n_pairs distinct index pairs (i, j), i != j."""
    rng = np.random.default_rng(seed)
    n_pairs = min(n_pairs, n * (n - 1) // 2)
    i = rng.integers(0, n, size=n_pairs)
    j = rng.integers(0, n, size=n_pairs)
    same = i == j
    while same.any():
        j[same] = rng.integers(0, n, size=same.sum())
        same = i == j
    return i, j


# ── C1: Local Smoothness ────────────────────────────────────────────────────

def local_smoothness(
    embeddings: np.ndarray,
    scores: np.ndarray,
    n_pairs: int = 20_000,
    seed: int = 0,
) -> float:
    """rho = Spearman(||phi(xi) - phi(xj)||, |f(xi) - f(xj)|) over sampled pairs.

    Higher rho means embedding distance tracks oracle score difference, i.e.
    the representation supports accurate surrogate interpolation.
    """
    i, j = _sample_pairs(len(embeddings), n_pairs, seed)
    emb_dist = np.linalg.norm(embeddings[i] - embeddings[j], axis=1)
    score_diff = np.abs(scores[i] - scores[j])
    rho, _ = spearmanr(emb_dist, score_diff)
    return float(rho)


# ── C2: Predictive Calibration ──────────────────────────────────────────────

def expected_calibration_error(
    y_true: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Regression ECE: partition predictions into uncertainty-based bins and
    check whether predicted variance matches observed MSE within each bin.

    ECE = sum_b (n_b / N) * |mean_var_b - mean_squared_error_b|
    """
    sq_err = (y_true - mu) ** 2
    var = sigma ** 2
    order = np.argsort(sigma)
    bins = np.array_split(order, n_bins)

    n = len(y_true)
    ece = 0.0
    for b in bins:
        if len(b) == 0:
            continue
        ece += (len(b) / n) * abs(var[b].mean() - sq_err[b].mean())
    return float(ece)


# ── C3: Exploration Diversity ────────────────────────────────────────────────

def latent_diversity(
    embeddings: np.ndarray,
    n_pairs: int = 20_000,
    seed: int = 0,
) -> float:
    """Mean pairwise L2 distance in embedding space (geometric coverage)."""
    i, j = _sample_pairs(len(embeddings), n_pairs, seed)
    return float(np.linalg.norm(embeddings[i] - embeddings[j], axis=1).mean())


def tanimoto_diversity(smiles: List[str], radius: int = 2, n_bits: int = 2048) -> float:
    """Mean pairwise Tanimoto distance over ECFP4 fingerprints:

    Div(S) = 2 / (|S|(|S|-1)) * sum_{i<j} D_Tan(xi, xj),  D_Tan = 1 - Tanimoto similarity.

    Measures scaffold-hopping ability of a discovered/selected molecule set.
    """
    from rdkit import Chem
    from rdkit import DataStructs
    from rdkit.Chem import rdMolDescriptors as rdmd

    fps = []
    for smi in smiles:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        fps.append(rdmd.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=n_bits))

    n = len(fps)
    if n < 2:
        return 0.0

    total = 0.0
    for idx in range(n - 1):
        sims = DataStructs.BulkTanimotoSimilarity(fps[idx], fps[idx + 1 :])
        total += sum(1.0 - s for s in sims)

    return total / (n * (n - 1) / 2)


# ── C4: Target-aware Organization ────────────────────────────────────────────

def knn_target_precision(
    embeddings: np.ndarray,
    scores: np.ndarray,
    k: int = 10,
    max_anchors: Optional[int] = 5_000,
    seed: int = 0,
) -> float:
    """kNN precision@k: for each anchor molecule, the fraction of its k
    nearest neighbors in embedding space that are also its k nearest
    neighbors by oracle score.

    max_anchors subsamples the anchor points (not the neighbor pool) for
    tractability on large libraries; neighbors are still searched over the
    full pool passed in.
    """
    n = len(embeddings)
    k = min(k, n - 1)

    if max_anchors is not None and n > max_anchors:
        rng = np.random.default_rng(seed)
        anchors = rng.choice(n, size=max_anchors, replace=False)
    else:
        anchors = np.arange(n)

    emb_nn = NearestNeighbors(n_neighbors=k + 1).fit(embeddings)
    _, emb_idx = emb_nn.kneighbors(embeddings[anchors])

    score_nn = NearestNeighbors(n_neighbors=k + 1).fit(scores.reshape(-1, 1))
    _, score_idx = score_nn.kneighbors(scores[anchors].reshape(-1, 1))

    precisions = np.empty(len(anchors))
    for row, a in enumerate(anchors):
        emb_neighbors = set(emb_idx[row].tolist()) - {a}
        score_neighbors = set(score_idx[row].tolist()) - {a}
        precisions[row] = len(emb_neighbors & score_neighbors) / k

    return float(precisions.mean())


# ── Convenience aggregator (one row of Table 1) ─────────────────────────────

def compute_criteria_row(
    embeddings: np.ndarray,
    scores: np.ndarray,
    smiles: List[str],
    mu: Optional[np.ndarray] = None,
    sigma: Optional[np.ndarray] = None,
    n_pairs: int = 20_000,
    knn_k: int = 10,
    seed: int = 0,
) -> dict:
    """Compute C1, C3 (both variants), C4, and (if mu/sigma given) C2 for one
    representation, matching the columns of Table 1 / tab:criteria.

    ECE requires surrogate predictions (mu, sigma) on the same set as
    `scores`; pass None to skip it (e.g. when only evaluating raw geometry).
    """
    row = {
        "smoothness_rho": local_smoothness(embeddings, scores, n_pairs=n_pairs, seed=seed),
        "latent_diversity": latent_diversity(embeddings, n_pairs=n_pairs, seed=seed),
        "tanimoto_diversity": tanimoto_diversity(smiles),
        "knn_precision": knn_target_precision(embeddings, scores, k=knn_k, seed=seed),
    }
    if mu is not None and sigma is not None:
        row["ece"] = expected_calibration_error(scores, mu, sigma)
    return row
