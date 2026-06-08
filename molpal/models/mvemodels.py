"""Embedding-based MVE models that wrap ALSU surrogates in the molpal Model ABC.

Two classes are provided:
  EmbeddingMVEModel    — multi-backbone: concatenates all embeddings in emb_dict
  SingleBackboneEmbeddingModel — single-backbone: uses one key from emb_dict

Both accept the same constructor signature and plug transparently into Explorer.
"""

from typing import Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np

from molpal.models.base import Model


class EmbeddingMVEModel(Model):
    """Wraps any ALSU surrogate using pre-extracted backbone embeddings.

    Attributes
    ----------
    surrogate : ALSU surrogate instance
        Must implement fit(X, y) and predict(X) -> (mu, sigma).
    emb_dict : dict[str, np.ndarray]
        Mapping backbone_name -> (N, D) float32 embedding matrix.
        Arrays are indexed in pool_smiles order.
    smi2idx : dict[str, int]
        Maps each SMILES string to its row index in the embedding matrices.
    embeddings_refreshed : bool
        Set to True by fit() when a scheduled fine-tuning surrogate updates
        its embeddings in-place. Explorer checks this flag after each training
        step and resets it.
    """

    @property
    def provides(self) -> Set[str]:
        return {"means", "vars"}

    @property
    def type_(self) -> str:
        return "mve"

    def __init__(
        self,
        surrogate,
        emb_dict: Dict[str, np.ndarray],
        pool_smiles: List[str],
        test_batch_size: int = 4096,
        **kwargs,
    ):
        super().__init__(test_batch_size=test_batch_size, **kwargs)
        self.surrogate = surrogate
        self.emb_dict = emb_dict
        self.smi2idx = {s: i for i, s in enumerate(pool_smiles)}
        self.embeddings_refreshed = False

    def _get_X(self, xs: Sequence[str]) -> np.ndarray:
        idxs = [self.smi2idx[s] for s in xs]
        parts = [emb[idxs] for emb in self.emb_dict.values()]
        return np.concatenate(parts, axis=1)  # (n, D_total)

    def train(
        self,
        xs: Iterable[str],
        ys: np.ndarray,
        *,
        featurizer: Optional[Callable] = None,
        retrain: bool = False,
        **kwargs,
    ) -> bool:
        xs = list(xs)
        X = self._get_X(xs)
        # Scheduled fine-tuning surrogates (needs_smiles=True) require the actual
        # SMILES strings in Phase 2 so they can run BackboneFinetuner.finetune().
        if getattr(self.surrogate, "needs_smiles", False):
            self.surrogate.fit(X, ys, labeled_smiles=xs)
        else:
            self.surrogate.fit(X, ys)
        # Propagate embedding-refresh signal from scheduled fine-tuning surrogates
        self.embeddings_refreshed = getattr(self.surrogate, "embeddings_refreshed", False)
        return True

    def get_means(self, xs: Sequence[str]) -> np.ndarray:
        xs = list(xs)
        mu, _ = self.surrogate.predict(self._get_X(xs))
        return mu

    def get_means_and_vars(self, xs: Sequence[str]) -> Tuple[np.ndarray, np.ndarray]:
        xs = list(xs)
        mu, sigma = self.surrogate.predict(self._get_X(xs))
        return mu, sigma ** 2  # Model ABC expects variance, not std

    def apply(self, x_ids, x_feats, batched_size=None, size=None, mean_only=True):
        xs = list(x_ids)
        if mean_only:
            return self.get_means(xs), np.array([])
        return self.get_means_and_vars(xs)

    def save(self, path) -> str:
        return str(path)

    def load(self, path):
        pass


class SingleBackboneEmbeddingModel(EmbeddingMVEModel):
    """Like EmbeddingMVEModel but uses a single backbone key from emb_dict.

    Useful with SingleBackboneMVESurrogate where concatenating all backbones
    would be incorrect.

    Parameters
    ----------
    backbone : str
        The key in emb_dict to use (e.g. "molformer", "grover", "unimol").
    """

    def __init__(
        self,
        surrogate,
        emb_dict: Dict[str, np.ndarray],
        pool_smiles: List[str],
        backbone: str,
        test_batch_size: int = 4096,
        **kwargs,
    ):
        super().__init__(surrogate, emb_dict, pool_smiles, test_batch_size, **kwargs)
        if backbone not in emb_dict:
            raise ValueError(
                f"backbone '{backbone}' not found in emb_dict. "
                f"Available: {list(emb_dict.keys())}"
            )
        self.backbone = backbone
        # Override emb_dict to expose only the selected backbone so that
        # _get_X() naturally does np.concatenate over a single array
        self.emb_dict = {backbone: emb_dict[backbone]}
