"""This module contains the Model ABC and various implementations thereof. A
model is used to predict an input's objective function based on prior
training data."""

from typing import Dict, List, Optional, Set

import numpy as np
from molpal.models.base import Model


def model(model: str, **kwargs) -> Model:
    """Model factory function"""
    if model == "mve":
        return mve(**kwargs)

    if model == "rf":
        from molpal.models.sklmodels import RFModel

        return RFModel(**kwargs)

    if model == "lgbm":
        from molpal.models.sklmodels import LightGBMModel
        return LightGBMModel(**kwargs)

    if model == "gp":
        from molpal.models.sklmodels import GPModel

        return GPModel(**kwargs)

    if model == "nn":
        return nn(**kwargs)

    if model == "mpn":
        return mpn(**kwargs)
    
    if model == "transformer":
        return molformer(**kwargs)
    
    if model == "molclr":
        return clr(**kwargs)

    if model == "random":
        from molpal.models.random import RandomModel

        return RandomModel(**kwargs)

    raise NotImplementedError(f'Unrecognized model: "{model}"')


def nn(conf_method: Optional[str] = None, **kwargs) -> Model:
    """NN-type Model factory function"""
    from molpal.models.nnmodels import NNModel, NNDropoutModel, NNEnsembleModel, NNTwoOutputModel

    try:
        return {
            "dropout": NNDropoutModel,
            "ensemble": NNEnsembleModel,
            "twooutput": NNTwoOutputModel,
            "mve": NNTwoOutputModel,
            "none": NNModel,
        }.get(conf_method, "none")(conf_method=conf_method, **kwargs)
    except KeyError:
        raise NotImplementedError(f'Unrecognized NN confidence method: "{conf_method}"')


def mpn(conf_method: Optional[str] = None, **kwargs) -> Model:
    """MPN-type Model factory function"""
    from molpal.models.mpnmodels import MPNModel, MPNDropoutModel, MPNTwoOutputModel

    try:
        return {
            "dropout": MPNDropoutModel,
            "twooutput": MPNTwoOutputModel,
            "mve": MPNTwoOutputModel,
            "none": MPNModel,
        }.get(conf_method, "none")(conf_method=conf_method, **kwargs)
    except KeyError:
        raise NotImplementedError(f'Unrecognized MPN confidence method: "{conf_method}"')


def molformer(conf_method: Optional[str] = None, **kwargs) -> Model:
    """Pretrained Transformer Model factory function"""
    from molpal.models.transformermodels import TransformerModel, TransformerTwoOutputModel

    try:
        return {
            "twooutput": TransformerTwoOutputModel,
            "mve": TransformerTwoOutputModel,
            "none": TransformerModel,
        }.get(conf_method, "none")(conf_method=conf_method, **kwargs)
    except KeyError:
        raise NotImplementedError(f'Unrecognized Transformer confidence method: "{conf_method}"')


def clr(conf_method: Optional[str] = None, **kwargs) -> Model:
    """Pretrained MolCLR Model factory function"""
    from molpal.models.molclrmodels import MolCLRModel, MolCLRTwoOutputModel

    try:
        return {
            "twooutput": MolCLRTwoOutputModel,
            "mve": MolCLRTwoOutputModel,
            "none": MolCLRModel,
        }.get(conf_method, "none")(conf_method=conf_method, **kwargs)
    except KeyError:
        raise NotImplementedError(f'Unrecognized MolCLR confidence method: "{conf_method}"')



def mve(
    surrogate_type: str = "single",
    backbone: str = "molformer",
    emb_dict: Optional[Dict[str, np.ndarray]] = None,
    pool_smiles: Optional[List[str]] = None,
    **kwargs,
) -> Model:
    """MVE surrogate model factory backed by ALSU surrogates.

    Parameters
    ----------
    surrogate_type : str
        One of: single, lightweight, bigfusion, ensemble, learned, nonlinear,
        attention, oof, ft_molformer, ft_grover, ft_unimol, ft_fusion.
    backbone : str
        Which backbone to use for 'single' surrogate_type.
    emb_dict : dict[str, np.ndarray]
        Pre-extracted embeddings keyed by backbone name.
    pool_smiles : list[str]
        Ordered SMILES corresponding to rows in emb_dict arrays.
    """
    import sys, pathlib
    # Make sure top-level surrogates.py is importable
    root = pathlib.Path(__file__).resolve().parent.parent.parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    from surrogates import (
        SingleBackboneMVESurrogate,
        LightweightMVESurrogate,
        BigFusionSurrogate,
        EnsembleFusionSurrogate,
        LearnedFusionSurrogate,
        NonlinearFusionSurrogate,
        AttentionFusionSurrogate,
        OOFFusionSurrogate,
        LightweightMoLFormerScheduleSurrogate,
        SingleBackboneFinetuneScheduleSurrogate,
        FTFusionSurrogate,
    )
    from molpal.models.mvemodels import EmbeddingMVEModel, SingleBackboneEmbeddingModel

    if emb_dict is None or pool_smiles is None:
        raise ValueError("mve models require emb_dict and pool_smiles")

    dims = {k: v.shape[1] for k, v in emb_dict.items()}
    in_dim_total = sum(dims.values())

    if surrogate_type == "single":
        bb = backbone if backbone in dims else next(iter(dims))
        surrogate = SingleBackboneMVESurrogate(in_dim=dims[bb])
        return SingleBackboneEmbeddingModel(surrogate, emb_dict, pool_smiles, backbone=bb, **kwargs)

    if surrogate_type == "lightweight":
        surrogate = LightweightMVESurrogate(in_dim=in_dim_total)
        return EmbeddingMVEModel(surrogate, emb_dict, pool_smiles, **kwargs)

    if surrogate_type == "bigfusion":
        surrogate = BigFusionSurrogate(dims=dims)
        return EmbeddingMVEModel(surrogate, emb_dict, pool_smiles, **kwargs)

    if surrogate_type == "ensemble":
        surrogate = EnsembleFusionSurrogate(dims=dims)
        return EmbeddingMVEModel(surrogate, emb_dict, pool_smiles, **kwargs)

    if surrogate_type == "learned":
        surrogate = LearnedFusionSurrogate(dims=dims)
        return EmbeddingMVEModel(surrogate, emb_dict, pool_smiles, **kwargs)

    if surrogate_type == "nonlinear":
        surrogate = NonlinearFusionSurrogate(dims=dims)
        return EmbeddingMVEModel(surrogate, emb_dict, pool_smiles, **kwargs)

    if surrogate_type == "attention":
        surrogate = AttentionFusionSurrogate(dims=dims)
        return EmbeddingMVEModel(surrogate, emb_dict, pool_smiles, **kwargs)

    if surrogate_type == "oof":
        surrogate = OOFFusionSurrogate(dims=dims)
        return EmbeddingMVEModel(surrogate, emb_dict, pool_smiles, **kwargs)

    # Scheduled fine-tuning surrogates need the emb_dict passed as a mutable reference
    # so the surrogate can update embeddings in-place during AL rounds.
    if surrogate_type == "ft_molformer":
        surrogate = LightweightMoLFormerScheduleSurrogate(
            dims=dims, pool_smiles=pool_smiles, emb_dict=emb_dict,
            **{k: v for k, v in kwargs.items() if k in (
                "dataset_name", "n_lt_rounds", "ft_epochs", "ft_lr_bb", "ft_lr_head")}
        )
        return EmbeddingMVEModel(surrogate, emb_dict, pool_smiles, **kwargs)

    if surrogate_type in ("ft_grover", "ft_unimol", "ft_molformer_single"):
        bb = surrogate_type.replace("ft_", "")
        if bb not in dims:
            raise ValueError(
                f"surrogate_type='{surrogate_type}' requires backbone '{bb}' in emb_dict. "
                f"Pass --backbones {bb} (and any others you want). "
                f"Available: {list(dims.keys())}"
            )
        # Use a single-backbone dict shared by both surrogate and model.
        # The surrogate updates single_emb_dict[bb] = new_emb in-place after fine-tuning,
        # and EmbeddingMVEModel._get_X() reads from the same dict so it picks up the update.
        single_emb_dict = {bb: emb_dict[bb]}
        surrogate = SingleBackboneFinetuneScheduleSurrogate(
            backbone=bb, in_dim=dims[bb],
            pool_smiles=pool_smiles, emb_dict=single_emb_dict,
            **{k: v for k, v in kwargs.items() if k in (
                "dataset_name", "n_ft_delay", "ft_epochs", "ft_lr_bb", "ft_lr_head")}
        )
        return EmbeddingMVEModel(surrogate, single_emb_dict, pool_smiles, **kwargs)

    if surrogate_type == "ft_fusion":
        surrogate = FTFusionSurrogate(
            dims=dims, pool_smiles=pool_smiles, emb_dict=emb_dict,
            **{k: v for k, v in kwargs.items() if k in (
                "dataset_name", "n_ft_delay", "ft_epochs", "ft_lr_bb", "ft_lr_head")}
        )
        return EmbeddingMVEModel(surrogate, emb_dict, pool_smiles, **kwargs)

    raise NotImplementedError(f'Unrecognized surrogate_type: "{surrogate_type}"')


def model_types() -> Set[str]:
    return {"rf", "gp", "nn", "mpn", "lgbm", "transformer", "molclr", "mve"}
