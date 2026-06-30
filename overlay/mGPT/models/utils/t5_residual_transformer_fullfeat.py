"""T5-1B residual token transformer with full P5 feature conditioning."""

from __future__ import annotations

from mGPT.models.utils.t5_residual_transformer import T5ResidualTokenTransformer


class T5ResidualFullFeatureTransformer(T5ResidualTokenTransformer):
    """Same lightweight T5-1 backbone, but conditioned on all 133 P5 features."""

    def __init__(self, *args, base_features: int = 133, **kwargs):
        super().__init__(*args, base_features=base_features, **kwargs)
