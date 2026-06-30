"""P5 Aggressive: three hand-only polishing passes for the controlled sweep."""

from .mgpt_mbart_nar_p5_hand_polish_base import Mbart_Based_MLM_NAR_P5_HandPolishBase


class Mbart_Based_MLM_NAR_P5_HandPolishAggressive(Mbart_Based_MLM_NAR_P5_HandPolishBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, hand_polish_ratios=(0.25, 0.15, 0.08), **kwargs)
