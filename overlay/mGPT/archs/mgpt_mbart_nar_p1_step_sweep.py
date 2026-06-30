"""P1 SOKE-NAR experiment: controlled sweep of fixed NAR refinement steps.

The current model and P0 remain untouched. This variant inherits P0 encoder
cache and makes the tested refinement budget explicit in a separate model.
"""
from .mgpt_mbart_nar_p0_encoder_cache import Mbart_Based_MLM_NAR_P0_EncoderCache


class Mbart_Based_MLM_NAR_P1_StepSweep(Mbart_Based_MLM_NAR_P0_EncoderCache):
    """SOKE-NAR P1 with a fixed, explicitly configured refinement budget."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.nar_steps < 1:
            raise ValueError("P1 nar_steps must be positive")
        print(f"[SOKE-NAR-P1] fixed refinement budget experiment: K={self.nar_steps}")
