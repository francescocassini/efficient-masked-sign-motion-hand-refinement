"""P6-D continuous gain regressor for hand-token candidates.

Unlike P6-C, this model predicts the expected continuous gain of replacing a
P5 hand token with a P6-B candidate token. The score is used only for ranking:
the final token still comes from P6-B and the body stays untouched.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from mGPT.models.utils.p6_hand_gain_gate import P6HandGainGate


class P6HandGainRegressor(P6HandGainGate):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        hidden_features = self.lhand_gate.in_features
        self.lhand_gate = nn.Sequential(
            nn.LayerNorm(hidden_features),
            nn.Linear(hidden_features, hidden_features),
            nn.GELU(),
            nn.Linear(hidden_features, 1),
        )
        self.rhand_gate = nn.Sequential(
            nn.LayerNorm(hidden_features),
            nn.Linear(hidden_features, hidden_features),
            nn.GELU(),
            nn.Linear(hidden_features, 1),
        )

    @staticmethod
    def hash_text(texts: list[str], vocab_size: int, max_len: int, device) -> Tensor:
        return P6HandGainGate.hash_text(texts, vocab_size, max_len, device)
