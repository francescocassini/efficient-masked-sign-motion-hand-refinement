"""P6-H learned selector for P6-B top-k hand-token candidates."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class P6TopKCandidateSelector(nn.Module):
    def __init__(
        self,
        body_vocab_size: int = 96,
        hand_vocab_size: int = 192,
        token_dim: int = 96,
        meta_dim: int = 10,
        max_rank: int = 8,
        hidden_features: int = 192,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.meta_dim = int(meta_dim)
        self.body_embedding = nn.Embedding(body_vocab_size, token_dim)
        self.orig_embedding = nn.Embedding(hand_vocab_size, token_dim)
        self.other_embedding = nn.Embedding(hand_vocab_size, token_dim)
        self.candidate_embedding = nn.Embedding(hand_vocab_size, token_dim)
        self.side_embedding = nn.Embedding(2, token_dim)
        self.rank_embedding = nn.Embedding(max_rank + 1, token_dim)
        self.meta_projection = nn.Sequential(
            nn.Linear(self.meta_dim, token_dim),
            nn.GELU(),
            nn.LayerNorm(token_dim),
        )
        in_features = token_dim * 7
        self.scorer = nn.Sequential(
            nn.LayerNorm(in_features),
            nn.Linear(in_features, hidden_features),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_features, hidden_features),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_features, 1),
        )

    def forward(
        self,
        body_tokens: Tensor,
        orig_tokens: Tensor,
        other_tokens: Tensor,
        candidate_tokens: Tensor,
        side: Tensor,
        rank: Tensor,
        meta: Tensor,
    ) -> Tensor:
        hidden = torch.cat(
            [
                self.body_embedding(body_tokens),
                self.orig_embedding(orig_tokens),
                self.other_embedding(other_tokens),
                self.candidate_embedding(candidate_tokens),
                self.side_embedding(side),
                self.rank_embedding(rank),
                self.meta_projection(meta),
            ],
            dim=-1,
        )
        return self.scorer(hidden).squeeze(-1)
