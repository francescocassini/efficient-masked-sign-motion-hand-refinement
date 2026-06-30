"""Standalone T5-1 residual token transformer.

This is the first lightweight generative test for T5 residual tokens. It is
kept separate from P3/P5 and intentionally small so we can validate whether
oracle residual tokens are predictable before wiring the full mBART context.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class T5ResidualTokenTransformer(nn.Module):
    def __init__(
        self,
        codebook_size: int = 32,
        base_features: int = 90,
        text_vocab_size: int = 4096,
        text_max_len: int = 96,
        hidden_features: int = 256,
        layers: int = 4,
        heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.codebook_size = int(codebook_size)
        self.text_max_len = int(text_max_len)
        self.base_projection = nn.Linear(base_features, hidden_features)
        self.text_embedding = nn.Embedding(text_vocab_size, hidden_features)
        self.text_projection = nn.Sequential(
            nn.LayerNorm(hidden_features),
            nn.Linear(hidden_features, hidden_features),
            nn.GELU(),
        )
        self.position_embedding = nn.Embedding(512, hidden_features)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_features,
            nhead=heads,
            dim_feedforward=hidden_features * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        self.output = nn.Linear(hidden_features, codebook_size)

    @staticmethod
    def hash_text(texts: list[str], vocab_size: int, max_len: int, device) -> Tensor:
        tokens = torch.zeros((len(texts), max_len), dtype=torch.long, device=device)
        for row, text in enumerate(texts):
            encoded = text.encode("utf-8", errors="ignore")[:max_len]
            for col, value in enumerate(encoded):
                tokens[row, col] = 1 + (int(value) % (vocab_size - 1))
        return tokens

    def forward(self, base_hand: Tensor, text_tokens: Tensor, mask: Tensor | None = None) -> Tensor:
        batch, time, _ = base_hand.shape
        text_emb = self.text_embedding(text_tokens)
        text_mask = text_tokens.ne(0).float().unsqueeze(-1)
        text_context = (text_emb * text_mask).sum(dim=1) / text_mask.sum(dim=1).clamp_min(1.0)
        text_context = self.text_projection(text_context).unsqueeze(1)

        positions = torch.arange(time, device=base_hand.device).clamp_max(511)
        hidden = self.base_projection(base_hand)
        hidden = hidden + self.position_embedding(positions).unsqueeze(0)
        hidden = hidden + text_context
        key_padding_mask = None if mask is None else ~mask.bool()
        hidden = self.encoder(hidden, src_key_padding_mask=key_padding_mask)
        return self.output(hidden)
