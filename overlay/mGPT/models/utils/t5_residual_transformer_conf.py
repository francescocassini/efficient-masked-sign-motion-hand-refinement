"""T5-1D residual predictor conditioned on P5 features, tokens and confidence."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from mGPT.models.utils.t5_residual_transformer import T5ResidualTokenTransformer


class T5ResidualConfidenceTransformer(nn.Module):
    def __init__(
        self,
        codebook_size: int = 32,
        base_features: int = 133,
        confidence_features: int = 3,
        body_vocab_size: int = 96,
        hand_vocab_size: int = 192,
        token_dim: int = 64,
        text_vocab_size: int = 4096,
        text_max_len: int = 96,
        hidden_features: int = 128,
        layers: int = 2,
        heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.codebook_size = int(codebook_size)
        self.text_max_len = int(text_max_len)
        self.body_embedding = nn.Embedding(body_vocab_size, token_dim)
        self.lhand_embedding = nn.Embedding(hand_vocab_size, token_dim)
        self.rhand_embedding = nn.Embedding(hand_vocab_size, token_dim)
        self.confidence_projection = nn.Sequential(
            nn.Linear(confidence_features, token_dim),
            nn.GELU(),
            nn.LayerNorm(token_dim),
        )
        self.input_projection = nn.Linear(
            base_features + token_dim * 4, hidden_features
        )
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
        return T5ResidualTokenTransformer.hash_text(texts, vocab_size, max_len, device)

    def forward(
        self,
        base_full: Tensor,
        body_tokens_frame: Tensor,
        lhand_tokens_frame: Tensor,
        rhand_tokens_frame: Tensor,
        confidence_frame: Tensor,
        text_tokens: Tensor,
        mask: Tensor | None = None,
    ) -> Tensor:
        _, time, _ = base_full.shape
        token_context = torch.cat(
            [
                self.body_embedding(body_tokens_frame),
                self.lhand_embedding(lhand_tokens_frame),
                self.rhand_embedding(rhand_tokens_frame),
                self.confidence_projection(confidence_frame),
            ],
            dim=-1,
        )
        text_emb = self.text_embedding(text_tokens)
        text_mask = text_tokens.ne(0).float().unsqueeze(-1)
        text_context = (text_emb * text_mask).sum(dim=1) / text_mask.sum(dim=1).clamp_min(1.0)
        text_context = self.text_projection(text_context).unsqueeze(1)

        positions = torch.arange(time, device=base_full.device).clamp_max(511)
        hidden = self.input_projection(torch.cat([base_full, token_context], dim=-1))
        hidden = hidden + self.position_embedding(positions).unsqueeze(0)
        hidden = hidden + text_context
        key_padding_mask = None if mask is None else ~mask.bool()
        hidden = self.encoder(hidden, src_key_padding_mask=key_padding_mask)
        return self.output(hidden)
