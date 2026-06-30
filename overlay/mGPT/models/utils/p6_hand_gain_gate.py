"""P6-C gain gate for hand-token candidates.

The gate does not generate new tokens. It scores whether a candidate token
predicted by P6-B should replace the original P5 token.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from mGPT.models.utils.t5_residual_transformer import T5ResidualTokenTransformer


class P6HandGainGate(nn.Module):
    def __init__(
        self,
        body_vocab_size: int = 96,
        hand_vocab_size: int = 192,
        token_dim: int = 96,
        text_vocab_size: int = 4096,
        text_max_len: int = 96,
        hidden_features: int = 192,
        layers: int = 3,
        heads: int = 4,
        dropout: float = 0.1,
        max_tokens: int = 256,
    ):
        super().__init__()
        self.text_max_len = int(text_max_len)
        self.body_embedding = nn.Embedding(body_vocab_size, token_dim)
        self.lhand_embedding = nn.Embedding(hand_vocab_size, token_dim)
        self.rhand_embedding = nn.Embedding(hand_vocab_size, token_dim)
        self.cand_lhand_embedding = nn.Embedding(hand_vocab_size, token_dim)
        self.cand_rhand_embedding = nn.Embedding(hand_vocab_size, token_dim)
        self.meta_projection = nn.Sequential(
            nn.Linear(7, token_dim),
            nn.GELU(),
            nn.LayerNorm(token_dim),
        )
        self.input_projection = nn.Linear(token_dim * 6, hidden_features)
        self.position_embedding = nn.Embedding(max_tokens, hidden_features)
        self.text_embedding = nn.Embedding(text_vocab_size, hidden_features)
        self.text_projection = nn.Sequential(
            nn.LayerNorm(hidden_features),
            nn.Linear(hidden_features, hidden_features),
            nn.GELU(),
        )
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
        self.lhand_gate = nn.Linear(hidden_features, 1)
        self.rhand_gate = nn.Linear(hidden_features, 1)

    @staticmethod
    def hash_text(texts: list[str], vocab_size: int, max_len: int, device) -> Tensor:
        return T5ResidualTokenTransformer.hash_text(texts, vocab_size, max_len, device)

    def forward(
        self,
        body_tokens: Tensor,
        lhand_tokens: Tensor,
        rhand_tokens: Tensor,
        cand_lhand_tokens: Tensor,
        cand_rhand_tokens: Tensor,
        meta: Tensor,
        text_tokens: Tensor,
        mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        _, time = body_tokens.shape
        hidden = torch.cat(
            [
                self.body_embedding(body_tokens),
                self.lhand_embedding(lhand_tokens),
                self.rhand_embedding(rhand_tokens),
                self.cand_lhand_embedding(cand_lhand_tokens),
                self.cand_rhand_embedding(cand_rhand_tokens),
                self.meta_projection(meta),
            ],
            dim=-1,
        )
        text_emb = self.text_embedding(text_tokens)
        text_mask = text_tokens.ne(0).float().unsqueeze(-1)
        text_context = (text_emb * text_mask).sum(dim=1) / text_mask.sum(dim=1).clamp_min(1.0)
        text_context = self.text_projection(text_context).unsqueeze(1)
        positions = torch.arange(time, device=body_tokens.device).clamp_max(
            self.position_embedding.num_embeddings - 1
        )
        hidden = self.input_projection(hidden)
        hidden = hidden + self.position_embedding(positions).unsqueeze(0)
        hidden = hidden + text_context
        key_padding_mask = None if mask is None else ~mask.bool()
        hidden = self.encoder(hidden, src_key_padding_mask=key_padding_mask)
        return self.lhand_gate(hidden).squeeze(-1), self.rhand_gate(hidden).squeeze(-1)
