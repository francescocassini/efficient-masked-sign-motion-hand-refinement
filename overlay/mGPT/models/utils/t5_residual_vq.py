"""Residual VQ components for the isolated T5 RVQ experiment.

These modules are intentionally standalone: they do not modify the historical
SOKE VQ tokenizer, P3, P5, or T4. T5 treats the current SOKE output as level-0
and learns a second residual level only for hand features.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class T5VectorQuantizer(nn.Module):
    """Small straight-through VQ codebook for residual hand features."""

    def __init__(self, codebook_size: int = 256, code_dim: int = 256, beta: float = 0.25):
        super().__init__()
        if codebook_size <= 1 or code_dim <= 0:
            raise ValueError("codebook_size must be >1 and code_dim must be positive")
        self.codebook_size = int(codebook_size)
        self.code_dim = int(code_dim)
        self.beta = float(beta)
        self.embedding = nn.Embedding(self.codebook_size, self.code_dim)
        nn.init.uniform_(self.embedding.weight, -1.0 / self.codebook_size, 1.0 / self.codebook_size)

    def forward(self, latents: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Quantize [B,T,C] latents.

        Returns quantized latents, VQ loss, token ids [B,T], and perplexity.
        """
        flat = latents.reshape(-1, self.code_dim)
        codebook = self.embedding.weight
        distances = (
            flat.pow(2).sum(dim=1, keepdim=True)
            - 2.0 * flat @ codebook.t()
            + codebook.pow(2).sum(dim=1).unsqueeze(0)
        )
        tokens = distances.argmin(dim=1)
        quantized = self.embedding(tokens).view_as(latents)

        codebook_loss = F.mse_loss(quantized, latents.detach())
        commitment_loss = F.mse_loss(latents, quantized.detach())
        loss = codebook_loss + self.beta * commitment_loss

        quantized = latents + (quantized - latents).detach()
        one_hot = F.one_hot(tokens, self.codebook_size).float()
        avg_probs = one_hot.mean(dim=0)
        perplexity = torch.exp(-(avg_probs * (avg_probs + 1e-10).log()).sum())
        return quantized, loss, tokens.view(latents.shape[0], latents.shape[1]), perplexity

    def decode_tokens(self, tokens: Tensor) -> Tensor:
        return self.embedding(tokens)


class T5ResidualVQVAE(nn.Module):
    """Temporal residual tokenizer for LH+RH hand deltas.

    Input and output are normalized residual hand features with shape
    [B,T,90]. A bounded output avoids destructive corrections during oracle
    reconstruction.
    """

    def __init__(
        self,
        input_features: int = 90,
        hidden_features: int = 256,
        codebook_size: int = 256,
        code_dim: int = 256,
        dropout: float = 0.1,
        max_residual: float = 1.0,
    ):
        super().__init__()
        self.input_features = int(input_features)
        self.max_residual = float(max_residual)
        self.encoder = nn.Sequential(
            nn.Conv1d(input_features, hidden_features, kernel_size=3, padding=1),
            nn.GroupNorm(8, hidden_features),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_features, hidden_features, kernel_size=3, padding=2, dilation=2),
            nn.GroupNorm(8, hidden_features),
            nn.GELU(),
            nn.Conv1d(hidden_features, code_dim, kernel_size=1),
        )
        self.quantizer = T5VectorQuantizer(codebook_size=codebook_size, code_dim=code_dim)
        self.decoder = nn.Sequential(
            nn.Conv1d(code_dim, hidden_features, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(hidden_features, hidden_features, kernel_size=3, padding=2, dilation=2),
            nn.GroupNorm(8, hidden_features),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_features, input_features, kernel_size=3, padding=1),
        )

    def encode_latents(self, residual: Tensor) -> Tensor:
        return self.encoder(residual.transpose(1, 2)).transpose(1, 2)

    def decode_latents(self, latents: Tensor) -> Tensor:
        decoded = self.decoder(latents.transpose(1, 2)).transpose(1, 2)
        return self.max_residual * torch.tanh(decoded)

    def forward(self, residual: Tensor) -> dict[str, Tensor]:
        latents = self.encode_latents(residual)
        quantized, vq_loss, tokens, perplexity = self.quantizer(latents)
        reconstruction = self.decode_latents(quantized)
        return {
            "reconstruction": reconstruction,
            "tokens": tokens,
            "vq_loss": vq_loss,
            "perplexity": perplexity,
        }

    @torch.no_grad()
    def encode(self, residual: Tensor) -> Tensor:
        latents = self.encode_latents(residual)
        _, _, tokens, _ = self.quantizer(latents)
        return tokens

    @torch.no_grad()
    def decode(self, tokens: Tensor) -> Tensor:
        latents = self.quantizer.decode_tokens(tokens)
        return self.decode_latents(latents)
