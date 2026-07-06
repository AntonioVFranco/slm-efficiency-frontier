"""A PyTorch cross-encoder baseline for judge/reranker calibration."""

from __future__ import annotations

import torch
from torch import nn


class CrossEncoder(nn.Module):
    def __init__(self, vocab_size: int = 1000, embed_dim: int = 64) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.encoder = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.Tanh(),
            nn.Linear(embed_dim, 1),
        )

    def forward(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        left_emb = self.embedding(left).mean(dim=1)
        right_emb = self.embedding(right).mean(dim=1)
        combined = torch.cat([left_emb, right_emb], dim=-1)
        return self.encoder(combined).squeeze(-1)
