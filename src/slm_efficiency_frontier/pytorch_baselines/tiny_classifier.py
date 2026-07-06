"""A tiny PyTorch classifier baseline for semantic classification tasks."""

from __future__ import annotations

import torch
from torch import nn


class TinyClassifier(nn.Module):
    def __init__(self, vocab_size: int = 1000, embed_dim: int = 64, num_classes: int = 5) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.encoder = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, num_classes),
        )

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        pooled = self.embedding(token_ids).mean(dim=1)
        return self.encoder(pooled)
