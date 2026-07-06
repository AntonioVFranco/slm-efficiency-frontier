"""A PyTorch reranker baseline using the cross-encoder."""

from __future__ import annotations

import torch

from .cross_encoder import CrossEncoder


class Reranker:
    def __init__(self, cross_encoder: CrossEncoder | None = None) -> None:
        self.cross_encoder = cross_encoder or CrossEncoder()

    def rank(self, query: torch.Tensor, candidates: list[torch.Tensor]) -> list[int]:
        scores = [float(self.cross_encoder(query, c).item()) for c in candidates]
        order = sorted(range(len(candidates)), key=lambda i: scores[i], reverse=True)
        return order
