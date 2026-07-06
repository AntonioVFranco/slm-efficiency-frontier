"""PyTorch dataset loaders for benchmark examples."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

import torch
from torch.utils.data import Dataset

from .schemas import Example


def load_examples(path: str | Path) -> list[Example]:
    path = Path(path)
    examples: list[Example] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            examples.append(Example(**data))
    return examples


class BenchmarkDataset(Dataset):
    """A torch Dataset wrapping benchmark examples."""

    def __init__(self, path: str | Path) -> None:
        self.examples = load_examples(path)
        for ex in self.examples:
            if not ex.validate():
                raise ValueError(f"Invalid example: {ex.id}")

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Example:
        return self.examples[idx]

    def task_indices(self, task: str) -> torch.Tensor:
        return torch.tensor([i for i, e in enumerate(self.examples) if e.task == task])

    def iterate(self) -> Iterator[Example]:
        yield from self.examples
