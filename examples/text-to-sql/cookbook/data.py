"""RL dataset for the cookbook Text-to-SQL recipe, built on ``utils.rewards.load_dataset_splits``."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import chz
from tinker_cookbook.rl.types import EnvGroupBuilder, RLDataset, RLDatasetBuilder
from utils.rewards import DEFAULT_DATASET, load_dataset_splits

from .env import SandboxFactory, TextToSqlGroupBuilder


class TextToSqlDataset(RLDataset):
  def __init__(self, builders: list[TextToSqlGroupBuilder], batch_size: int):
    self.builders = builders
    self.batch_size = batch_size

  def get_batch(self, index: int) -> Sequence[EnvGroupBuilder]:
    start = index * self.batch_size
    return self.builders[start : start + self.batch_size]

  def __len__(self) -> int:
    return max(1, len(self.builders) // self.batch_size)


@chz.chz
class TextToSqlDatasetBuilder(RLDatasetBuilder):
  model_name_for_tokenizer: str
  renderer_name: str
  group_size: int
  groups_per_batch: int
  dataset_name: str = DEFAULT_DATASET
  dataset_limit: int = 12_500
  train_limit: int = 5_000
  eval_limit: int = 100
  seed: int = 42
  exec_timeout: int = 30
  # Injected by train.py; None runs the model's SQL in-process.
  sandbox_factory: SandboxFactory | None = None

  def _builders(self, rows: list[dict[str, Any]], group_size: int) -> list[TextToSqlGroupBuilder]:
    return [
      TextToSqlGroupBuilder(
        row=row,
        model_name_for_tokenizer=self.model_name_for_tokenizer,
        renderer_name=self.renderer_name,
        group_size=group_size,
        sandbox_factory=self.sandbox_factory,
        exec_timeout=self.exec_timeout,
      )
      for row in rows
    ]

  async def __call__(self) -> tuple[RLDataset, RLDataset | None]:
    train_rows, eval_rows = load_dataset_splits(
      dataset_name=self.dataset_name,
      dataset_limit=self.dataset_limit,
      train_limit=self.train_limit,
      eval_limit=self.eval_limit,
      seed=self.seed,
    )
    train = TextToSqlDataset(self._builders(train_rows, self.group_size), self.groups_per_batch)
    # Eval: one sample per prompt, whole eval split in one batch.
    test = TextToSqlDataset(self._builders(eval_rows, 1), max(1, len(eval_rows)))
    return train, test
