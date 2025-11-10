"""Configuration dataclasses for the transaction GAN pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from .gan import GANTrainingConfig


@dataclass
class SchemaConfig:
    """Describe how dataset columns should be interpreted."""

    id_column: str | None = None
    geo_column: str | None = None
    date_column: str | None = None
    continuous_columns: Sequence[str] = ()
    categorical_columns: Sequence[str] = ()
    drop_columns: Sequence[str] = ()


@dataclass
class PipelineConfig:
    """Configuration for orchestrating the synthetic data pipeline."""

    data_path: Path
    output_path: Path
    schema: SchemaConfig
    gan: GANTrainingConfig
    samples_to_generate: int = 256
    metrics_path: Path | None = None
    visualization_path: Path | None = None
    training_history_path: Path | None = None
    testing_report_path: Path | None = None


__all__ = ["PipelineConfig", "SchemaConfig"]
