"""Configuration dataclasses for the transaction GAN pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .gan import GANTrainingConfig


@dataclass
class PipelineConfig:
    """Configuration for orchestrating the synthetic data pipeline."""

    data_path: Path
    output_path: Path
    numeric_features: Iterable[str]
    categorical_features: Iterable[str]
    drop_features: Iterable[str]
    gan: GANTrainingConfig
    samples_to_generate: int = 256


__all__ = ["PipelineConfig"]
