"""PyTorch CTGAN integration via the SDV library."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence
import warnings

import pandas as pd
import torch

try:  # pragma: no cover - import guard for optional dependency
    from sdv.metadata import SingleTableMetadata
    from sdv.single_table import CTGANSynthesizer
except ModuleNotFoundError as exc:  # pragma: no cover - guidance for missing deps
    raise ModuleNotFoundError(
        "sdv is required to run the CTGAN-based synthesizer. "
        "Install it via `pip install sdv[torch]`."
    ) from exc


@dataclass
class GANTrainingConfig:
    """Expose the subset of CTGAN hyperparameters we surface to users."""

    noise_dim: int = 128  # maps to embedding_dim in SDV
    hidden_dim: int = 256  # used for both generator/discriminator layer widths
    epochs: int = 300
    learning_rate: float = 2e-4
    batch_size: int = 500
    pac: int = 10
    generator_dims: Sequence[int] | None = None
    discriminator_dims: Sequence[int] | None = None
    use_cuda: bool = False


class SyntheticDataGenerator:
    """Wrapper that adapts our matrix-based pipeline to SDV's CTGAN API."""

    def __init__(
        self,
        output_dim: int,
        config: GANTrainingConfig | None = None,
        *,
        categorical_slices: Iterable[slice] | None = None,
    ) -> None:
        self.config = config or GANTrainingConfig()
        self.output_dim = output_dim
        self.column_names = [f"feature_{index}" for index in range(output_dim)]
        self.metadata: SingleTableMetadata | None = None
        self.model: CTGANSynthesizer | None = None
        self._history: Dict[str, List[float]] = {"epochs": [], "generator": [], "discriminator": []}
        self._trained = False
        self._categorical_slices = list(categorical_slices or [])  # kept for compatibility

    def _build_model(self) -> CTGANSynthesizer:
        if self.metadata is None:
            raise RuntimeError("Metadata must be initialised before building the CTGAN model.")
        generator_dims = tuple(self.config.generator_dims) if self.config.generator_dims else (
            self.config.hidden_dim,
            self.config.hidden_dim,
        )
        discriminator_dims = tuple(self.config.discriminator_dims) if self.config.discriminator_dims else (
            self.config.hidden_dim,
            self.config.hidden_dim,
        )
        use_cuda = self.config.use_cuda and torch.cuda.is_available()
        return CTGANSynthesizer(
            metadata=self.metadata,
            epochs=self.config.epochs,
            batch_size=self.config.batch_size,
            embedding_dim=self.config.noise_dim,
            generator_dim=generator_dims,
            discriminator_dim=discriminator_dims,
            generator_lr=self.config.learning_rate,
            discriminator_lr=self.config.learning_rate,
            pac=self.config.pac,
            cuda=use_cuda,
        )

    def _to_dataframe(self, matrix: Sequence[Sequence[float]]) -> pd.DataFrame:
        if not matrix:
            raise ValueError("CTGAN cannot be trained on an empty dataset.")
        return pd.DataFrame(matrix, columns=self.column_names, dtype="float32")

    def train(self, matrix: Sequence[Sequence[float]]) -> None:
        df = self._to_dataframe(matrix)
        if self.metadata is None:
            metadata = SingleTableMetadata()
            metadata.detect_from_dataframe(df)
            self.metadata = metadata
        if self.model is None:
            self.model = self._build_model()
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="PerformanceAlert: Using the CTGANSynthesizer on this data is not recommended.*",
            )
            self.model.fit(df)
        self._trained = True
        try:
            loss_df = self.model.get_loss_values()
        except AttributeError:  # SDV < 1.10
            self._history = {"epochs": [], "generator": [], "discriminator": []}
        else:
            records = loss_df.to_dict("records")
            epochs: List[float] = []
            generator: List[float] = []
            discriminator: List[float] = []
            for index, record in enumerate(records):
                epochs.append(float(record.get("Epoch", index + 1)))
                generator.append(float(record.get("Generator Loss", 0.0)))
                discriminator.append(float(record.get("Discriminator Loss", 0.0)))
            self._history = {
                "epochs": epochs,
                "generator": generator,
                "discriminator": discriminator,
            }

    def generate(self, rows: int) -> List[List[float]]:
        if not self._trained:
            raise RuntimeError("CTGAN must be trained before generating samples.")
        if self.model is None:
            raise RuntimeError("CTGAN model is unavailable.")
        samples = self.model.sample(num_rows=rows)
        return samples[self.column_names].to_numpy(dtype="float32").tolist()

    def get_training_history(self) -> Dict[str, List[float]]:
        return {
            "epochs": list(self._history.get("epochs", [])),
            "generator": list(self._history.get("generator", [])),
            "discriminator": list(self._history.get("discriminator", [])),
        }


__all__ = ["GANTrainingConfig", "SyntheticDataGenerator"]
