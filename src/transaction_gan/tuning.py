"""Hyperparameter tuning helpers for transaction GAN models."""

from __future__ import annotations

from dataclasses import replace
from typing import Dict, Iterable, List, Sequence, Tuple

from .evaluation import compare_statistics
from .gan import GANTrainingConfig, SyntheticDataGenerator
from .preprocessing import TransactionPreprocessor


DEFAULT_TUNING_GRID: Tuple[Dict[str, object], ...] = (
    {"epochs": 200},
    {"epochs": 400, "learning_rate": 1e-4},
    {"hidden_dim": 384, "epochs": 300},
)


def _score_synthetic(
    synthetic_matrix: List[List[float]],
    validation_records: List[dict],
    preprocessor: TransactionPreprocessor,
    numeric_features: Sequence[str],
    *,
    epsilon: float = 1e-12,
) -> float:
    synthetic_records = preprocessor.inverse_transform(synthetic_matrix)
    evaluation = compare_statistics(
        validation_records,
        synthetic_records,
        numeric_features=numeric_features,
    )
    if not evaluation.ks_statistics:
        return float("inf")
    return sum(evaluation.ks_statistics.values()) / (len(evaluation.ks_statistics) + epsilon)


def auto_tune_gan(
    processed_train: List[List[float]],
    processed_validation: List[List[float]],
    validation_records: List[dict],
    *,
    base_config: GANTrainingConfig,
    preprocessor: TransactionPreprocessor,
    categorical_slices: Iterable[slice],
    numeric_features: Sequence[str],
    tuning_grid: Sequence[Dict[str, object]] | None = None,
) -> Tuple[GANTrainingConfig, List[Dict[str, object]]]:
    """Evaluate multiple GAN configs and return the best-performing one."""

    if not processed_validation or not validation_records:
        return base_config, []

    grid = tuning_grid or DEFAULT_TUNING_GRID
    history: List[Dict[str, object]] = []
    best_config = base_config
    best_score = float("inf")

    for overrides in grid:
        trial_config = replace(base_config, **overrides)
        generator = SyntheticDataGenerator(
            len(processed_train[0]),
            trial_config,
            categorical_slices=categorical_slices,
        )
        generator.train(processed_train)
        synthetic_matrix = generator.generate(len(processed_validation))
        score = _score_synthetic(
            synthetic_matrix,
            validation_records,
            preprocessor=preprocessor,
            numeric_features=numeric_features,
        )
        history.append({"overrides": overrides, "ks_score": score})
        if score < best_score:
            best_score = score
            best_config = trial_config

    return best_config, history


__all__ = ["auto_tune_gan", "DEFAULT_TUNING_GRID"]
