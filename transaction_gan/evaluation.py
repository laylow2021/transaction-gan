"""Evaluation metrics for synthetic transaction data."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List

from .data_loader import TransactionRecord


@dataclass
class EvaluationResult:
    """Container for evaluation results."""

    column_wise_mean_diff: Dict[str, float]
    column_wise_std_diff: Dict[str, float]


def _numeric_values(data: List[TransactionRecord], column: str) -> List[float]:
    values: List[float] = []
    for row in data:
        try:
            values.append(float(row[column]))
        except (KeyError, TypeError, ValueError):
            continue
    return values


def compare_statistics(
    real: List[TransactionRecord],
    synthetic: List[TransactionRecord],
    *,
    numeric_features: Iterable[str],
) -> EvaluationResult:
    """Compare distribution statistics between real and synthetic datasets."""

    mean_diff: Dict[str, float] = {}
    std_diff: Dict[str, float] = {}

    for column in numeric_features:
        real_values = _numeric_values(real, column)
        synthetic_values = _numeric_values(synthetic, column)
        if not real_values or not synthetic_values:
            continue
        real_mean = sum(real_values) / len(real_values)
        synthetic_mean = sum(synthetic_values) / len(synthetic_values)
        mean_diff[column] = abs(real_mean - synthetic_mean)

        real_var = sum((value - real_mean) ** 2 for value in real_values) / len(real_values)
        synthetic_var = sum((value - synthetic_mean) ** 2 for value in synthetic_values) / len(synthetic_values)
        std_diff[column] = abs(real_var ** 0.5 - synthetic_var ** 0.5)

    return EvaluationResult(column_wise_mean_diff=mean_diff, column_wise_std_diff=std_diff)


__all__ = ["EvaluationResult", "compare_statistics"]
