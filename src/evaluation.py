"""Evaluation metrics for synthetic transaction data."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List

from .data_loader import TransactionRecord


@dataclass
class EvaluationResult:
    """Container for distribution similarity metrics."""

    ks_statistics: Dict[str, float]
    wasserstein_distances: Dict[str, float]

    def to_dict(self) -> Dict[str, Dict[str, float]]:
        return {
            "ks_statistics": self.ks_statistics,
            "wasserstein_distances": self.wasserstein_distances,
        }


def _numeric_values(data: List[TransactionRecord], column: str) -> List[float]:
    values: List[float] = []
    for row in data:
        try:
            values.append(float(row[column]))
        except (KeyError, TypeError, ValueError):
            continue
    return values


def _kolmogorov_smirnov(real: List[float], synthetic: List[float]) -> float:
    real_sorted = sorted(real)
    synthetic_sorted = sorted(synthetic)
    n = len(real_sorted)
    m = len(synthetic_sorted)
    i = j = 0
    cdf_real = cdf_synth = 0.0
    max_diff = 0.0

    while i < n or j < m:
        if j == m or (i < n and real_sorted[i] <= synthetic_sorted[j]):
            value = real_sorted[i]
        else:
            value = synthetic_sorted[j]

        while i < n and real_sorted[i] == value:
            i += 1
        while j < m and synthetic_sorted[j] == value:
            j += 1

        cdf_real = i / n
        cdf_synth = j / m
        diff = abs(cdf_real - cdf_synth)
        if diff > max_diff:
            max_diff = diff

    return max_diff


def _wasserstein(real: List[float], synthetic: List[float]) -> float:
    real_sorted = sorted(real)
    synthetic_sorted = sorted(synthetic)
    n = len(real_sorted)
    m = len(synthetic_sorted)
    if not n or not m:
        return float("nan")

    events: List[tuple[float, float]] = []
    weight_real = 1.0 / n
    weight_synth = 1.0 / m
    events.extend((value, weight_real) for value in real_sorted)
    events.extend((value, -weight_synth) for value in synthetic_sorted)
    events.sort(key=lambda item: item[0])

    distance = 0.0
    cumulative = 0.0
    previous_value = events[0][0]
    for value, weight in events:
        distance += abs(cumulative) * (value - previous_value)
        cumulative += weight
        previous_value = value

    return distance


def compare_statistics(
    real: List[TransactionRecord],
    synthetic: List[TransactionRecord],
    *,
    numeric_features: Iterable[str],
) -> EvaluationResult:
    """Compare distribution statistics between real and synthetic datasets."""

    ks_stats: Dict[str, float] = {}
    wasserstein: Dict[str, float] = {}

    for column in numeric_features:
        real_values = _numeric_values(real, column)
        synthetic_values = _numeric_values(synthetic, column)
        if not real_values or not synthetic_values:
            continue
        ks_stats[column] = _kolmogorov_smirnov(real_values, synthetic_values)
        wasserstein[column] = _wasserstein(real_values, synthetic_values)

    return EvaluationResult(ks_statistics=ks_stats, wasserstein_distances=wasserstein)


def build_quality_report(
    result: EvaluationResult,
    *,
    ks_threshold: float = 0.15,
    wasserstein_threshold: float = 0.5,
) -> Dict[str, Dict[str, object]]:
    """Produce human-readable pass/fail testing metrics for GAN output."""

    summary: Dict[str, Dict[str, object]] = {}
    columns = set(result.ks_statistics) | set(result.wasserstein_distances)
    for column in columns:
        ks_value = result.ks_statistics.get(column, float("nan"))
        wasserstein_value = result.wasserstein_distances.get(column, float("nan"))
        summary[column] = {
            "ks_statistic": ks_value,
            "wasserstein_distance": wasserstein_value,
            "ks_pass": ks_value <= ks_threshold if ks_value == ks_value else False,
            "wasserstein_pass": wasserstein_value <= wasserstein_threshold
            if wasserstein_value == wasserstein_value
            else False,
        }
    return summary


__all__ = ["EvaluationResult", "build_quality_report", "compare_statistics"]
