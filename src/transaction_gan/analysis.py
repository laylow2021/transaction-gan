"""Exploratory analysis utilities for transaction data."""

from __future__ import annotations

from collections import Counter
from typing import Dict, Iterable, List, Mapping

from .data_loader import TransactionRecord

NumericSummary = Mapping[str, Dict[str, float]]
CategoricalSummary = Mapping[str, Dict[str, int]]


def _to_float_list(data: List[TransactionRecord], column: str) -> List[float]:
    values: List[float] = []
    for row in data:
        try:
            values.append(float(row[column]))
        except (KeyError, TypeError, ValueError):
            continue
    return values


def describe_transactions(
    data: List[TransactionRecord],
    *,
    numeric_features: Iterable[str],
    categorical_features: Iterable[str],
) -> Dict[str, Mapping[str, Mapping[str, float]]]:
    """Summarise the transaction dataset."""

    summary: Dict[str, Mapping[str, Mapping[str, float]]] = {}

    numeric_summary: Dict[str, Dict[str, float]] = {}
    for column in numeric_features:
        values = _to_float_list(data, column)
        if not values:
            continue
        count = len(values)
        mean = sum(values) / count
        variance = sum((value - mean) ** 2 for value in values) / count
        numeric_summary[column] = {
            "mean": mean,
            "std": variance ** 0.5,
            "min": min(values),
            "max": max(values),
        }

    categorical_summary: Dict[str, Dict[str, int]] = {}
    for column in categorical_features:
        counter = Counter(str(row.get(column, "")) for row in data)
        categorical_summary[column] = {key: counter[key] for key in sorted(counter)}

    summary["numeric"] = numeric_summary
    summary["categorical"] = categorical_summary

    return summary


__all__ = ["describe_transactions"]
