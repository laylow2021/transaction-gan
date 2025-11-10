"""Independent testing utilities comparing real vs synthetic datasets."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Sequence

from .config import SchemaConfig
from .data_loader import TransactionRecord


def _numeric_values(records: List[TransactionRecord], column: str) -> List[float]:
    values: List[float] = []
    for row in records:
        try:
            values.append(float(row[column]))
        except (TypeError, ValueError, KeyError):
            continue
    return values


def compare_categorical_frequencies(
    real: List[TransactionRecord],
    synthetic: List[TransactionRecord],
    *,
    categorical_columns: Sequence[str],
    top_n: int | None = None,
) -> Dict[str, Dict[str, Dict[str, int]]]:
    """Return frequency tables for categorical columns."""

    summary: Dict[str, Dict[str, Dict[str, int]]] = {}
    for column in categorical_columns:
        real_counter = Counter(str(row.get(column, "")) for row in real)
        synth_counter = Counter(str(row.get(column, "")) for row in synthetic)
        if top_n:
            real_items = real_counter.most_common(top_n)
            synth_items = synth_counter.most_common(top_n)
        else:
            real_items = sorted(real_counter.items())
            synth_items = sorted(synth_counter.items())

        summary[column] = {
            "real": dict(real_items),
            "synthetic": dict(synth_items),
        }
    return summary


def _histogram(values: List[float], bins: int = 20) -> Dict[str, List[float]]:
    if not values:
        return {"bins": [], "counts": []}
    minimum = min(values)
    maximum = max(values)
    if minimum == maximum:
        minimum -= 0.5
        maximum += 0.5

    bin_width = (maximum - minimum) / bins
    edges = [minimum + bin_width * i for i in range(bins + 1)]
    counts = [0] * bins
    for value in values:
        index = int((value - minimum) / bin_width)
        index = min(max(index, 0), bins - 1)
        counts[index] += 1
    return {"bins": edges, "counts": counts}


def compare_continuous_histograms(
    real: List[TransactionRecord],
    synthetic: List[TransactionRecord],
    *,
    continuous_columns: Sequence[str],
    bins: int = 20,
) -> Dict[str, Dict[str, Dict[str, List[float]]]]:
    """Compare histogram counts for continuous variables."""

    summary: Dict[str, Dict[str, Dict[str, List[float]]]] = {}
    for column in continuous_columns:
        real_hist = _histogram(_numeric_values(real, column), bins=bins)
        synth_hist = _histogram(_numeric_values(synthetic, column), bins=bins)
        summary[column] = {"real": real_hist, "synthetic": synth_hist}
    return summary


def compare_grouped_histograms(
    real: List[TransactionRecord],
    synthetic: List[TransactionRecord],
    *,
    continuous_columns: Sequence[str],
    categorical_columns: Sequence[str],
    bins: int = 15,
    top_k: int = 5,
) -> Dict[str, Dict[str, Dict[str, Dict[str, List[float]]]]]:
    """Compare histograms for each continuous variable grouped by categorical values."""

    report: Dict[str, Dict[str, Dict[str, Dict[str, List[float]]]]] = {}
    for categorical in categorical_columns:
        category_values = Counter(str(row.get(categorical, "")) for row in real + synthetic)
        selected = [name for name, _ in category_values.most_common(top_k)]
        cat_entry: Dict[str, Dict[str, Dict[str, List[float]]]] = {}
        for continuous in continuous_columns:
            cont_entry: Dict[str, Dict[str, List[float]]] = {}
            for category in selected:
                real_subset = [
                    float(row[continuous])
                    for row in real
                    if row.get(categorical, "") == category and continuous in row
                ]
                synth_subset = [
                    float(row[continuous])
                    for row in synthetic
                    if row.get(categorical, "") == category and continuous in row
                ]
                cont_entry[category] = {
                    "real": _histogram(real_subset, bins=bins),
                    "synthetic": _histogram(synth_subset, bins=bins),
                }
            cat_entry[continuous] = cont_entry
        report[categorical] = cat_entry
    return report


def generate_testing_report(
    real: List[TransactionRecord],
    synthetic: List[TransactionRecord],
    *,
    schema: SchemaConfig,
    bins: int = 20,
    top_k: int = 5,
) -> Dict[str, object]:
    """Produce a comprehensive testing summary comparing key distributions."""

    categorical = tuple(schema.categorical_columns)
    continuous = tuple(schema.continuous_columns)

    frequencies = compare_categorical_frequencies(
        real,
        synthetic,
        categorical_columns=categorical,
        top_n=top_k,
    )
    histograms = compare_continuous_histograms(
        real,
        synthetic,
        continuous_columns=continuous,
        bins=bins,
    )
    grouped = compare_grouped_histograms(
        real,
        synthetic,
        continuous_columns=continuous,
        categorical_columns=categorical,
        bins=max(5, bins // 2),
        top_k=top_k,
    )

    return {
        "categorical_frequencies": frequencies,
        "continuous_histograms": histograms,
        "grouped_histograms": grouped,
        "bins": bins,
        "top_k": top_k,
    }


__all__ = [
    "compare_categorical_frequencies",
    "compare_continuous_histograms",
    "compare_grouped_histograms",
    "generate_testing_report",
]
