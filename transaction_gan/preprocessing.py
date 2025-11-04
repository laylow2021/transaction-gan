"""Data preprocessing utilities for the transaction GAN pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Sequence

from .data_loader import TransactionRecord


@dataclass
class TransactionPreprocessor:
    """Normalise numeric features and one-hot encode categorical ones."""

    numeric_features: Sequence[str]
    categorical_features: Sequence[str]
    drop_features: Sequence[str] = ()
    numeric_stats: Dict[str, Dict[str, float]] = field(init=False, default_factory=dict)
    categorical_levels: Dict[str, List[str]] = field(init=False, default_factory=dict)
    feature_slices: Dict[str, slice] = field(init=False, default_factory=dict)

    def fit(self, data: List[TransactionRecord]) -> "TransactionPreprocessor":
        self.numeric_stats = {}
        self.categorical_levels = {}
        self.feature_slices = {}

        start = 0
        for column in self.numeric_features:
            values = [float(row[column]) for row in data if column in row]
            if not values:
                continue
            mean = sum(values) / len(values)
            variance = sum((value - mean) ** 2 for value in values) / len(values)
            std = variance ** 0.5 or 1.0
            self.numeric_stats[column] = {"mean": mean, "std": std}
            self.feature_slices[column] = slice(start, start + 1)
            start += 1

        for column in self.categorical_features:
            levels = sorted({str(row.get(column, "")) for row in data})
            if "" not in levels:
                levels.append("")
            self.categorical_levels[column] = levels
            self.feature_slices[column] = slice(start, start + len(levels))
            start += len(levels)

        self.output_dim = start
        return self

    def transform(self, data: List[TransactionRecord]) -> List[List[float]]:
        if not self.feature_slices:
            raise RuntimeError("The preprocessor must be fitted before calling transform().")

        transformed: List[List[float]] = []
        for row in data:
            features: List[float] = [0.0] * self.output_dim

            for column, stats in self.numeric_stats.items():
                value = float(row.get(column, stats["mean"]))
                normalised = (value - stats["mean"]) / stats["std"]
                slice_ = self.feature_slices[column]
                features[slice_.start] = normalised

            for column, levels in self.categorical_levels.items():
                value = str(row.get(column, ""))
                if value not in levels:
                    index = len(levels) - 1  # map to empty placeholder
                else:
                    index = levels.index(value)
                slice_ = self.feature_slices[column]
                offset = slice_.start + index
                features[offset] = 1.0

            transformed.append(features)

        return transformed

    def fit_transform(self, data: List[TransactionRecord]) -> List[List[float]]:
        return self.fit(data).transform(data)

    def inverse_transform(self, matrix: List[List[float]]) -> List[TransactionRecord]:
        if not self.feature_slices:
            raise RuntimeError("The preprocessor must be fitted before calling inverse_transform().")

        records: List[TransactionRecord] = []
        for vector in matrix:
            row: TransactionRecord = {}
            for column, stats in self.numeric_stats.items():
                slice_ = self.feature_slices[column]
                value = vector[slice_.start] * stats["std"] + stats["mean"]
                row[column] = round(value, 5)
            for column, levels in self.categorical_levels.items():
                slice_ = self.feature_slices[column]
                window = vector[slice_.start : slice_.stop]
                if not window:
                    continue
                max_index = max(range(len(window)), key=lambda idx: window[idx])
                row[column] = levels[max_index]
            records.append(row)
        return records


__all__ = ["TransactionPreprocessor"]
