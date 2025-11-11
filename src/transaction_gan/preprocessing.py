"""Data preprocessing utilities for the transaction GAN pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Set

from .data_loader import TransactionRecord


@dataclass
class TransactionPreprocessor:
    """Normalise mixed-type features and provide reversible transforms."""

    continuous_features: Sequence[str]
    categorical_features: Sequence[str]
    id_feature: str | None = None
    drop_features: Sequence[str] = ()
    hierarchical_categorical_groups: Sequence[Sequence[str]] = ()

    continuous_stats: Dict[str, Dict[str, float]] = field(init=False, default_factory=dict)
    categorical_levels: Dict[str, List[str]] = field(init=False, default_factory=dict)
    feature_slices: Dict[str, slice] = field(init=False, default_factory=dict)
    hierarchical_group_configs: List[Dict[str, object]] = field(init=False, default_factory=list)
    hierarchical_columns: Set[str] = field(init=False, default_factory=set)
    output_dim: int = field(init=False, default=0)

    def fit(self, data: List[TransactionRecord]) -> "TransactionPreprocessor":
        self.continuous_stats = {}
        self.categorical_levels = {}
        self.feature_slices = {}
        self.hierarchical_group_configs = []
        self.hierarchical_columns = {
            column for group in self.hierarchical_categorical_groups for column in group
        }

        start = 0

        for column in self.continuous_features:
            values = [float(row[column]) for row in data if column in row]
            if not values:
                continue
            mean = sum(values) / len(values)
            variance = sum((value - mean) ** 2 for value in values) / len(values)
            std = variance ** 0.5 or 1.0
            self.continuous_stats[column] = {"mean": mean, "std": std, "slice": slice(start, start + 1)}
            self.feature_slices[column] = slice(start, start + 1)
            start += 1

        for group in self.hierarchical_categorical_groups:
            columns = tuple(group)
            if not columns:
                continue
            combinations = {
                tuple(str(row.get(column, "")) for column in columns) for row in data
            }
            fallback = tuple("" for _ in columns)
            levels = sorted(combinations)
            if fallback in levels:
                levels.pop(levels.index(fallback))
            levels.append(fallback)
            slice_ = slice(start, start + len(levels))
            self.hierarchical_group_configs.append(
                {"columns": columns, "levels": levels, "slice": slice_}
            )
            start += len(levels)

        for column in self.categorical_features:
            if column in self.hierarchical_columns:
                continue
            levels = sorted({str(row.get(column, "")) for row in data})
            if "" not in levels:
                levels.append("")
            slice_ = slice(start, start + len(levels))
            self.categorical_levels[column] = levels
            self.feature_slices[column] = slice_
            start += len(levels)

        self.output_dim = start
        return self

    def transform(self, data: List[TransactionRecord]) -> List[List[float]]:
        if not self.feature_slices:
            raise RuntimeError("The preprocessor must be fitted before calling transform().")

        transformed: List[List[float]] = []
        for row in data:
            features: List[float] = [0.0] * self.output_dim

            for column, stats in self.continuous_stats.items():
                value = float(row.get(column, stats["mean"]))
                normalised = (value - stats["mean"]) / stats["std"]
                slice_ = stats["slice"]
                features[slice_.start] = normalised

            for config in self.hierarchical_group_configs:
                key = tuple(str(row.get(column, "")) for column in config["columns"])
                try:
                    index = config["levels"].index(key)
                except ValueError:
                    index = len(config["levels"]) - 1
                slice_ = config["slice"]
                features[slice_.start + index] = 1.0

            for column, levels in self.categorical_levels.items():
                value = str(row.get(column, ""))
                try:
                    index = levels.index(value)
                except ValueError:
                    index = len(levels) - 1
                slice_ = self.feature_slices[column]
                features[slice_.start + index] = 1.0

            transformed.append(features)

        return transformed

    def fit_transform(self, data: List[TransactionRecord]) -> List[List[float]]:
        return self.fit(data).transform(data)

    def inverse_transform(self, matrix: List[List[float]]) -> List[TransactionRecord]:
        if not self.feature_slices:
            raise RuntimeError("The preprocessor must be fitted before calling inverse_transform().")

        records: List[TransactionRecord] = []
        for index, vector in enumerate(matrix, start=1):
            row: TransactionRecord = {}
            for column, stats in self.continuous_stats.items():
                slice_ = stats["slice"]
                value = vector[slice_.start] * stats["std"] + stats["mean"]
                row[column] = round(value, 5)

            for config in self.hierarchical_group_configs:
                slice_ = config["slice"]
                window = vector[slice_.start : slice_.stop]
                if not window:
                    continue
                max_index = max(range(len(window)), key=lambda idx: window[idx])
                values = config["levels"][max_index]
                for column, value in zip(config["columns"], values):
                    row[column] = value

            for column, levels in self.categorical_levels.items():
                slice_ = self.feature_slices[column]
                window = vector[slice_.start : slice_.stop]
                if not window:
                    continue
                max_index = max(range(len(window)), key=lambda idx: window[idx])
                row[column] = levels[max_index]

            if self.id_feature and self.id_feature not in row:
                row[self.id_feature] = f"synthetic_{index}"

            records.append(row)

        return records

    def categorical_input_slices(self) -> List[slice]:
        """Return slices corresponding to categorical one-hot segments."""

        slices: List[slice] = []
        for column in self.categorical_levels:
            slices.append(self.feature_slices[column])
        for config in self.hierarchical_group_configs:
            slices.append(config["slice"])
        return slices


__all__ = ["TransactionPreprocessor"]
