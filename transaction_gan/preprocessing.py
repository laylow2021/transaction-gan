"""Data preprocessing utilities for the transaction GAN pipeline."""

from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Sequence

from .data_loader import TransactionRecord
from .geo import GeoResolver

DATE_FORMATS = [
    "%Y-%m-%d",
    "%Y/%m/%d",
    "%d-%m-%Y",
    "%m/%d/%Y",
    "%Y-%m-%d %H:%M:%S",
]


def _parse_date(value: object) -> _dt.date | None:
    if value is None:
        return None
    if isinstance(value, _dt.datetime):
        return value.date()
    if isinstance(value, _dt.date):
        return value
    text = str(value).strip()
    if not text:
        return None
    try:
        return _dt.datetime.fromisoformat(text).date()
    except ValueError:
        pass
    for fmt in DATE_FORMATS:
        try:
            return _dt.datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    return None


@dataclass
class TransactionPreprocessor:
    """Normalise mixed-type features and provide reversible transforms."""

    continuous_features: Sequence[str]
    categorical_features: Sequence[str]
    date_feature: str | None = None
    geo_feature: str | None = None
    id_feature: str | None = None
    drop_features: Sequence[str] = ()
    geo_resolver: GeoResolver = field(default_factory=GeoResolver)

    continuous_stats: Dict[str, Dict[str, float]] = field(init=False, default_factory=dict)
    categorical_levels: Dict[str, List[str]] = field(init=False, default_factory=dict)
    feature_slices: Dict[str, slice] = field(init=False, default_factory=dict)
    date_stats: Dict[str, float] | None = field(init=False, default=None)
    geo_stats: Dict[str, Dict[str, float]] | None = field(init=False, default=None)
    output_dim: int = field(init=False, default=0)

    def fit(self, data: List[TransactionRecord]) -> "TransactionPreprocessor":
        self.continuous_stats = {}
        self.categorical_levels = {}
        self.feature_slices = {}
        self.date_stats = None
        self.geo_stats = None

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

        if self.date_feature:
            ordinals = [date.toordinal() for row in data if (date := _parse_date(row.get(self.date_feature)))]
            if ordinals:
                mean = sum(ordinals) / len(ordinals)
                variance = sum((value - mean) ** 2 for value in ordinals) / len(ordinals)
                std = variance ** 0.5 or 1.0
                slice_ = slice(start, start + 1)
                self.date_stats = {"mean": mean, "std": std, "slice": slice_}
                self.feature_slices[self.date_feature] = slice_
                start += 1

        if self.geo_feature:
            lat_values: List[float] = []
            lon_values: List[float] = []
            for row in data:
                lat, lon = self.geo_resolver.resolve(row.get(self.geo_feature))
                lat_values.append(lat)
                lon_values.append(lon)
            if lat_values and lon_values:
                def _stats(values: List[float]) -> Dict[str, float]:
                    mean = sum(values) / len(values)
                    variance = sum((value - mean) ** 2 for value in values) / len(values)
                    std = variance ** 0.5 or 1.0
                    return {"mean": mean, "std": std}

                lat_stats = _stats(lat_values)
                lon_stats = _stats(lon_values)
                lat_slice = slice(start, start + 1)
                lon_slice = slice(start + 1, start + 2)
                self.geo_stats = {
                    "lat": {**lat_stats, "slice": lat_slice},
                    "lon": {**lon_stats, "slice": lon_slice},
                }
                self.feature_slices[f"{self.geo_feature}_lat"] = lat_slice
                self.feature_slices[f"{self.geo_feature}_lon"] = lon_slice
                start += 2

        for column in self.categorical_features:
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

            if self.date_stats and self.date_feature:
                date = _parse_date(row.get(self.date_feature))
                ordinal = date.toordinal() if date else self.date_stats["mean"]
                normalised = (ordinal - self.date_stats["mean"]) / self.date_stats["std"]
                slice_ = self.date_stats["slice"]
                features[slice_.start] = normalised

            if self.geo_stats and self.geo_feature:
                lat, lon = self.geo_resolver.resolve(row.get(self.geo_feature))
                for key, value in zip(("lat", "lon"), (lat, lon)):
                    stats = self.geo_stats[key]
                    normalised = (value - stats["mean"]) / stats["std"]
                    slice_ = stats["slice"]
                    features[slice_.start] = normalised

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

            if self.date_stats and self.date_feature:
                slice_ = self.date_stats["slice"]
                value = vector[slice_.start] * self.date_stats["std"] + self.date_stats["mean"]
                ordinal = int(round(value))
                date_value = _dt.date.fromordinal(max(ordinal, 1))
                row[self.date_feature] = date_value.isoformat()

            if self.geo_stats and self.geo_feature:
                lat_slice = self.geo_stats["lat"]["slice"]
                lon_slice = self.geo_stats["lon"]["slice"]
                latitude = vector[lat_slice.start] * self.geo_stats["lat"]["std"] + self.geo_stats["lat"]["mean"]
                longitude = vector[lon_slice.start] * self.geo_stats["lon"]["std"] + self.geo_stats["lon"]["mean"]
                row[f"{self.geo_feature}_lat"] = round(latitude, 5)
                row[f"{self.geo_feature}_lon"] = round(longitude, 5)
                row[self.geo_feature] = self.geo_resolver.reverse(latitude, longitude)

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


__all__ = ["TransactionPreprocessor"]
