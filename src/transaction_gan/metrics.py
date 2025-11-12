"""Auxiliary ML metrics for evaluating synthetic vs real data quality."""

from __future__ import annotations

from typing import Dict, Iterable, Sequence, Tuple

import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


def _prepare_matrix(matrix: Sequence[Sequence[float]], limit: int | None) -> np.ndarray:
    arr = np.asarray(matrix, dtype=np.float32)
    if limit is not None and len(arr) > limit:
        arr = arr[:limit]
    return arr


def real_vs_synthetic_auc(
    real_matrix: Sequence[Sequence[float]],
    synthetic_matrix: Sequence[Sequence[float]],
    *,
    max_samples: int = 5000,
    seed: int = 42,
) -> float:
    """Train a simple discriminator and report ROC-AUC (closer to 0.5 is better)."""

    if not real_matrix or not synthetic_matrix:
        return float("nan")

    half_limit = max_samples // 2 if max_samples else None
    real = _prepare_matrix(real_matrix, half_limit)
    synth = _prepare_matrix(synthetic_matrix, half_limit)

    min_len = min(len(real), len(synth))
    if min_len == 0:
        return float("nan")

    real = real[:min_len]
    synth = synth[:min_len]

    X = np.vstack([real, synth])
    y = np.concatenate([np.ones(len(real)), np.zeros(len(synth))])

    clf = LogisticRegression(max_iter=1000, random_state=seed)
    clf.fit(X, y)
    probs = clf.predict_proba(X)[:, 1]
    return float(roc_auc_score(y, probs))


def _features_and_labels(
    records: Sequence[dict],
    target_column: str,
    *,
    limit: int | None = None,
) -> Tuple[list, list]:
    features = []
    labels = []
    for row in records:
        value = row.get(target_column)
        if value in (None, ""):
            continue
        labels.append(str(value))
        features.append({key: row[key] for key in row if key != target_column})
        if limit and len(features) >= limit:
            break
    return features, labels


def tstr_scores(
    real_train: Sequence[dict],
    synthetic_records: Sequence[dict],
    holdout_records: Sequence[dict],
    *,
    target_column: str,
    max_samples: int = 5000,
    seed: int = 42,
) -> Dict[str, float]:
    """Train on synthetic data, test on real holdout, return accuracy/F1."""

    if not target_column:
        return {"accuracy": float("nan"), "f1": float("nan")}

    half_limit = max_samples // 2 if max_samples else None
    synth_features, synth_labels = _features_and_labels(synthetic_records, target_column, limit=half_limit)
    holdout_features, holdout_labels = _features_and_labels(holdout_records, target_column, limit=half_limit)
    if len(set(synth_labels)) < 2 or not holdout_features:
        return {"accuracy": float("nan"), "f1": float("nan")}

    vectorizer = DictVectorizer(sparse=True)
    vectorizer.fit(synth_features + holdout_features)
    X_train = vectorizer.transform(synth_features)
    X_test = vectorizer.transform(holdout_features)

    clf = LogisticRegression(max_iter=1000, random_state=seed)
    clf.fit(X_train, synth_labels)
    predictions = clf.predict(X_test)
    accuracy = accuracy_score(holdout_labels, predictions)
    f1 = f1_score(holdout_labels, predictions, average="macro")
    return {"accuracy": float(accuracy), "f1": float(f1)}


__all__ = ["real_vs_synthetic_auc", "tstr_scores"]
