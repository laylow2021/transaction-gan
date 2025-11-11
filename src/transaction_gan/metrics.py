"""Auxiliary ML metrics for evaluating synthetic vs real data quality."""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


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


__all__ = ["real_vs_synthetic_auc"]
