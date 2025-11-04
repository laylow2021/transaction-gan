"""Utility functions for loading transaction data."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List, Union

TransactionRecord = Dict[str, str]


def load_transactions(path: Union[str, Path]) -> List[TransactionRecord]:
    """Load a CSV file containing transaction data."""

    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Transaction file '{csv_path}' was not found.")

    with csv_path.open("r", newline="", encoding="utf8") as handle:
        reader = csv.DictReader(handle)
        rows = [dict(row) for row in reader]

    if not rows:
        raise ValueError("The transaction dataset is empty.")

    return rows


__all__ = ["load_transactions", "TransactionRecord"]
