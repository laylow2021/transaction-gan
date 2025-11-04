"""End-to-end pipeline for analysing and generating transaction data."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List

from .analysis import describe_transactions
from .config import PipelineConfig
from .data_loader import load_transactions
from .evaluation import compare_statistics
from .gan import GANTrainingConfig, SyntheticDataGenerator
from .preprocessing import TransactionPreprocessor


DEFAULT_NUMERIC_FEATURES = ["amount", "customer_age"]
DEFAULT_CATEGORICAL_FEATURES = ["merchant_category", "transaction_type"]
DEFAULT_DROP_FEATURES = ["transaction_id"]
DEFAULT_SAMPLES = 512
DEFAULT_OUTPUT_PATH = Path("data/synthetic_transactions.csv")


def _prepare_preprocessor(
    *,
    numeric_features: Iterable[str],
    categorical_features: Iterable[str],
    drop_features: Iterable[str],
) -> TransactionPreprocessor:
    return TransactionPreprocessor(
        numeric_features=list(numeric_features),
        categorical_features=list(categorical_features),
        drop_features=list(drop_features),
    )


def generate_synthetic_transactions(
    data_path: Path | str,
    *,
    output_path: Path | str = DEFAULT_OUTPUT_PATH,
    numeric_features: Iterable[str] = DEFAULT_NUMERIC_FEATURES,
    categorical_features: Iterable[str] = DEFAULT_CATEGORICAL_FEATURES,
    drop_features: Iterable[str] = DEFAULT_DROP_FEATURES,
    gan_config: GANTrainingConfig | None = None,
    samples_to_generate: int = DEFAULT_SAMPLES,
) -> Dict[str, object]:
    """Run the full analysis, training, generation, and evaluation pipeline."""

    records = load_transactions(data_path)

    analysis_summary = describe_transactions(
        records,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
    )

    preprocessor = _prepare_preprocessor(
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        drop_features=drop_features,
    )

    processed = preprocessor.fit_transform(records)
    gan = SyntheticDataGenerator(len(processed[0]), gan_config)
    gan.train(processed)

    synthetic_matrix = gan.generate(samples_to_generate)
    synthetic_records = preprocessor.inverse_transform(synthetic_matrix)

    evaluation = compare_statistics(
        records,
        synthetic_records,
        numeric_features=numeric_features,
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(numeric_features) + list(categorical_features)
    with output_path.open("w", newline="", encoding="utf8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in synthetic_records:
            writer.writerow({name: row.get(name, "") for name in fieldnames})

    preview = synthetic_records[:5]

    return {
        "analysis": analysis_summary,
        "evaluation": evaluation,
        "synthetic_path": str(output_path),
        "synthetic_preview": preview,
    }


def load_config(path: Path | str) -> PipelineConfig:
    """Load a pipeline configuration from a JSON file."""

    with open(path, "r", encoding="utf8") as handle:
        raw = json.load(handle)

    config = PipelineConfig(
        data_path=Path(raw["data_path"]),
        output_path=Path(raw.get("output_path", DEFAULT_OUTPUT_PATH)),
        numeric_features=raw.get("numeric_features", DEFAULT_NUMERIC_FEATURES),
        categorical_features=raw.get("categorical_features", DEFAULT_CATEGORICAL_FEATURES),
        drop_features=raw.get("drop_features", DEFAULT_DROP_FEATURES),
        gan=GANTrainingConfig(**raw.get("gan", {})),
        samples_to_generate=int(raw.get("samples_to_generate", DEFAULT_SAMPLES)),
    )
    return config


def run_from_config(config: PipelineConfig) -> Dict[str, object]:
    """Execute the pipeline using a :class:`PipelineConfig`."""

    return generate_synthetic_transactions(
        config.data_path,
        output_path=config.output_path,
        numeric_features=config.numeric_features,
        categorical_features=config.categorical_features,
        drop_features=config.drop_features,
        gan_config=config.gan,
        samples_to_generate=config.samples_to_generate,
    )


__all__ = [
    "DEFAULT_CATEGORICAL_FEATURES",
    "DEFAULT_DROP_FEATURES",
    "DEFAULT_NUMERIC_FEATURES",
    "DEFAULT_OUTPUT_PATH",
    "DEFAULT_SAMPLES",
    "generate_synthetic_transactions",
    "load_config",
    "run_from_config",
]
