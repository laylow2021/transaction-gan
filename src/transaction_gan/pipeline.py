"""End-to-end pipeline for analysing and generating transaction data."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List

from .analysis import describe_transactions
from .config import PipelineConfig, SchemaConfig
from .data_loader import load_transactions
from .evaluation import build_quality_report, compare_statistics
from .gan import GANTrainingConfig, SyntheticDataGenerator
from .preprocessing import TransactionPreprocessor
from .testing import generate_testing_report
from .visualization import (
    plot_numeric_comparison,
    plot_testing_report as plot_testing_visual,
    plot_training_history,
)


DEFAULT_SCHEMA = SchemaConfig(
    id_column="transaction_id",
    geo_column="transaction_address",
    date_column="transaction_date",
    continuous_columns=("amount", "customer_age"),
    categorical_columns=("merchant_category", "transaction_type", "merchant_code", "is_fraud"),
    drop_columns=("transaction_id",),
)
DEFAULT_SAMPLES = 512
DEFAULT_OUTPUT_PATH = Path("data/synthetic_transactions.csv")
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _resolve_schema(
    schema: SchemaConfig | None,
    *,
    continuous_columns: Iterable[str] | None,
    categorical_columns: Iterable[str] | None,
    drop_columns: Iterable[str] | None,
    id_column: str | None,
    geo_column: str | None,
    date_column: str | None,
) -> SchemaConfig:
    if schema:
        return schema

    continuous = tuple(continuous_columns or DEFAULT_SCHEMA.continuous_columns)
    categorical = tuple(categorical_columns or DEFAULT_SCHEMA.categorical_columns)
    drops = tuple(drop_columns or DEFAULT_SCHEMA.drop_columns)

    return SchemaConfig(
        id_column=id_column if id_column is not None else DEFAULT_SCHEMA.id_column,
        geo_column=geo_column if geo_column is not None else DEFAULT_SCHEMA.geo_column,
        date_column=date_column if date_column is not None else DEFAULT_SCHEMA.date_column,
        continuous_columns=continuous,
        categorical_columns=categorical,
        drop_columns=drops,
    )


def _prepare_preprocessor(schema: SchemaConfig) -> TransactionPreprocessor:
    return TransactionPreprocessor(
        continuous_features=list(schema.continuous_columns),
        categorical_features=list(schema.categorical_columns),
        date_feature=schema.date_column,
        geo_feature=schema.geo_column,
        id_feature=schema.id_column,
        drop_features=list(schema.drop_columns),
    )


def _trim_records(records: List[Dict[str, str]], schema: SchemaConfig) -> List[Dict[str, str]]:
    if not schema.drop_columns:
        return list(records)
    return [
        {key: value for key, value in row.items() if key not in schema.drop_columns}
        for row in records
    ]


def _output_fieldnames(schema: SchemaConfig) -> List[str]:
    ordered: List[str] = []

    def _add(column: str | None) -> None:
        if column and column not in ordered:
            ordered.append(column)

    _add(schema.id_column)
    for column in schema.continuous_columns:
        _add(column)
    for column in schema.categorical_columns:
        _add(column)
    _add(schema.date_column)
    if schema.geo_column:
        _add(schema.geo_column)
        _add(f"{schema.geo_column}_lat")
        _add(f"{schema.geo_column}_lon")
    return ordered

def _resolve_input_path(path: Path | str) -> Path:
    raw_path = Path(path)
    if raw_path.is_absolute():
        return raw_path

    cwd_candidate = Path.cwd() / raw_path
    if cwd_candidate.exists():
        return cwd_candidate

    project_candidate = PROJECT_ROOT / raw_path
    if project_candidate.exists():
        return project_candidate

    return cwd_candidate


def _resolve_output_path(path: Path | str) -> Path:
    raw_path = Path(path)
    if raw_path.is_absolute():
        return raw_path

    cwd_candidate = Path.cwd() / raw_path
    if cwd_candidate.parent.exists():
        return cwd_candidate

    return PROJECT_ROOT / raw_path


def generate_synthetic_transactions(
    data_path: Path | str,
    *,
    output_path: Path | str = DEFAULT_OUTPUT_PATH,
    schema: SchemaConfig | None = None,
    numeric_features: Iterable[str] | None = None,
    categorical_features: Iterable[str] | None = None,
    drop_features: Iterable[str] | None = None,
    id_column: str | None = None,
    geo_column: str | None = None,
    date_column: str | None = None,
    gan_config: GANTrainingConfig | None = None,
    samples_to_generate: int = DEFAULT_SAMPLES,
    metrics_path: Path | str | None = None,
    visualization_path: Path | str | None = None,
    training_history_path: Path | str | None = None,
    testing_report_path: Path | str | None = None,
    testing_visualization_path: Path | str | None = None,
) -> Dict[str, object]:
    """Run the full analysis, training, generation, and evaluation pipeline."""

    absolute_data_path = _resolve_input_path(data_path)
    records = load_transactions(absolute_data_path)
    resolved_schema = _resolve_schema(
        schema,
        continuous_columns=numeric_features,
        categorical_columns=categorical_features,
        drop_columns=drop_features,
        id_column=id_column,
        geo_column=geo_column,
        date_column=date_column,
    )

    analysis_summary = describe_transactions(
        records,
        numeric_features=resolved_schema.continuous_columns,
        categorical_features=resolved_schema.categorical_columns,
    )

    training_rows = _trim_records(records, resolved_schema)
    preprocessor = _prepare_preprocessor(resolved_schema)
    processed = preprocessor.fit_transform(training_rows)
    if not processed:
        raise ValueError("No rows were available for training after preprocessing.")

    gan = SyntheticDataGenerator(len(processed[0]), gan_config)
    gan.train(processed)
    training_history = gan.get_training_history()

    synthetic_matrix = gan.generate(samples_to_generate)
    synthetic_records = preprocessor.inverse_transform(synthetic_matrix)
    round_trip_preview = preprocessor.inverse_transform(processed[:5])

    evaluation = compare_statistics(
        training_rows,
        synthetic_records,
        numeric_features=resolved_schema.continuous_columns,
    )
    quality_report = build_quality_report(evaluation)

    testing_report = generate_testing_report(
        training_rows,
        synthetic_records,
        schema=resolved_schema,
    )

    output_path = _resolve_output_path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = _output_fieldnames(resolved_schema)
    with output_path.open("w", newline="", encoding="utf8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in synthetic_records:
            writer.writerow({name: row.get(name, "") for name in fieldnames})

    metrics_path = _resolve_output_path(metrics_path) if metrics_path else output_path.with_suffix(".metrics.json")
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with metrics_path.open("w", encoding="utf8") as handle:
        json.dump(
            {
                "quality_report": quality_report,
                "ks_statistics": evaluation.ks_statistics,
                "wasserstein_distances": evaluation.wasserstein_distances,
            },
            handle,
            indent=2,
        )

    viz_target = _resolve_output_path(visualization_path) if visualization_path else output_path.with_suffix(".png")
    viz_path = plot_numeric_comparison(
        training_rows,
        synthetic_records,
        numeric_features=resolved_schema.continuous_columns,
        output_path=viz_target,
    )

    history_target = (
        _resolve_output_path(training_history_path)
        if training_history_path
        else output_path.with_suffix(".loss.png")
    )
    history_path = plot_training_history(
        training_history,
        output_path=history_target,
    )

    testing_target = (
        _resolve_output_path(testing_report_path)
        if testing_report_path
        else output_path.with_suffix(".testing.json")
    )
    testing_target.parent.mkdir(parents=True, exist_ok=True)
    with testing_target.open("w", encoding="utf8") as handle:
        json.dump(testing_report, handle, indent=2)

    testing_viz_target = (
        _resolve_output_path(testing_visualization_path)
        if testing_visualization_path
        else output_path.with_suffix(".testing.png")
    )
    testing_viz_path = plot_testing_visual(
        testing_report,
        categorical_columns=resolved_schema.categorical_columns,
        continuous_columns=resolved_schema.continuous_columns,
        output_path=testing_viz_target,
    )

    preview = synthetic_records[:5]

    return {
        "analysis": analysis_summary,
        "evaluation": evaluation,
        "quality_report": quality_report,
        "synthetic_path": str(output_path),
        "metrics_path": str(metrics_path),
        "visualization_path": str(viz_path),
        "training_history_path": str(history_path),
        "testing_report_path": str(testing_target),
        "testing_visualization_path": str(testing_viz_path),
        "synthetic_preview": preview,
        "training_round_trip_preview": round_trip_preview,
        "training_history": training_history,
        "testing_report": testing_report,
    }


def load_config(path: Path | str) -> PipelineConfig:
    """Load a pipeline configuration from a JSON file."""

    with open(path, "r", encoding="utf8") as handle:
        raw = json.load(handle)

    schema_payload = raw.get("schema", raw)
    schema = SchemaConfig(
        id_column=schema_payload.get("id_column", DEFAULT_SCHEMA.id_column),
        geo_column=schema_payload.get("geo_column", DEFAULT_SCHEMA.geo_column),
        date_column=schema_payload.get("date_column", DEFAULT_SCHEMA.date_column),
        continuous_columns=tuple(
            schema_payload.get("continuous_columns")
            or schema_payload.get("numeric_features")
            or DEFAULT_SCHEMA.continuous_columns
        ),
        categorical_columns=tuple(
            schema_payload.get("categorical_columns")
            or schema_payload.get("categorical_features")
            or DEFAULT_SCHEMA.categorical_columns
        ),
        drop_columns=tuple(
            schema_payload.get("drop_columns")
            or schema_payload.get("drop_features")
            or DEFAULT_SCHEMA.drop_columns
        ),
    )

    config = PipelineConfig(
        data_path=Path(raw["data_path"]),
        output_path=Path(raw.get("output_path", DEFAULT_OUTPUT_PATH)),
        schema=schema,
        gan=GANTrainingConfig(**raw.get("gan", {})),
        samples_to_generate=int(raw.get("samples_to_generate", DEFAULT_SAMPLES)),
        metrics_path=Path(raw["metrics_path"]) if raw.get("metrics_path") else None,
        visualization_path=Path(raw["visualization_path"]) if raw.get("visualization_path") else None,
        training_history_path=Path(raw["training_history_path"]) if raw.get("training_history_path") else None,
        testing_report_path=Path(raw["testing_report_path"]) if raw.get("testing_report_path") else None,
        testing_visualization_path=Path(raw["testing_visualization_path"]) if raw.get("testing_visualization_path") else None,
    )
    return config


def run_from_config(config: PipelineConfig) -> Dict[str, object]:
    """Execute the pipeline using a :class:`PipelineConfig`."""

    return generate_synthetic_transactions(
        config.data_path,
        output_path=config.output_path,
        schema=config.schema,
        gan_config=config.gan,
        samples_to_generate=config.samples_to_generate,
        metrics_path=config.metrics_path,
        visualization_path=config.visualization_path,
        training_history_path=config.training_history_path,
        testing_report_path=config.testing_report_path,
        testing_visualization_path=config.testing_visualization_path,
    )


__all__ = [
    "DEFAULT_OUTPUT_PATH",
    "DEFAULT_SCHEMA",
    "DEFAULT_SAMPLES",
    "generate_synthetic_transactions",
    "load_config",
    "run_from_config",
]
