"""End-to-end pipeline for analysing and generating transaction data."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

from .analysis import describe_transactions
from .config import PipelineConfig, SchemaConfig
from .data_loader import load_transactions
from .data_split import split_records
from .evaluation import build_quality_report, compare_statistics
from .gan import GANTrainingConfig, SyntheticDataGenerator
from .metrics import real_vs_synthetic_auc, tstr_scores
from .preprocessing import TransactionPreprocessor
from .testing import generate_testing_report
from .tuning import DEFAULT_TUNING_GRID, auto_tune_gan
from .visualization import (
    plot_numeric_comparison,
    plot_testing_report as plot_testing_visual,
    plot_training_history,
)


DEFAULT_SCHEMA = SchemaConfig(
    id_column="transaction_id",
    continuous_columns=("amount", "customer_age"),
    categorical_columns=(
        "merchant_category",
        "transaction_type",
        "transaction_date",
        "merchant_code",
        "is_fraud",
        "transaction_address",
    ),
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
    hierarchical_categorical_groups: Iterable[Iterable[str]] | None,
) -> SchemaConfig:
    if schema:
        return schema

    continuous = tuple(continuous_columns or DEFAULT_SCHEMA.continuous_columns)
    categorical_list = list(categorical_columns or DEFAULT_SCHEMA.categorical_columns)
    drops = tuple(drop_columns or DEFAULT_SCHEMA.drop_columns)
    hierarchical = tuple(
        tuple(group)
        for group in (
            hierarchical_categorical_groups or DEFAULT_SCHEMA.hierarchical_categorical_groups
        )
    )

    for group in hierarchical:
        for column in group:
            if column not in categorical_list:
                categorical_list.append(column)
    categorical = tuple(categorical_list)

    return SchemaConfig(
        id_column=id_column if id_column is not None else DEFAULT_SCHEMA.id_column,
        continuous_columns=continuous,
        categorical_columns=categorical,
        hierarchical_categorical_groups=hierarchical,
        drop_columns=drops,
    )


def _prepare_preprocessor(schema: SchemaConfig) -> TransactionPreprocessor:
    return TransactionPreprocessor(
        continuous_features=list(schema.continuous_columns),
        categorical_features=list(schema.categorical_columns),
        id_feature=schema.id_column,
        drop_features=list(schema.drop_columns),
        hierarchical_categorical_groups=list(schema.hierarchical_categorical_groups),
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
    hierarchical_categorical_groups: Iterable[Iterable[str]] | None = None,
    gan_config: GANTrainingConfig | None = None,
    samples_to_generate: int = DEFAULT_SAMPLES,
    metrics_path: Path | str | None = None,
    visualization_path: Path | str | None = None,
    training_history_path: Path | str | None = None,
    testing_report_path: Path | str | None = None,
    testing_visualization_path: Path | str | None = None,
    train_fraction: float = 0.8,
    split_seed: int = 42,
    auto_tune: bool = False,
    validation_fraction: float = 0.2,
    tuning_overrides: Sequence[Dict[str, object]] | None = None,
    tstr_target: str | None = None,
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
        hierarchical_categorical_groups=hierarchical_categorical_groups,
    )

    analysis_summary = describe_transactions(
        records,
        numeric_features=resolved_schema.continuous_columns,
        categorical_features=resolved_schema.categorical_columns,
    )

    trimmed_rows = _trim_records(records, resolved_schema)
    train_rows, holdout_rows = split_records(
        trimmed_rows,
        train_fraction=train_fraction,
        seed=split_seed,
    )
    if not train_rows:
        raise ValueError("No rows were available for training after preprocessing.")

    validation_rows: List[Dict[str, str]] = []
    core_train_rows = list(train_rows)
    if 0.0 < validation_fraction < 1.0 and auto_tune and len(train_rows) > 5:
        core_train_rows, validation_rows = split_records(
            train_rows,
            train_fraction=1.0 - validation_fraction,
            seed=split_seed + 1,
        )
        if not core_train_rows:
            core_train_rows = list(train_rows)

    preprocessor = _prepare_preprocessor(resolved_schema)
    preprocessor.fit(train_rows)
    processed_train_full = preprocessor.transform(train_rows)
    processed_holdout = preprocessor.transform(holdout_rows) if holdout_rows else []
    processed_core = preprocessor.transform(core_train_rows)
    processed_validation = (
        preprocessor.transform(validation_rows) if validation_rows else []
    )

    categorical_slices = preprocessor.categorical_input_slices()
    selected_config = gan_config
    tuning_history: List[Dict[str, object]] = []
    if auto_tune and processed_validation and validation_rows:
        selected_config, tuning_history = auto_tune_gan(
            processed_train=processed_core,
            processed_validation=processed_validation,
            validation_records=validation_rows,
            base_config=gan_config,
            preprocessor=preprocessor,
            categorical_slices=categorical_slices,
            numeric_features=resolved_schema.continuous_columns,
            tuning_grid=tuning_overrides,
        )

    gan = SyntheticDataGenerator(
        len(processed_train_full[0]),
        selected_config,
        categorical_slices=categorical_slices,
    )
    gan.train(processed_train_full)
    training_history = gan.get_training_history()

    synthetic_matrix = gan.generate(samples_to_generate)
    synthetic_records = preprocessor.inverse_transform(synthetic_matrix)
    round_trip_preview = preprocessor.inverse_transform(processed_train_full[:5])

    evaluation_train = compare_statistics(
        train_rows,
        synthetic_records,
        numeric_features=resolved_schema.continuous_columns,
    )
    evaluation_validation = (
        compare_statistics(
            validation_rows,
            synthetic_records,
            numeric_features=resolved_schema.continuous_columns,
        )
        if validation_rows
        else None
    )
    evaluation_holdout = (
        compare_statistics(
            holdout_rows,
            synthetic_records,
            numeric_features=resolved_schema.continuous_columns,
        )
        if holdout_rows
        else None
    )
    evaluation = {
        "train": evaluation_train,
        "validation": evaluation_validation,
        "holdout": evaluation_holdout,
    }
    quality_report = {
        "train": build_quality_report(evaluation_train),
        "validation": build_quality_report(evaluation_validation)
        if evaluation_validation
        else None,
        "holdout": build_quality_report(evaluation_holdout) if evaluation_holdout else None,
    }

    testing_report_train = generate_testing_report(
        train_rows,
        synthetic_records,
        schema=resolved_schema,
    )
    testing_report_validation = (
        generate_testing_report(
            validation_rows,
            synthetic_records,
            schema=resolved_schema,
        )
        if validation_rows
        else None
    )
    testing_report_holdout = (
        generate_testing_report(
            holdout_rows,
            synthetic_records,
            schema=resolved_schema,
        )
        if holdout_rows
        else None
    )
    testing_reports = {
        "train": testing_report_train,
        "validation": testing_report_validation,
        "holdout": testing_report_holdout,
    }

    auc_train = real_vs_synthetic_auc(processed_train_full, synthetic_matrix)
    auc_holdout = (
        real_vs_synthetic_auc(processed_holdout, synthetic_matrix) if processed_holdout else None
    )
    auc_scores = {"train": auc_train, "holdout": auc_holdout}

    tstr = None
    if tstr_target:
        tstr = tstr_scores(
            real_train=train_rows,
            synthetic_records=synthetic_records,
            holdout_records=holdout_rows or [],
            target_column=tstr_target,
        )

    split_summary = {
        "train_fraction": train_fraction,
        "train_size": len(train_rows),
        "holdout_size": len(holdout_rows),
        "split_seed": split_seed,
    }

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
                "evaluation": {
                    "train": evaluation_train.to_dict(),
                    "validation": evaluation_validation.to_dict() if evaluation_validation else None,
                    "holdout": evaluation_holdout.to_dict() if evaluation_holdout else None,
                },
                "real_vs_synthetic_auc": auc_scores,
                "split_summary": split_summary,
                "tstr_scores": tstr,
                "selected_gan_config": asdict(selected_config),
                "tuning_history": tuning_history,
            },
            handle,
            indent=2,
        )

    viz_target = _resolve_output_path(visualization_path) if visualization_path else output_path.with_suffix(".png")
    viz_path = plot_numeric_comparison(
        train_rows,
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
        json.dump(testing_reports, handle, indent=2)

    testing_viz_target = (
        _resolve_output_path(testing_visualization_path)
        if testing_visualization_path
        else output_path.with_suffix(".testing.png")
    )
    testing_viz_path = plot_testing_visual(
        testing_report_train,
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
        "testing_report": testing_reports,
        "real_vs_synthetic_auc": auc_scores,
        "split_summary": split_summary,
        "tstr_scores": tstr,
        "selected_gan_config": asdict(selected_config),
        "tuning_history": tuning_history,
    }


def load_config(path: Path | str) -> PipelineConfig:
    """Load a pipeline configuration from a JSON file."""

    with open(path, "r", encoding="utf8") as handle:
        raw = json.load(handle)

    schema_payload = raw.get("schema", raw)
    legacy_geo = schema_payload.get("geo_column")
    legacy_date = schema_payload.get("date_column")
    categorical_candidates = (
        schema_payload.get("categorical_columns")
        or schema_payload.get("categorical_features")
        or DEFAULT_SCHEMA.categorical_columns
    )
    categorical = list(categorical_candidates)
    for column in (legacy_date, legacy_geo):
        if column and column not in categorical:
            categorical.append(column)

    hierarchical_groups = tuple(
        tuple(group)
        for group in schema_payload.get(
            "hierarchical_categorical_groups", DEFAULT_SCHEMA.hierarchical_categorical_groups
        )
    )
    for group in hierarchical_groups:
        for column in group:
            if column not in categorical:
                categorical.append(column)

    schema = SchemaConfig(
        id_column=schema_payload.get("id_column", DEFAULT_SCHEMA.id_column),
        continuous_columns=tuple(
            schema_payload.get("continuous_columns")
            or schema_payload.get("numeric_features")
            or DEFAULT_SCHEMA.continuous_columns
        ),
        categorical_columns=tuple(categorical),
        hierarchical_categorical_groups=hierarchical_groups,
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
        train_fraction=float(raw.get("train_fraction", 0.8)),
        split_seed=int(raw.get("split_seed", 42)),
        auto_tune=bool(raw.get("auto_tune", False)),
        validation_fraction=float(raw.get("validation_fraction", 0.2)),
        tuning_overrides=raw.get("tuning_overrides"),
        tstr_target=raw.get("tstr_target"),
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
        train_fraction=config.train_fraction,
        split_seed=config.split_seed,
        auto_tune=config.auto_tune,
        validation_fraction=config.validation_fraction,
        tuning_overrides=config.tuning_overrides,
        tstr_target=config.tstr_target,
    )


__all__ = [
    "DEFAULT_OUTPUT_PATH",
    "DEFAULT_SCHEMA",
    "DEFAULT_SAMPLES",
    "generate_synthetic_transactions",
    "load_config",
    "run_from_config",
]
