"""Command line interface for running the transaction GAN pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

from .config import SchemaConfig
from .gan import GANTrainingConfig
from .pipeline import (
    DEFAULT_OUTPUT_PATH,
    DEFAULT_SAMPLES,
    DEFAULT_SCHEMA,
    generate_synthetic_transactions,
    load_config,
    run_from_config,
)


class EnhancedJSONEncoder(json.JSONEncoder):
    """Handle dataclasses and custom result objects."""

    def default(self, obj: Any) -> Any:  # pragma: no cover - delegated to json
        if hasattr(obj, "__dict__"):
            return obj.__dict__
        return super().default(obj)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_path", type=Path, help="Path to a CSV file of real transactions.")
    parser.add_argument(
        "--config",
        type=Path,
        help="Optional path to a JSON configuration file. Overrides other CLI arguments.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Where to store the generated synthetic dataset.",
    )
    parser.add_argument(
        "--continuous",
        "--numeric",
        dest="continuous",
        nargs="*",
        default=list(DEFAULT_SCHEMA.continuous_columns),
        help="Continuous columns to analyse and transform.",
    )
    parser.add_argument(
        "--categorical",
        nargs="*",
        default=list(DEFAULT_SCHEMA.categorical_columns),
        help="Categorical columns to one-hot encode.",
    )
    parser.add_argument(
        "--drop",
        nargs="*",
        default=list(DEFAULT_SCHEMA.drop_columns),
        help="Columns to drop prior to modelling (e.g. identifiers).",
    )
    parser.add_argument(
        "--id-col",
        default=DEFAULT_SCHEMA.id_column,
        help="Column that uniquely identifies a record.",
    )
    parser.add_argument(
        "--geo-col",
        default=DEFAULT_SCHEMA.geo_column,
        help="Column containing address or geographic information.",
    )
    parser.add_argument(
        "--date-col",
        default=DEFAULT_SCHEMA.date_column,
        help="Column containing transaction dates.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=GANTrainingConfig().epochs,
        help="Number of GAN training epochs.",
    )
    parser.add_argument(
        "--noise-dim",
        type=int,
        default=GANTrainingConfig().noise_dim,
        help="Dimensionality of the random noise fed into the generator.",
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=GANTrainingConfig().hidden_dim,
        help="Width of the hidden layers in the GAN networks.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=GANTrainingConfig().learning_rate,
        help="Learning rate for gradient descent.",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=DEFAULT_SAMPLES,
        help="How many synthetic samples to produce.",
    )
    parser.add_argument(
        "--metrics-path",
        type=Path,
        help="Optional path to store evaluation/testing metrics JSON.",
    )
    parser.add_argument(
        "--viz-path",
        type=Path,
        help="Optional path to store the statistics comparison plot.",
    )
    parser.add_argument(
        "--history-path",
        type=Path,
        help="Optional path to store the GAN training loss plot.",
    )
    parser.add_argument(
        "--testing-path",
        type=Path,
        help="Optional path to store the independent testing report JSON.",
    )
    return parser


def run_cli(args: argparse.Namespace | None = None) -> Dict[str, Any]:
    parser = build_parser()
    parsed = parser.parse_args(args=args)

    if parsed.config:
        config = load_config(parsed.config)
        result = run_from_config(config)
    else:
        gan_config = GANTrainingConfig(
            noise_dim=parsed.noise_dim,
            hidden_dim=parsed.hidden_dim,
            epochs=parsed.epochs,
            learning_rate=parsed.learning_rate,
        )
        schema = SchemaConfig(
            id_column=parsed.id_col,
            geo_column=parsed.geo_col,
            date_column=parsed.date_col,
            continuous_columns=parsed.continuous,
            categorical_columns=parsed.categorical,
            drop_columns=parsed.drop,
        )
        result = generate_synthetic_transactions(
            parsed.data_path,
            output_path=parsed.output,
            schema=schema,
            gan_config=gan_config,
            samples_to_generate=parsed.samples,
            metrics_path=parsed.metrics_path,
            visualization_path=parsed.viz_path,
            training_history_path=parsed.history_path,
            testing_report_path=parsed.testing_path,
        )

    print(json.dumps(result, indent=2, cls=EnhancedJSONEncoder))
    return result


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    run_cli()
