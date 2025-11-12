"""Command line interface for running the transaction GAN pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

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
        "--hierarchical",
        action="append",
        help=(
            "Comma-separated categorical columns that form a hierarchy "
            "(e.g. 'country,state,city'). Provide multiple times for multiple hierarchies."
        ),
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
    parser.add_argument(
        "--testing-plot-path",
        type=Path,
        help="Optional path to store the testing report visual summary.",
    )
    parser.add_argument(
        "--train-fraction",
        type=float,
        default=0.8,
        help="Portion of rows reserved for GAN training (0-1].",
    )
    parser.add_argument(
        "--split-seed",
        type=int,
        default=42,
        help="Random seed used while shuffling before the train/holdout split.",
    )
    parser.add_argument(
        "--auto-tune",
        action="store_true",
        help="Evaluate multiple CTGAN configs using a validation split before final training.",
    )
    parser.add_argument(
        "--validation-fraction",
        type=float,
        default=0.2,
        help="Fraction of the training subset reserved for tuning validation.",
    )
    parser.add_argument(
        "--tstr-target",
        help="Categorical column to use for train-on-synthetic/test-on-real scoring.",
    )
    parser.add_argument(
        "--tuning-grid",
        type=Path,
        help="Optional JSON file containing a list of GAN config overrides for auto-tuning.",
    )
    return parser


def run_cli(args: argparse.Namespace | None = None) -> Dict[str, Any]:
    parser = build_parser()
    parsed = parser.parse_args(args=args)

    if parsed.config:
        config = load_config(parsed.config)
        result = run_from_config(config)
    else:
        def _parse_hierarchical(raw: List[str] | None) -> List[tuple[str, ...]]:
            if raw is None:
                return [tuple(group) for group in DEFAULT_SCHEMA.hierarchical_categorical_groups]
            groups: List[tuple[str, ...]] = []
            for entry in raw:
                columns = [column.strip() for column in entry.split(",") if column.strip()]
                if len(columns) >= 2:
                    groups.append(tuple(columns))
            return groups

        hierarchical_groups = _parse_hierarchical(parsed.hierarchical)
        gan_config = GANTrainingConfig(
            noise_dim=parsed.noise_dim,
            hidden_dim=parsed.hidden_dim,
            epochs=parsed.epochs,
            learning_rate=parsed.learning_rate,
        )
        schema = SchemaConfig(
            id_column=parsed.id_col,
            continuous_columns=parsed.continuous,
            categorical_columns=parsed.categorical,
            drop_columns=parsed.drop,
            hierarchical_categorical_groups=hierarchical_groups,
        )
        tuning_overrides = None
        if parsed.tuning_grid:
            tuning_overrides = json.loads(parsed.tuning_grid.read_text())

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
            testing_visualization_path=parsed.testing_plot_path,
            hierarchical_categorical_groups=hierarchical_groups,
            train_fraction=parsed.train_fraction,
            split_seed=parsed.split_seed,
            auto_tune=parsed.auto_tune,
            validation_fraction=parsed.validation_fraction,
            tstr_target=parsed.tstr_target,
            tuning_overrides=tuning_overrides,
        )

    print(json.dumps(result, indent=2, cls=EnhancedJSONEncoder))
    return result


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    run_cli()
