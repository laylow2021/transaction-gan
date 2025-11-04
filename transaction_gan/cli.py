"""Command line interface for running the transaction GAN pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

from .gan import GANTrainingConfig
from .pipeline import (
    DEFAULT_CATEGORICAL_FEATURES,
    DEFAULT_DROP_FEATURES,
    DEFAULT_NUMERIC_FEATURES,
    DEFAULT_OUTPUT_PATH,
    DEFAULT_SAMPLES,
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
        "--numeric",
        nargs="*",
        default=DEFAULT_NUMERIC_FEATURES,
        help="Numeric columns to analyse and transform.",
    )
    parser.add_argument(
        "--categorical",
        nargs="*",
        default=DEFAULT_CATEGORICAL_FEATURES,
        help="Categorical columns to one-hot encode.",
    )
    parser.add_argument(
        "--drop",
        nargs="*",
        default=DEFAULT_DROP_FEATURES,
        help="Columns to drop prior to modelling (e.g. identifiers).",
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
        result = generate_synthetic_transactions(
            parsed.data_path,
            output_path=parsed.output,
            numeric_features=parsed.numeric,
            categorical_features=parsed.categorical,
            drop_features=parsed.drop,
            gan_config=gan_config,
            samples_to_generate=parsed.samples,
        )

    print(json.dumps(result, indent=2, cls=EnhancedJSONEncoder))
    return result


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    run_cli()
