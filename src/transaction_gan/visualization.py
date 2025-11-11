"""Visualisations comparing real vs synthetic statistics."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Sequence

try:  # pragma: no cover - optional dependency
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: E402
except Exception:  # pragma: no cover - executed when matplotlib is unavailable
    plt = None

from .data_loader import TransactionRecord


def _numeric_values(records: List[TransactionRecord], column: str) -> List[float]:
    values: List[float] = []
    for row in records:
        try:
            values.append(float(row[column]))
        except (KeyError, TypeError, ValueError):
            continue
    return values


def plot_numeric_comparison(
    real: List[TransactionRecord],
    synthetic: List[TransactionRecord],
    *,
    numeric_features: Iterable[str],
    output_path: Path | str,
) -> Path:
    """Generate a bar chart to show numeric similarity."""

    columns = list(numeric_features)
    if not columns:
        return Path(output_path)

    real_means = []
    synthetic_means = []
    for column in columns:
        real_values = _numeric_values(real, column)
        synthetic_values = _numeric_values(synthetic, column)
        real_means.append(sum(real_values) / len(real_values) if real_values else 0.0)
        synthetic_means.append(sum(synthetic_values) / len(synthetic_values) if synthetic_values else 0.0)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if plt is None:
        _write_text_summary(output_path, columns, real_means, synthetic_means)
        return output_path

    width = 0.35
    x_positions = list(range(len(columns)))
    fig, ax = plt.subplots(figsize=(max(6, len(columns) * 2.5), 4.5))
    ax.bar([x - width / 2 for x in x_positions], real_means, width=width, label="Real")
    ax.bar([x + width / 2 for x in x_positions], synthetic_means, width=width, label="Synthetic")
    ax.set_xticks(x_positions)
    ax.set_xticklabels(columns, rotation=30, ha="right")
    ax.set_ylabel("Mean value")
    ax.set_title("Real vs Synthetic Feature Means")
    ax.legend()
    ax.grid(axis="y", linestyle=":", alpha=0.4)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def _write_text_summary(path: Path, columns: List[str], real: List[float], synthetic: List[float]) -> None:
    """Fallback text-based summary when matplotlib is unavailable."""

    lines = ["feature,real_mean,synthetic_mean"]
    for name, real_mean, synthetic_mean in zip(columns, real, synthetic):
        lines.append(f"{name},{real_mean:.5f},{synthetic_mean:.5f}")
    path.write_text("\n".join(lines), encoding="utf8")


def plot_training_history(
    history: Dict[str, List[float]] | List[Dict[str, float]],
    *,
    output_path: Path | str,
) -> Path:
    """Plot generator/discriminator losses over epochs."""

    if isinstance(history, list):
        # Fallback for legacy history shape (list of records)
        generator = [record.get("generator_loss", 0.0) for record in history]
        discriminator = [record.get("discriminator_loss", 0.0) for record in history]
        epochs = [int(record.get("epoch", idx + 1)) for idx, record in enumerate(history)]
    else:
        generator = history.get("generator", [])
        discriminator = history.get("discriminator", [])
        epochs = history.get("epochs") or list(range(1, max(len(generator), len(discriminator)) + 1))

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not epochs:
        output_path.write_text("epoch,generator_loss,discriminator_loss", encoding="utf8")
        return output_path

    if plt is None:
        lines = ["epoch,generator_loss,discriminator_loss"]
        for idx in epochs:
            gen_value = generator[idx - 1] if idx - 1 < len(generator) else ""
            disc_value = discriminator[idx - 1] if idx - 1 < len(discriminator) else ""
            lines.append(f"{idx},{gen_value},{disc_value}")
        output_path.write_text("\n".join(lines), encoding="utf8")
        return output_path

    fig, ax = plt.subplots(figsize=(6, 4))
    if generator:
        ax.plot(range(1, len(generator) + 1), generator, label="Generator Loss")
    if discriminator:
        ax.plot(range(1, len(discriminator) + 1), discriminator, label="Discriminator Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("GAN Training Progress")
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def plot_testing_report(
    testing_report: Dict[str, object],
    *,
    categorical_columns: Sequence[str],
    continuous_columns: Sequence[str],
    output_path: Path | str,
) -> Path:
    """Create a simple visual summary from the stored testing report."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    categorical_data: Dict[str, Dict[str, Dict[str, int]]] = testing_report.get("categorical_frequencies", {})  # type: ignore[assignment]
    continuous_data: Dict[str, Dict[str, Dict[str, List[float]]]] = testing_report.get("continuous_histograms", {})  # type: ignore[assignment]

    target_cat = next((column for column in categorical_columns if column in categorical_data), None)
    target_cont = next((column for column in continuous_columns if column in continuous_data), None)

    if plt is None:
        lines = ["testing_plot_summary"]
        lines.append(f"categorical_column={target_cat}")
        lines.append(f"continuous_column={target_cont}")
        output_path.write_text("\n".join(lines), encoding="utf8")
        return output_path

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    if target_cat:
        freq_real = categorical_data[target_cat]["real"]
        freq_synth = categorical_data[target_cat]["synthetic"]
        labels = sorted(set(freq_real) | set(freq_synth))
        real_counts = [freq_real.get(label, 0) for label in labels]
        synth_counts = [freq_synth.get(label, 0) for label in labels]
        positions = range(len(labels))
        axes[0].bar([p - 0.15 for p in positions], real_counts, width=0.3, label="Real")
        axes[0].bar([p + 0.15 for p in positions], synth_counts, width=0.3, label="Synthetic")
        axes[0].set_xticks(list(positions))
        axes[0].set_xticklabels(labels, rotation=30, ha="right")
        axes[0].set_title(f"Categorical frequency: {target_cat}")
        axes[0].legend()
    else:
        axes[0].text(0.5, 0.5, "No categorical data", ha="center", va="center")
        axes[0].set_axis_off()

    if target_cont:
        real_hist = continuous_data[target_cont]["real"]
        synth_hist = continuous_data[target_cont]["synthetic"]
        bins = real_hist["bins"]
        real_counts = real_hist["counts"]
        synth_counts = synth_hist["counts"]
        centers = [(bins[i] + bins[i + 1]) / 2 for i in range(len(bins) - 1)]
        axes[1].plot(centers, real_counts, label="Real")
        axes[1].plot(centers, synth_counts, label="Synthetic")
        axes[1].set_title(f"Continuous histogram: {target_cont}")
        axes[1].legend()
    else:
        axes[1].text(0.5, 0.5, "No continuous data", ha="center", va="center")
        axes[1].set_axis_off()

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


__all__ = ["plot_numeric_comparison", "plot_testing_report", "plot_training_history"]
