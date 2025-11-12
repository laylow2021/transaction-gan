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


def _plot_categorical_axis(ax, name: str, freq_real: Dict[str, int], freq_synth: Dict[str, int]) -> None:
    labels = sorted(set(freq_real) | set(freq_synth))
    positions = range(len(labels))
    real_counts = [freq_real.get(label, 0) for label in labels]
    synth_counts = [freq_synth.get(label, 0) for label in labels]
    max_real = max(real_counts) if real_counts else 0
    max_synth = max(synth_counts) if synth_counts else 0
    min_non_zero = min([value for value in real_counts + synth_counts if value > 0], default=1)
    ratio = max(max_real, max_synth) / min_non_zero if min_non_zero else 1

    width = 0.4
    if ratio <= 3:
        ax.bar([p - width / 2 for p in positions], real_counts, width=width, label="Real")
        ax.bar([p + width / 2 for p in positions], synth_counts, width=width, label="Synthetic")
        ax.set_ylabel("Count")
        ax.legend()
    else:
        primary = ax
        secondary = primary.twinx()
        primary.bar(
            [p - width / 2 for p in positions],
            real_counts,
            width=width,
            label="Real",
            color="#1f77b4",
        )
        secondary.bar(
            [p + width / 2 for p in positions],
            synth_counts,
            width=width,
            label="Synthetic",
            color="#ff7f0e",
            alpha=0.7,
        )
        primary.set_ylabel("Real count")
        secondary.set_ylabel("Synthetic count")
        handles, labels_real = primary.get_legend_handles_labels()
        handles2, labels_synth = secondary.get_legend_handles_labels()
        primary.legend(handles + handles2, labels_real + labels_synth)
    ax.set_xticks(list(positions))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_title(f"Categorical frequency: {name}")


def _plot_continuous_axis(ax, name: str, real_hist: Dict[str, List[float]], synth_hist: Dict[str, List[float]]) -> None:
    bins = real_hist["bins"]
    real_counts = real_hist["counts"]
    synth_counts = synth_hist["counts"]
    if not bins:
        bins = list(range(len(real_counts) + 1))
    centers = [(bins[i] + bins[i + 1]) / 2 for i in range(len(bins) - 1)]
    real_total = sum(real_counts) or 1
    synth_total = sum(synth_counts) or 1
    real_pct = [(value / real_total) * 100 for value in real_counts]
    synth_pct = [(value / synth_total) * 100 for value in synth_counts]
    ax.plot(centers, real_pct, label="Real %", linewidth=2)
    ax.plot(centers, synth_pct, label="Synthetic %", linewidth=2, linestyle="--")
    ax.set_title(f"Continuous histogram (%): {name}")
    ax.set_ylabel("Percentage of samples")
    ax.set_xlabel(name)
    ax.legend()


def _plot_grouped_axis(
    ax,
    cat_name: str,
    cont_name: str,
    cont_map: Dict[str, Dict[str, Dict[str, List[float]]]],
) -> None:
    categories = list(cont_map.keys())[:4]
    if not categories:
        ax.text(0.5, 0.5, "No grouped data", ha="center", va="center")
        ax.set_axis_off()
        return
    for category in categories:
        real_hist = cont_map[category]["real"]
        synth_hist = cont_map[category]["synthetic"]
        bins = real_hist["bins"]
        if not bins:
            continue
        centers = [(bins[i] + bins[i + 1]) / 2 for i in range(len(bins) - 1)]
        real_total = sum(real_hist["counts"]) or 1
        synth_total = sum(synth_hist["counts"]) or 1
        real_pct = [(value / real_total) * 100 for value in real_hist["counts"]]
        synth_pct = [(value / synth_total) * 100 for value in synth_hist["counts"]]
        ax.plot(centers, real_pct, label=f"{category} · Real", linestyle="-")
        ax.plot(centers, synth_pct, label=f"{category} · Synth", linestyle="--")
    ax.set_title(f"{cont_name} by {cat_name} (top {len(categories)})")
    ax.set_ylabel("Percentage of samples")
    ax.set_xlabel(cont_name)
    ax.legend(fontsize=8)


def plot_testing_report(
    testing_report: Dict[str, object],
    *,
    categorical_columns: Sequence[str],
    continuous_columns: Sequence[str],
    output_path: Path | str,
) -> Path:
    """Create a visual summary covering every available testing metric."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    categorical_data: Dict[str, Dict[str, Dict[str, int]]] = testing_report.get("categorical_frequencies", {})  # type: ignore[assignment]
    continuous_data: Dict[str, Dict[str, Dict[str, List[float]]]] = testing_report.get("continuous_histograms", {})  # type: ignore[assignment]
    grouped_data: Dict[str, Dict[str, Dict[str, Dict[str, List[float]]]]] = testing_report.get("grouped_histograms", {})  # type: ignore[assignment]

    available_cats = [column for column in categorical_columns if column in categorical_data]
    available_cont = [column for column in continuous_columns if column in continuous_data]
    available_pairs: List[tuple[str, str]] = []
    for categorical in categorical_columns:
        cont_map = grouped_data.get(categorical, {})
        for cont in continuous_columns:
            if cont in cont_map:
                available_pairs.append((categorical, cont))

    if plt is None:
        lines = ["testing_plot_summary"]
        for column in available_cats:
            lines.append(f"categorical_column={column}")
        for column in available_cont:
            lines.append(f"continuous_column={column}")
        for cat, cont in available_pairs:
            lines.append(f"grouped_pair={cat}:{cont}")
        output_path.write_text("\n".join(lines) or "No data available", encoding="utf8")
        return output_path

    panels = len(available_cats) + len(available_cont) + len(available_pairs)
    if panels == 0:
        panels = 1
    fig, axes = plt.subplots(panels, 1, figsize=(12, max(4, 3 * panels)))
    if panels == 1:
        axes = [axes]  # type: ignore[assignment]
    else:
        axes = list(axes)  # type: ignore[assignment]

    axis_index = 0
    for column in available_cats:
        _plot_categorical_axis(
            axes[axis_index],
            column,
            categorical_data[column]["real"],
            categorical_data[column]["synthetic"],
        )
        axis_index += 1

    for column in available_cont:
        _plot_continuous_axis(
            axes[axis_index],
            column,
            continuous_data[column]["real"],
            continuous_data[column]["synthetic"],
        )
        axis_index += 1

    for cat_name, cont_name in available_pairs:
        _plot_grouped_axis(
            axes[axis_index],
            cat_name,
            cont_name,
            grouped_data[cat_name][cont_name],
        )
        axis_index += 1

    if panels == 1 and axis_index == 0:
        axes[0].text(0.5, 0.5, "No testing data available", ha="center", va="center")
        axes[0].set_axis_off()

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


__all__ = ["plot_numeric_comparison", "plot_testing_report", "plot_training_history"]
