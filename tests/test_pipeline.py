import csv
import json
from pathlib import Path

import pytest

pytest.importorskip("sdv")

from transaction_gan.config import SchemaConfig
from transaction_gan.gan import GANTrainingConfig
from transaction_gan.pipeline import generate_synthetic_transactions


def test_pipeline_runs_end_to_end(tmp_path):
    data_path = Path("data/sample_transactions.csv")
    output_path = tmp_path / "synthetic.csv"
    metrics_path = tmp_path / "metrics.json"
    viz_path = tmp_path / "viz.png"
    history_path = tmp_path / "loss.png"
    testing_path = tmp_path / "testing.json"
    testing_plot_path = tmp_path / "testing_plot.png"

    schema = SchemaConfig(
        id_column="transaction_id",
        continuous_columns=["amount", "customer_age"],
        categorical_columns=[
            "merchant_category",
            "transaction_type",
            "merchant_code",
            "is_fraud",
            "transaction_date",
            "transaction_address",
        ],
        hierarchical_categorical_groups=[("merchant_category", "transaction_type")],
        drop_columns=["transaction_id"],
    )

    result = generate_synthetic_transactions(
        data_path,
        output_path=output_path,
        schema=schema,
        gan_config=GANTrainingConfig(
            epochs=5,
            noise_dim=4,
            hidden_dim=32,
            learning_rate=1e-3,
            batch_size=64,
        ),
        samples_to_generate=16,
        metrics_path=metrics_path,
        visualization_path=viz_path,
        training_history_path=history_path,
        testing_report_path=testing_path,
        testing_visualization_path=testing_plot_path,
    )

    assert output_path.exists()
    assert metrics_path.exists()
    assert viz_path.exists()
    assert history_path.exists()
    assert testing_path.exists()
    assert testing_plot_path.exists()
    lines = output_path.read_text().strip().splitlines()
    assert len(lines) == 17  # header + 16 records
    assert "analysis" in result
    assert "evaluation" in result
    assert "quality_report" in result
    assert "training_history" in result
    assert "testing_report" in result
    assert "testing_visualization_path" in result
    assert result["synthetic_preview"]
    assert set(result["evaluation"]) >= {"train", "holdout"}
    assert result["split_summary"]["train_size"] > 0
    assert result["real_vs_synthetic_auc"]["train"] <= 1.0
    assert "tstr_scores" in result
    assert "selected_gan_config" in result

    metrics_payload = json.loads(metrics_path.read_text())
    assert "split_summary" in metrics_payload
    testing_payload = json.loads(testing_path.read_text())
    assert "train" in testing_payload
    assert testing_payload["train"]

    def _combos(path: Path) -> set[tuple[str, str]]:
        combos: set[tuple[str, str]] = set()
        with path.open("r", encoding="utf8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                combos.add((row["merchant_category"], row["transaction_type"]))
        return combos

    training_combos = _combos(data_path)
    synthetic_combos = _combos(output_path)
    assert synthetic_combos.issubset(training_combos)
