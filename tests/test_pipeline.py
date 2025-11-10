from pathlib import Path

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
        geo_column="transaction_address",
        date_column="transaction_date",
        continuous_columns=["amount", "customer_age"],
        categorical_columns=["merchant_category", "transaction_type", "merchant_code", "is_fraud"],
        drop_columns=["transaction_id"],
    )

    result = generate_synthetic_transactions(
        data_path,
        output_path=output_path,
        schema=schema,
        gan_config=GANTrainingConfig(epochs=5, noise_dim=4, hidden_dim=8, learning_rate=1e-3),
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
