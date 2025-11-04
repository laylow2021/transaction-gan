from pathlib import Path

from transaction_gan.pipeline import generate_synthetic_transactions


def test_pipeline_runs_end_to_end(tmp_path):
    data_path = Path("data/sample_transactions.csv")
    output_path = tmp_path / "synthetic.csv"

    result = generate_synthetic_transactions(
        data_path,
        output_path=output_path,
        samples_to_generate=16,
        drop_features=["transaction_id", "is_fraud"],
    )

    assert output_path.exists()
    lines = output_path.read_text().strip().splitlines()
    assert len(lines) == 17  # header + 16 records
    assert "analysis" in result
    assert "evaluation" in result
    assert result["synthetic_preview"]
