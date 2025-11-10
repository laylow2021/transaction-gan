from transaction_gan.config import SchemaConfig
from transaction_gan.testing import generate_testing_report


def test_generate_testing_report_handles_basic_inputs():
    real = [
        {"amount": "10", "category": "A"},
        {"amount": "20", "category": "B"},
        {"amount": "30", "category": "A"},
    ]
    synthetic = [
        {"amount": "12", "category": "A"},
        {"amount": "25", "category": "B"},
        {"amount": "35", "category": "C"},
    ]
    schema = SchemaConfig(
        continuous_columns=["amount"],
        categorical_columns=["category"],
    )

    report = generate_testing_report(real, synthetic, schema=schema, bins=5, top_k=2)

    assert "categorical_frequencies" in report
    assert "continuous_histograms" in report
    assert "grouped_histograms" in report
    freq = report["categorical_frequencies"]["category"]
    assert freq["real"]["A"] == 2
    assert freq["synthetic"]["B"] == 1
