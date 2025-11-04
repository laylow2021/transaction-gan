# Transaction GAN

This project provides an end-to-end workflow for analysing, transforming, and generating
synthetic financial transaction data using a lightweight Generative Adversarial Network (GAN)
implemented entirely with the Python standard library. It includes:

- Exploratory analysis utilities for numeric and categorical transaction features.
- A preprocessing pipeline that standardises numeric fields and one-hot encodes categorical
  attributes.
- A pure Python GAN for modelling transaction patterns and generating new samples without
  external dependencies.
- Evaluation helpers that compare summary statistics between real and synthetic datasets.
- A command line interface and automated tests to validate the pipeline.

## Quick start

1. Create a virtual environment (optional) and install the project in editable mode if desired:

   ```bash
   pip install -e .[dev]
   ```

2. Run the pipeline using the included sample dataset:

   ```bash
   python -m transaction_gan.cli data/sample_transactions.csv --samples 128 --epochs 200
   ```

   The command prints a JSON summary and stores the generated data in
   `data/synthetic_transactions.csv` by default.

3. Execute the automated tests:

   ```bash
   pytest
   ```

## Configuration

The pipeline can be controlled through either CLI arguments or a JSON configuration file.
A configuration file supports the following structure:

```json
{
  "data_path": "data/sample_transactions.csv",
  "output_path": "data/synthetic_transactions.csv",
  "numeric_features": ["amount", "customer_age"],
  "categorical_features": ["merchant_category", "transaction_type"],
  "drop_features": ["transaction_id", "is_fraud"],
  "gan": {
    "noise_dim": 8,
    "hidden_dim": 16,
    "epochs": 200,
    "learning_rate": 0.001
  },
  "samples_to_generate": 512
}
```

Save the file and point the CLI to it with `--config path/to/config.json`.

## Project structure

- `transaction_gan/`: Python package containing the pipeline implementation.
- `data/sample_transactions.csv`: Example dataset used for development and testing.
- `tests/`: Automated tests verifying preprocessing, GAN behaviour, and the full pipeline.
- `pyproject.toml`: Python packaging metadata.

## License

This project is provided for demonstration purposes and does not include a specific license.
