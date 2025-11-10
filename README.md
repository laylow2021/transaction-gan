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

1. Ensure you are using Python 3.11 (the project targets 3.11 as its base interpreter). Create a virtual environment (optional) and install the project in editable mode if desired:

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

## Streamlit interface

Launch the interactive app to run the pipeline end-to-end without writing code:

```bash
streamlit run src/transaction_gan/interfaces/streamlit_app.py
# or
python -m streamlit run src.transaction_gan.interfaces.streamlit_app
```

The UI lets you:

- Upload a local CSV (or reference an on-disk path) and preview the data.
- Assign each column to the correct semantic type (ID, date, geo, continuous, categorical, drop).
- Configure GAN hyperparameters and the number of synthetic rows to generate.
- Download the synthetic dataset plus testing artefacts, and inspect KS/Wasserstein metrics,
  categorical/continuous comparisons, and real-vs-synthetic plots directly in the browser.

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

### Column-type handling

The preprocessor is schema-driven so you can plug the GAN into arbitrary tabular data.
Each column type is treated as follows:

- **ID columns (`id_column`)** – dropped during training but automatically reintroduced
  for synthetic rows (e.g., `synthetic_1234`) to keep outputs easy to join or trace.
- **Continuous columns (`continuous_columns`)** – converted to floats, normalised via
  z-scores, and later inverse-transformed to their original scale.
- **Categorical columns (`categorical_columns`)** – one-hot encoded with an explicit
  “unknown/empty” bucket. During inverse transform the most probable category is chosen.
- **Date column (`date_column`)** – parsed from common date formats, converted to ordinal
  integers, normalised, and emitted back as ISO strings (`YYYY-MM-DD`).
- **Geographic column (`geo_column`)** – free-form addresses are resolved to latitude/
  longitude pairs using a lightweight lookup (street → city → state → country). Both the
  original text and the derived lat/lon features are preserved on inverse transform.
- **Drop columns (`drop_columns`)** – removed before training so sensitive identifiers
  never enter the GAN, but you can still include them in the final CSV ordering.

When running via the CLI (`src/transaction_gan/cli.py`) or notebook code, pass these schema
settings using `SchemaConfig` or the associated CLI flags (`--id-col`, `--geo-col`, etc.).

### Testing & visualisation outputs

Every pipeline run also produces diagnostics so you can validate the GAN:

- **Metrics JSON** – `quality_report` plus Kolmogorov–Smirnov statistics and Wasserstein
  distances are written to `OUTPUT.metrics.json` (or a custom `--metrics-path`). Each column
  gets pass/fail flags to gate GAN quality objectively.
- **Comparison plot** – by default a PNG stored alongside the CSV (or `--viz-path`) shows
  real vs synthetic feature means. If Matplotlib is unavailable the code writes a CSV
  summary instead, keeping the workflow headless-friendly.
- **Training history plot** – generator/discriminator loss curves are exported (or a CSV
  fallback) so you can visualise convergence; override with `--history-path` if desired.
- **Independent testing report** – the `transaction_gan.testing` module benchmarks
  categorical frequencies, continuous histograms, and grouped histograms (continuous per
  category). The CLI/notebook flow saves this as `OUTPUT.testing.json` unless you pass
  `--testing-path`.
- **Round-trip preview** – the return dictionary includes `synthetic_preview` and
  `training_round_trip_preview` so notebooks/CLI outputs can show how well the reversible
  transforms behave before and after GAN sampling.

## Project structure

- `src/transaction_gan/`: Core Python package containing the pipeline implementation.
- `src/transaction_gan/interfaces/`: UI front-ends such as the Streamlit application.
- `src/transaction_gan/testing.py`: Independent comparison utilities for categorical/continuous diagnostics.
- `data/sample_transactions.csv`: Example dataset used for development and testing.
- `tests/`: Automated tests verifying preprocessing, GAN behaviour, and the full pipeline.
- `pyproject.toml`: Python packaging metadata.

## License

This project is provided for demonstration purposes and does not include a specific license.
