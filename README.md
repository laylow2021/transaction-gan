# Transaction GAN

This project provides an end-to-end workflow for analysing, transforming, and generating
synthetic financial transaction data using the SDV PyTorch-based Conditional Tabular GAN (CTGAN).
It includes:

- Exploratory analysis utilities for numeric and categorical transaction features.
- A preprocessing pipeline that standardises numeric fields and one-hot encodes categorical
  attributes.
- A CTGAN implementation (via `sdv.single_table.CTGANSynthesizer`) that respects categorical
  dependencies and hierarchical groupings.
- Evaluation helpers that compare summary statistics between real and synthetic datasets.
- A command line interface and automated tests to validate the pipeline.

## Quick start

1. Ensure you are using Python 3.11 (the project targets 3.11 as its base interpreter). Create a virtual environment (optional) and install the project in editable mode if desired:

   ```bash
   pip install -e .[dev] "sdv[torch]"
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

## PySide6 interface

Launch the desktop app to run the pipeline end-to-end without writing code:

```bash
python -m transaction_gan.interfaces.pyside_app
```

Steps:

1. Install the GUI dependency (already included in `pip install -e .[dev] "sdv[torch]"`; otherwise run `pip install PySide6`).
2. From the project root, activate your virtual environment.
3. Execute the command above; a Qt window will open.
4. Pick a CSV, configure columns/split settings, and click **Generate synthetic data**.

The GUI lets you:

- Select a CSV (or paste a path), preview the first few rows, and configure column roles.
- Define hierarchical categorical groups in a free-form text box.
- Tune CTGAN hyperparameters (samples, epochs, noise dimension, hidden size, learning rate).
- Trigger dataset generation and inspect the resulting quality report directly inside the app.

## Configuration

The pipeline can be controlled through either CLI arguments or a JSON configuration file.
A configuration file supports the following structure:

```json
{
  "data_path": "data/sample_transactions.csv",
  "output_path": "data/synthetic_transactions.csv",
  "numeric_features": ["amount", "customer_age"],
  "categorical_features": ["merchant_category", "transaction_type"],
  "hierarchical_categorical_groups": [
    ["merchant_category", "transaction_type"]
  ],
  "drop_features": ["transaction_id", "is_fraud"],
  "gan": {
    "noise_dim": 8,
    "hidden_dim": 16,
    "epochs": 200,
    "learning_rate": 0.001,
    "batch_size": 256
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
  Use this for any string-like fields, including dates or addresses, if you need to retain
  them in the synthetic output.
- **Hierarchical categorical groups (`hierarchical_categorical_groups`)** – optional ordered
  lists of categorical columns (e.g. country → state → city). Each hierarchy is one-hot
  encoded as a single composite feature so the original combinations are preserved when the
  GAN generates new rows. Synthetic data can only contain combinations observed in the real
  data for those hierarchies.
- **Drop columns (`drop_columns`)** – removed before training so sensitive identifiers
  never enter the GAN, but you can still include them in the final CSV ordering.

When running via the CLI (`src/transaction_gan/cli.py`) or notebook code, pass these schema
settings using `SchemaConfig` or the associated CLI flags (`--id-col`, `--continuous`, `--categorical`,
`--hierarchical`, etc.).

### Train/holdout split & evaluation

After trimming the dataset, the pipeline automatically splits rows into training and holdout
partitions (80/20 by default, configurable via `--train-fraction` and `--split-seed`). CTGAN
is fitted on the training subset only, and every evaluation artifact now includes:

- Train vs synthetic metrics (KS/Wasserstein, histograms, categorical frequencies).
- Holdout vs synthetic metrics to verify the model generalises to unseen rows.
- A simple “real vs synthetic” classifier AUC (values near 0.5 mean the discriminator can’t
  tell them apart).
- Optional train-on-synthetic / test-on-real (TSTR) scores when you provide a target column
  (e.g., `--tstr-target is_fraud`).

Turn on the built-in CTGAN tuner (`--auto-tune`) to automatically evaluate a small grid of
hyperparameters using a validation slice of the training data. The pipeline then retrains on
the full training split with the best-performing configuration and records the tuning history
inside `*.metrics.json`. Provide a custom JSON list via `--tuning-grid path/to/grid.json` if you
only want to try a handful of overrides; otherwise the default search space is used.

Both the CLI output and `*.metrics.json` file contain split summaries plus these scores, and the
PySide6 GUI surfaces the key information in its results pane.

### CTGAN training

The generator now relies on SDV's PyTorch CTGAN implementation. Tune its behaviour through
`GANTrainingConfig` fields (`noise_dim` → embedding dimension, `hidden_dim` → generator/discriminator
layer widths, `epochs`, `learning_rate`, `batch_size`) either directly in code or in the `"gan"`
block of a JSON configuration file. Install the necessary dependencies with:

```bash
pip install sdv[torch]
```

Lower settings speed up experimentation while higher ones usually improve fidelity (at the cost of
extra GPU/CPU time).

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
- **Testing visual summary** – a companion PNG (or text fallback) is produced for quick
  inspection of the testing report (`OUTPUT.testing.png`, configurable via `--testing-plot-path`).
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
