"""Streamlit interface for running the Transaction GAN pipeline."""

from __future__ import annotations

import csv
import json
import sys
import tempfile
from pathlib import Path
from typing import List, Sequence

import streamlit as st

try:  # pragma: no cover - supports direct `streamlit run` invocation
    from ..config import SchemaConfig
    from ..gan import GANTrainingConfig
    from ..pipeline import generate_synthetic_transactions
except ImportError:  # pragma: no cover
    PACKAGE_ROOT = Path(__file__).resolve().parent
    PROJECT_ROOT = PACKAGE_ROOT.parent.parent
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.append(str(PROJECT_ROOT))
    from transaction_gan.config import SchemaConfig
    from transaction_gan.gan import GANTrainingConfig
    from transaction_gan.pipeline import generate_synthetic_transactions


def _read_columns(path: Path) -> List[str]:
    with path.open("r", encoding="utf8") as handle:
        reader = csv.reader(handle)
        header = next(reader, [])
    return header


def _rows_preview(path: Path, limit: int = 5) -> List[dict]:
    preview: List[dict] = []
    with path.open("r", encoding="utf8") as handle:
        reader = csv.DictReader(handle)
        for idx, row in enumerate(reader):
            if idx >= limit:
                break
            preview.append(row)
    return preview


def _option_list(columns: Sequence[str], label: str) -> str | None:
    choices = ["<None>"] + list(columns)
    selection = st.selectbox(label, choices, index=0)
    return None if selection == "<None>" else selection


def main() -> None:
    st.set_page_config(page_title="Transaction GAN Studio", layout="wide")
    st.title("Transaction GAN Studio")
    st.write(
        "Upload a CSV, pick the column roles for the GAN, and generate synthetic data with "
        "built-in testing metrics and visual comparisons."
    )

    with st.sidebar:
        st.header("1 · Load Data")
        data_source = st.radio("Data source", ["Upload CSV", "Use local path"])
        data_path: Path | None = None
        uploaded_file = None
        if data_source == "Upload CSV":
            uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])
            if uploaded_file:
                tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
                tmp.write(uploaded_file.getbuffer())
                tmp.flush()
                data_path = Path(tmp.name)
        else:
            raw_path = st.text_input("Local CSV path", value="data/sample_transactions.csv")
            if raw_path:
                candidate = Path(raw_path).expanduser()
                if candidate.exists():
                    data_path = candidate
                else:
                    st.warning(f"File '{raw_path}' was not found.")

        st.header("2 · GAN Settings")
        samples = st.number_input("Synthetic samples", min_value=16, max_value=20000, value=512, step=16)
        epochs = st.number_input("Epochs", min_value=1, max_value=2000, value=200, step=10)
        noise_dim = st.number_input("Noise dimension", min_value=2, max_value=128, value=8, step=1)
        hidden_dim = st.number_input("Hidden layer width", min_value=4, max_value=256, value=16, step=4)
        learning_rate = st.number_input(
            "Learning rate", min_value=1e-5, max_value=1e-1, value=1e-3, step=1e-4, format="%.5f"
        )

        output_filename = st.text_input("Output filename", value="synthetic_transactions.csv")

    if not data_path:
        st.info("Upload a CSV or provide a valid local path to get started.")
        return

    columns = _read_columns(data_path)
    if not columns:
        st.error("The selected file appears to be empty or missing a header row.")
        return

    st.subheader("Preview real data")
    st.caption(f"Loaded **{data_path}**")
    st.dataframe(_rows_preview(data_path))

    st.header("3 · Column configuration")
    col1, col2, col3 = st.columns(3)
    with col1:
        id_column = _option_list(columns, "ID column")
        date_column = _option_list(columns, "Date column")
    with col2:
        geo_column = _option_list(columns, "Address / Geo column")
        drop_columns = st.multiselect(
            "Columns to drop before training",
            options=columns,
            default=[col for col in (id_column,) if col],
        )
    with col3:
        continuous_columns = st.multiselect(
            "Continuous columns",
            options=columns,
            default=[col for col in columns if col in {"amount", "customer_age"}],
        )
        categorical_columns = st.multiselect(
            "Categorical columns",
            options=columns,
            default=[col for col in columns if col in {"merchant_category", "transaction_type", "merchant_code", "is_fraud"}],
        )

    ready = st.button("Generate synthetic data", type="primary")

    if not ready:
        return

    if not continuous_columns or not categorical_columns:
        st.error("Select at least one continuous and one categorical column before running the GAN.")
        return

    output_dir = Path(tempfile.mkdtemp(prefix="txn-gan-"))
    output_path = output_dir / output_filename

    metrics_path = output_dir / "metrics.json"
    viz_path = output_dir / "comparison.png"
    history_path = output_dir / "loss.png"
    testing_path = output_dir / "testing.json"
    testing_plot_path = output_dir / "testing_plot.png"

    schema = SchemaConfig(
        id_column=id_column,
        geo_column=geo_column,
        date_column=date_column,
        continuous_columns=continuous_columns,
        categorical_columns=categorical_columns,
        drop_columns=drop_columns,
    )
    gan_config = GANTrainingConfig(
        noise_dim=int(noise_dim),
        hidden_dim=int(hidden_dim),
        epochs=int(epochs),
        learning_rate=float(learning_rate),
    )

    with st.spinner("Training GAN and generating synthetic samples..."):
        result = generate_synthetic_transactions(
            data_path=data_path,
            output_path=output_path,
            schema=schema,
            gan_config=gan_config,
            samples_to_generate=int(samples),
            metrics_path=metrics_path,
            visualization_path=viz_path,
            training_history_path=history_path,
            testing_report_path=testing_path,
            testing_visualization_path=testing_plot_path,
        )

    st.success("Synthetic dataset generated!")

    st.subheader("Synthetic preview")
    st.dataframe(result["synthetic_preview"])

    def _download_button(label: str, path: Path, mime: str, file_label: str) -> None:
        if not path.exists():
            return
        st.download_button(
            label=label,
            data=path.read_bytes(),
            file_name=file_label,
            mime=mime,
        )

    st.write("Download artefacts:")
    download_col1, download_col2, download_col3 = st.columns(3)
    with download_col1:
        _download_button("Synthetic CSV", output_path, "text/csv", output_filename)
    with download_col2:
        _download_button("Metrics JSON", metrics_path, "application/json", "metrics.json")
    with download_col3:
        _download_button("Testing report", testing_path, "application/json", "testing.json")
        _download_button("Testing plot", testing_plot_path, "image/png", "testing_plot.png")

    st.header("Testing statistics")
    st.subheader("Quality report (KS & Wasserstein)")
    st.json(result["quality_report"])

    st.subheader("Independent testing report")
    st.json(result["testing_report"])

    st.header("Visual comparisons")
    col_plot1, col_plot2, col_plot3 = st.columns(3)
    with col_plot1:
        st.caption("Real vs Synthetic feature means")
        if Path(result["visualization_path"]).suffix.lower() in {".png", ".jpg", ".jpeg"}:
            st.image(result["visualization_path"])
        else:
            st.download_button(
                "Download comparison summary",
                Path(result["visualization_path"]).read_bytes(),
                file_name="comparison.csv",
                mime="text/csv",
            )
    with col_plot2:
        st.caption("GAN training history")
        if Path(result["training_history_path"]).suffix.lower() in {".png", ".jpg", ".jpeg"}:
            st.image(result["training_history_path"])
        else:
            st.download_button(
                "Download training history",
                Path(result["training_history_path"]).read_bytes(),
                file_name="training_history.csv",
                mime="text/csv",
            )


if __name__ == "__main__":  # pragma: no cover
    main()
    with col_plot3:
        st.caption("Testing report summary")
        if Path(result["testing_visualization_path"]).suffix.lower() in {".png", ".jpg", ".jpeg"}:
            st.image(result["testing_visualization_path"])
        else:
            st.download_button(
                "Download testing summary",
                Path(result["testing_visualization_path"]).read_bytes(),
                file_name="testing_visualization.txt",
                mime="text/plain",
            )
