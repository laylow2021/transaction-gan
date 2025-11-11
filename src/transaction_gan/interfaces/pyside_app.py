"""PySide6 desktop interface for running the Transaction GAN pipeline."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import List, Sequence

try:  # pragma: no cover - GUI dependency is optional
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import (
        QApplication,
        QComboBox,
        QFileDialog,
        QFormLayout,
        QGroupBox,
        QHBoxLayout,
        QLabel,
        QLineEdit,
        QListWidget,
        QListWidgetItem,
        QMainWindow,
        QMessageBox,
        QPushButton,
        QPlainTextEdit,
        QSpinBox,
        QDoubleSpinBox,
        QVBoxLayout,
        QWidget,
    )
except ModuleNotFoundError as exc:  # pragma: no cover - guidance for runtime users
    raise ModuleNotFoundError(
        "PySide6 is required for the desktop GUI. Install it via `pip install PySide6`."
    ) from exc

try:  # pragma: no cover - import fallback for standalone invocation
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


class TransactionGanWindow(QMainWindow):
    """Main window housing the PySide6 controls."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Transaction GAN Studio (PySide6)")
        self.resize(1200, 800)

        self.data_path: Path | None = None
        self.columns: Sequence[str] = []

        self._build_ui()

    # --- UI construction helpers -------------------------------------------------
    def _build_ui(self) -> None:
        central = QWidget(self)
        layout = QVBoxLayout(central)

        dataset_group = self._build_dataset_section()
        layout.addWidget(dataset_group)

        columns_group = self._build_column_section()
        layout.addWidget(columns_group)

        settings_group = self._build_settings_section()
        layout.addWidget(settings_group)

        self.run_button = QPushButton("Generate synthetic data")
        self.run_button.clicked.connect(self.run_pipeline)
        layout.addWidget(self.run_button)

        result_label = QLabel("Result preview")
        layout.addWidget(result_label)
        self.result_view = QPlainTextEdit()
        self.result_view.setReadOnly(True)
        layout.addWidget(self.result_view, stretch=1)

        central.setLayout(layout)
        self.setCentralWidget(central)

    def _build_dataset_section(self) -> QGroupBox:
        group = QGroupBox("1 · Load data")
        form = QFormLayout()

        self.data_path_edit = QLineEdit(str(Path("data/sample_transactions.csv")))
        browse_button = QPushButton("Browse…")
        browse_button.clicked.connect(self.select_data_file)
        path_row = QHBoxLayout()
        path_row.addWidget(self.data_path_edit, stretch=1)
        path_row.addWidget(browse_button)
        path_container = QWidget()
        path_container.setLayout(path_row)
        form.addRow("CSV path", path_container)

        self.preview_box = QPlainTextEdit()
        self.preview_box.setReadOnly(True)
        self.preview_box.setPlaceholderText("Load a CSV file to preview the first few rows.")
        form.addRow("Preview", self.preview_box)

        load_button = QPushButton("Load columns")
        load_button.clicked.connect(self.load_columns_from_path)
        form.addRow(load_button)

        group.setLayout(form)
        return group

    def _build_column_section(self) -> QGroupBox:
        group = QGroupBox("2 · Column configuration")
        layout = QVBoxLayout()

        form = QFormLayout()
        self.id_combo = QComboBox()
        self.id_combo.addItem("<None>")
        form.addRow("ID column", self.id_combo)

        self.drop_list = QListWidget()
        self.drop_list.setSelectionMode(QListWidget.MultiSelection)
        form.addRow("Drop columns", self.drop_list)

        self.continuous_list = QListWidget()
        self.continuous_list.setSelectionMode(QListWidget.MultiSelection)
        form.addRow("Continuous columns", self.continuous_list)

        self.categorical_list = QListWidget()
        self.categorical_list.setSelectionMode(QListWidget.MultiSelection)
        form.addRow("Categorical columns", self.categorical_list)

        self.hierarchical_edit = QPlainTextEdit()
        self.hierarchical_edit.setPlaceholderText("country,state,city")
        form.addRow("Hierarchical groups\n(one per line)", self.hierarchical_edit)

        layout.addLayout(form)
        group.setLayout(layout)
        return group

    def _build_settings_section(self) -> QGroupBox:
        group = QGroupBox("3 · GAN settings")
        form = QFormLayout()

        self.samples_spin = QSpinBox()
        self.samples_spin.setRange(16, 20000)
        self.samples_spin.setValue(512)
        self.samples_spin.setSingleStep(16)
        form.addRow("Synthetic samples", self.samples_spin)

        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(1, 5000)
        self.epochs_spin.setValue(200)
        form.addRow("Epochs", self.epochs_spin)

        self.noise_spin = QSpinBox()
        self.noise_spin.setRange(4, 512)
        self.noise_spin.setValue(128)
        form.addRow("Noise dimension", self.noise_spin)

        self.hidden_spin = QSpinBox()
        self.hidden_spin.setRange(8, 1024)
        self.hidden_spin.setValue(256)
        form.addRow("Hidden layer width", self.hidden_spin)

        self.learning_rate_spin = QDoubleSpinBox()
        self.learning_rate_spin.setRange(1e-5, 1e-1)
        self.learning_rate_spin.setDecimals(6)
        self.learning_rate_spin.setSingleStep(1e-4)
        self.learning_rate_spin.setValue(2e-4)
        form.addRow("Learning rate", self.learning_rate_spin)

        self.train_fraction_spin = QDoubleSpinBox()
        self.train_fraction_spin.setRange(0.1, 0.95)
        self.train_fraction_spin.setSingleStep(0.05)
        self.train_fraction_spin.setValue(0.8)
        form.addRow("Train fraction", self.train_fraction_spin)

        self.split_seed_spin = QSpinBox()
        self.split_seed_spin.setRange(0, 10_000)
        self.split_seed_spin.setValue(42)
        form.addRow("Split seed", self.split_seed_spin)

        self.output_path_edit = QLineEdit(str(Path("data/synthetic_transactions_gui.csv")))
        form.addRow("Output CSV", self.output_path_edit)

        group.setLayout(form)
        return group

    # --- Dataset loading ---------------------------------------------------------
    def select_data_file(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select transaction CSV",
            str(Path.cwd()),
            "CSV files (*.csv)",
        )
        if file_path:
            self.data_path_edit.setText(file_path)
            self.load_columns_from_path()

    def load_columns_from_path(self) -> None:
        path = Path(self.data_path_edit.text()).expanduser()
        if not path.exists():
            QMessageBox.warning(self, "File not found", f"The path '{path}' does not exist.")
            return

        try:
            columns = _read_columns(path)
        except Exception as exc:  # pragma: no cover - user feedback
            QMessageBox.critical(self, "Failed to read CSV", str(exc))
            return

        if not columns:
            QMessageBox.warning(self, "No columns detected", "The selected CSV has no header row.")
            return

        self.data_path = path
        self.columns = columns
        self._populate_column_widgets(columns)

        preview_rows = _rows_preview(path)
        self.preview_box.setPlainText(json.dumps(preview_rows, indent=2))

    def _populate_column_widgets(self, columns: Sequence[str]) -> None:
        # ID combo
        current_id = self.id_combo.currentText()
        self.id_combo.blockSignals(True)
        self.id_combo.clear()
        self.id_combo.addItem("<None>")
        for column in columns:
            self.id_combo.addItem(column)
        if current_id and current_id in columns:
            self.id_combo.setCurrentText(current_id)
        self.id_combo.blockSignals(False)

        def _fill_list(widget: QListWidget, defaults: Sequence[str] | None = None) -> None:
            widget.clear()
            defaults = defaults or []
            for column in columns:
                item = QListWidgetItem(column)
                if column in defaults:
                    item.setSelected(True)
                widget.addItem(item)

        _fill_list(
            self.continuous_list,
            defaults=[col for col in columns if col in {"amount", "customer_age"}],
        )
        _fill_list(
            self.categorical_list,
            defaults=[
                col
                for col in columns
                if col
                in {
                    "merchant_category",
                    "transaction_type",
                    "merchant_code",
                    "is_fraud",
                    "transaction_date",
                    "transaction_address",
                }
            ],
        )
        _fill_list(self.drop_list, defaults=[self.id_combo.currentText()] if self.id_combo.currentIndex() > 0 else [])

    # --- Helpers -----------------------------------------------------------------
    @staticmethod
    def _selected_items(widget: QListWidget) -> List[str]:
        return [item.text() for item in widget.selectedItems()]

    @staticmethod
    def _parse_hierarchical(text: str) -> List[List[str]]:
        groups: List[List[str]] = []
        for line in text.splitlines():
            tokens = [token.strip() for token in line.split(",") if token.strip()]
            if len(tokens) >= 2:
                groups.append(tokens)
        return groups

    # --- Pipeline execution ------------------------------------------------------
    def run_pipeline(self) -> None:
        if not self.data_path:
            QMessageBox.warning(self, "Load data first", "Select a CSV file and load its columns.")
            return

        continuous = self._selected_items(self.continuous_list)
        categorical = self._selected_items(self.categorical_list)
        drop_columns = self._selected_items(self.drop_list)
        id_column = self.id_combo.currentText()
        if id_column == "<None>":
            id_column = None

        if not continuous:
            QMessageBox.warning(self, "Missing configuration", "Select at least one continuous column.")
            return
        if not categorical:
            QMessageBox.warning(self, "Missing configuration", "Select at least one categorical column.")
            return

        hierarchical_groups = self._parse_hierarchical(self.hierarchical_edit.toPlainText())

        schema = SchemaConfig(
            id_column=id_column,
            continuous_columns=continuous,
            categorical_columns=categorical,
            drop_columns=drop_columns,
            hierarchical_categorical_groups=hierarchical_groups,
        )
        gan_config = GANTrainingConfig(
            noise_dim=int(self.noise_spin.value()),
            hidden_dim=int(self.hidden_spin.value()),
            epochs=int(self.epochs_spin.value()),
            learning_rate=float(self.learning_rate_spin.value()),
            batch_size=int(min(self.samples_spin.value(), 1024)),
        )

        output_path = Path(self.output_path_edit.text()).expanduser()
        output_path.parent.mkdir(parents=True, exist_ok=True)

        self.run_button.setEnabled(False)
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            result = generate_synthetic_transactions(
                data_path=self.data_path,
                output_path=output_path,
                schema=schema,
                gan_config=gan_config,
                samples_to_generate=int(self.samples_spin.value()),
                train_fraction=float(self.train_fraction_spin.value()),
                split_seed=int(self.split_seed_spin.value()),
            )
        except Exception as exc:  # pragma: no cover - user feedback
            QMessageBox.critical(self, "Generation failed", str(exc))
            self.result_view.setPlainText(f"Error: {exc}")
        else:
            summary = {
                "split_summary": result.get("split_summary"),
                "quality_report": result.get("quality_report"),
                "real_vs_synthetic_auc": result.get("real_vs_synthetic_auc"),
            }
            self.result_view.setPlainText(json.dumps(summary, indent=2))
            QMessageBox.information(
                self,
                "Synthetic dataset ready",
                f"Synthetic CSV saved to:\n{result['synthetic_path']}",
            )
        finally:
            QApplication.restoreOverrideCursor()
            self.run_button.setEnabled(True)


def main() -> None:
    app = QApplication(sys.argv)
    window = TransactionGanWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":  # pragma: no cover - manual entry point
    main()
