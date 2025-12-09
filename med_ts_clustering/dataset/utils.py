from .data import FeatureConfig, MedicalTimeSeriesDataset, collate_patient_sequences
import pandas as pd
from pathlib import Path


def _read_table_auto(path: str) -> pd.DataFrame:
    """Automatically read CSV / Excel based on file extension."""
    ext = Path(path).suffix.lower()
    if ext in [".csv", ".txt"]:
        return pd.read_csv(path)
    if ext in [".xls", ".xlsx"]:
        return pd.read_excel(path)
    # fallback: try csv first, then excel
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.read_excel(path)


def prepare_dataset(
    static_csv_path: str,
    events_csv_path: str,
    feature_cfg: FeatureConfig,
    max_seq_len: int,
):
    print("📂 Loading raw tables...")

    static_df = _read_table_auto(static_csv_path)
    events_df = _read_table_auto(events_csv_path)

    print(f"   - Static rows: {len(static_df)}")
    print(f"   - Events rows: {len(events_df)}")
    print("🔧 Building MedicalTimeSeriesDataset...")

    dataset = MedicalTimeSeriesDataset(
        static_df=static_df,
        events_df=events_df,
        feature_cfg=feature_cfg,
        max_seq_len=max_seq_len,
        artifacts=None,
        fit_preprocessors=True,
    )

    print("✅ Dataset ready.")
    print(f"   - Number of patients: {len(dataset)}")
    print(f"   - Max sequence length: {max_seq_len}")

    return dataset
