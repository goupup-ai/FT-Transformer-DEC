import argparse
from pathlib import Path

import torch

from med_ts_clustering.trainer import (
    export_cluster_assignments,
)
from med_ts_clustering.models import TimeSeriesClusteringModel
from med_ts_clustering.trainer import train_dec
from med_ts_clustering.dataset.utils import prepare_dataset
from med_ts_clustering.dataset.data import FeatureConfig


def build_feature_config() -> FeatureConfig:
    """Feature configuration copied from run_dec_training.py for consistent inference."""
    return FeatureConfig(
        # 患者/序列 ID 与时间索引
        patient_id_col="stay_id",
        time_col="hour_index",
        # 动态连续特征
        cont_cols=[
            "heart_rate",
            "resp_rate",
            "temperature",
            "sbp",
            "dbp",
            "map",
            "fio2",
            "ph",
            "pco2",
            "po2",
            "hco3",
            "lactate",
            "sao2",
            "base_excess",
            "tco2",
            "sodium",
            "potassium",
            "chloride",
            "calcium",
            "hemoglobin",
            "hematocrit",
            "norepinephrine",
            "dopamine",
            "epinephrine",
            "vasopressin",
            "phenylephrine",
            "in_volume_ml_hr",
            "crystalloid_in_ml_hr",
            "colloid_in_ml_hr",
            "out_volume_ml_hr",
            "urine_output_ml_hr",
            "pf_ratio",
            "anion_gap",
        ],
        # 动态离散特征（当前示例为空，如有额外分类特征可在此补充）
        cat_cols=[],
        # 静态连续特征
        static_cont_cols=[
            "admission_age",
            "charlson_comorbidity_index",
            "height",
            "weight",
            "sofa",
            "sapsii",
            "oasis",
            "total_mv_hours",
            "crrt_hours",
            "icu_los_hours",
            "icu_first_gcs_total",
        ],
        # 静态离散特征
        static_cat_cols=[
            "gender",
            "ventilated",
            "crrt_used",
            "hospital_expire_flag",
            "mortality_28d_post_discharge",
            "icu_first_gcs_source",
        ],
    )


def load_model_from_checkpoint(
    ckpt_path: Path,
    n_clusters: int,
    d_model: int,
    device: torch.device,
) -> TimeSeriesClusteringModel:
    """Rebuild model and load weights from checkpoint."""
    # NOTE: we need dataset artifacts for cat cardinalities; here we only
    # reconstruct the model architecture and load state dict assuming it
    # matches training-time architecture.
    checkpoint = torch.load(ckpt_path, map_location=device)
    # feature_cfg is saved but we rebuild it from code for robustness

    # The actual model architecture (n_cont_features, cat_cardinalities, etc.)
    # is determined by the dataset; for inference on the same data as training,
    # we can reconstruct dataset and then build the model as in train_dec.
    raise NotImplementedError(
        "This helper is not used directly; inference builds dataset and model "
        "via train_dec/prepare_dataset instead."
    )


def run_inference_for_checkpoint(
    ckpt_path: Path,
    static_csv: str,
    events_csv: str,
    feature_cfg: FeatureConfig,
    *,
    d_model: int = 128,
    n_clusters: int = 4,
    max_seq_len: int = 128,
    batch_size: int = 8,
    device: str | None = None,
    output_dir: str = "../dec_outputs",
) -> None:
    """Load a specific epoch checkpoint and export cluster assignments."""
    from med_ts_clustering.dataset.utils import prepare_dataset
    from med_ts_clustering.models import TimeSeriesClusteringModel
    from med_ts_clustering.dataset.data import collate_patient_sequences
    from torch.utils.data import DataLoader

    device_t = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    # Prepare dataset exactly as in training
    dataset = prepare_dataset(static_csv, events_csv, feature_cfg, max_seq_len)

    # Build model architecture consistent with training
    n_cont_features = len(feature_cfg.cont_cols) + len(feature_cfg.static_cont_cols)
    cat_cardinalities = []
    for col in feature_cfg.cat_cols:
        cat_cardinalities.append(
            int(dataset.artifacts.cat_vocab_maps[col].__len__())
        )
    for col in feature_cfg.static_cat_cols:
        cat_cardinalities.append(
            int(dataset.artifacts.static_cat_vocab_maps[col].__len__())
        )

    model = TimeSeriesClusteringModel(
        n_cont_features=n_cont_features,
        cat_cardinalities=cat_cardinalities,
        d_model=d_model,
        n_clusters=n_clusters,
        time_transformer_cfg={
            "d_model": d_model,
            "max_len": max_seq_len,
        },
        ft_kwargs=None,
    ).to(device_t)

    # Load weights
    ckpt = torch.load(ckpt_path, map_location=device_t)
    model.load_state_dict(ckpt["model_state_dict"])

    # Export cluster assignments
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    epoch_tag = ckpt_path.stem  # e.g. model_epoch_5
    out_csv = output_path / f"cluster_assignments_{epoch_tag}.csv"

    export_cluster_assignments(
        model=model,
        dataset=dataset,
        feature_cfg=feature_cfg,
        max_seq_len=max_seq_len,
        batch_size=batch_size,
        device=str(device_t),
        output_csv=str(out_csv),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run inference with different DEC checkpoints and save cluster assignments."
    )
    parser.add_argument(
        "--static_csv",
        type=str,
        default="./med_ts_clustering/Data/test2.xlsx",
        help="Path to static feature table (same as training).",
    )
    parser.add_argument(
        "--events_csv",
        type=str,
        default="./med_ts_clustering/Data/test1_filled.xlsx",
        help="Path to events time series table (same as training).",
    )
    parser.add_argument(
        "--ckpt_dir",
        type=str,
        default="./dec_outputs",
        help="Directory containing model_epoch_*.pt checkpoints.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        nargs="+",
        required=True,
        help="Epoch numbers to run inference for, e.g. --epochs 5 10 20.",
    )
    parser.add_argument(
        "--d_model",
        type=int,
        default=128,
        help="Embedding dimension used during training.",
    )
    parser.add_argument(
        "--n_clusters",
        type=int,
        default=4,
        help="Number of clusters used during training.",
    )
    parser.add_argument(
        "--max_seq_len",
        type=int,
        default=128,
        help="Max sequence length used in dataset preparation.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for inference dataloader.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help='Device for inference, e.g. "cuda" or "cpu". Defaults to CUDA if available.',
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./dec_outputs",
        help="Directory to save cluster assignment CSVs.",
    )

    args = parser.parse_args()

    feature_cfg = build_feature_config()
    ckpt_dir = Path(args.ckpt_dir)

    for epoch in args.epochs:
        ckpt_path = ckpt_dir / f"model_epoch_{epoch}.pt"
        if not ckpt_path.exists():
            print(f"Checkpoint not found: {ckpt_path}, skip.")
            continue
        print(f"Running inference for checkpoint: {ckpt_path}")
        run_inference_for_checkpoint(
            ckpt_path=ckpt_path,
            static_csv=args.static_csv,
            events_csv=args.events_csv,
            feature_cfg=feature_cfg,
            d_model=args.d_model,
            n_clusters=args.n_clusters,
            max_seq_len=args.max_seq_len,
            batch_size=args.batch_size,
            device=args.device,
            output_dir=args.output_dir,
        )


if __name__ == "__main__":
    main()


