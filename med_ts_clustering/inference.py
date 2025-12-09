import torch
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path
from .dataset.data import collate_patient_sequences

def export_clusters(model, dataset, feature_cfg, config, output_name="cluster_assignments.csv"):
    device = torch.device(config["training"]["device"])
    batch_size = config["training"]["batch_size"]

    output_dir = Path(config["paths"]["output_dir"]) / "inference"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_csv = output_dir / output_name

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_patient_sequences)
    model = model.to(device)
    model.eval()

    records = []
    embeddings = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Exporting Clusters"):
            x_cont = batch["x_cont"].to(device)
            x_cat = batch["x_cat"].to(device)
            mask = batch["mask"].to(device)
            times = batch["times"]
            patient_ids = batch["patient_ids"]

            h, q = model(x_cont, x_cat, mask, times)
            cluster_ids = q.argmax(dim=-1).cpu()

            mask_np = mask.cpu().numpy()
            B, T = cluster_ids.shape

            for i in range(B):
                pid = patient_ids[i]
                for t in range(T):
                    if not mask_np[i, t]:
                        continue
                    records.append({
                        feature_cfg.patient_id_col: pid,
                        feature_cfg.time_col: times[i][t],
                        "cluster_id": int(cluster_ids[i, t]),
                    })
                    embeddings.append(h[i, t].cpu().numpy())

    df = pd.DataFrame(records)
    df.to_csv(output_csv, index=False)
    print(f"Saved cluster assignments → {output_csv}")

    if len(embeddings) > 0:
        emb_path = output_dir / "embeddings.npy"
        import numpy as np
        np.save(emb_path, np.array(embeddings))
        print(f"Saved embeddings → {emb_path}")

    return df


# Standalone inference 入口
if __name__ == "__main__":
    import yaml
    from dataset.data import FeatureConfig
    from dataset.utils import prepare_dataset
    from models import TimeSeriesClusteringModel

    cfg = yaml.safe_load(open("config.yaml", encoding="utf-8"))

    feature_cfg = FeatureConfig(
        patient_id_col=cfg["feature"]["patient_id_col"],
        time_col=cfg["feature"]["time_col"],
        cont_cols=cfg["feature"]["cont_cols"],
        cat_cols=cfg["feature"]["cat_cols"],
        static_cont_cols=cfg["feature"]["static_cont_cols"],
        static_cat_cols=cfg["feature"]["static_cat_cols"],
    )

    dataset = prepare_dataset(
        cfg["paths"]["static_csv"],
        cfg["paths"]["events_csv"],
        feature_cfg,
        cfg["model"]["max_seq_len"],
    )

    n_cont = len(feature_cfg.cont_cols) + len(feature_cfg.static_cont_cols)
    cat_card = []
    for col in feature_cfg.cat_cols:
        cat_card.append(len(dataset.artifacts.cat_vocab_maps[col]))
    for col in feature_cfg.static_cat_cols:
        cat_card.append(len(dataset.artifacts.static_cat_vocab_maps[col]))

    model = TimeSeriesClusteringModel(
        n_cont_features=n_cont,
        cat_cardinalities=cat_card,
        d_model=cfg["model"]["d_model"],
        n_clusters=cfg["model"]["n_clusters"],
        time_transformer_cfg=cfg["model"]["time_transformer_cfg"],
        ft_kwargs=cfg["model"]["ft_kwargs"],
        enable_reconstruction=False,
        use_missing_embedding=True,
        use_mask_embedding=True,
    )
    model.load_state_dict(torch.load(cfg["paths"]["model_path"]))

    print("======================= 开始聚类结果导出 =======================")
    export_clusters(model, dataset, feature_cfg, cfg)
    print("======================= 聚类结果导出完成 =======================")