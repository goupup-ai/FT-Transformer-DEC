import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path
from .Modules.dec import target_distribution, dec_loss
from .models import TimeSeriesClusteringModel
from typing import List
from .dataset.data import collate_patient_sequences

def initialize_kmeans(
    model: TimeSeriesClusteringModel,
    dataloader: DataLoader,
    device: torch.device,
) -> None:
    """Run encoder once over all data to initialize DEC cluster centers with KMeans."""
    model.eval()
    embeddings: List[torch.Tensor] = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="KMeans init"):
            x_cont = batch["x_cont"].to(device)  # (B, T, Cc)
            x_cat = batch["x_cat"].to(device)  # (B, T, Ck)
            mask = batch["mask"].to(device)  # (B, T)
            times = batch["times"]
            missing_mask_cont = batch.get("missing_cont", torch.zeros_like(x_cont, dtype=torch.bool)).to(device)
            missing_mask_cat = batch.get("missing_cat", torch.zeros_like(x_cat, dtype=torch.bool)).to(device)

            h = model.encode(
                x_cont, x_cat, mask, times,
                missing_mask_cont=missing_mask_cont,
                missing_mask_cat=missing_mask_cat,
            )  # (B, T, D)
            valid_mask = mask  # True for valid
            h_valid = h[valid_mask]  # (N_valid, D)
            embeddings.append(h_valid.cpu())

    all_embeddings = torch.cat(embeddings, dim=0)
    model.clustering.init_from_kmeans(all_embeddings)


def train_dec(model, dataset, config):
    device = torch.device(config["training"]["device"])
    batch_size = config["training"]["batch_size"]
    n_epochs = config["dec"]["n_epochs"]
    lr = config["dec"]["lr"]

    output_dir = Path(config["paths"]["output_dir"]) / "dec_finetune"
    output_dir.mkdir(parents=True, exist_ok=True)
    save_every = config["dec"].get("save_every", 10)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_patient_sequences)

    print("===== Initializing cluster centers using KMeans =====")
    initialize_kmeans(model, loader, device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(1, n_epochs + 1):
        total_loss = 0.0
        for batch in tqdm(loader, desc=f"DEC Epoch {epoch}"):
            x_cont = batch["x_cont"].to(device)
            x_cat = batch["x_cat"].to(device)
            mask = batch["mask"].to(device)
            times = batch["times"]

            h, q = model(x_cont, x_cat, mask, times)
            q_valid = q[mask]

            p = target_distribution(q_valid.detach())
            loss = dec_loss(q_valid, p)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += float(loss.detach().cpu())

        print(f"[DEC Epoch {epoch}] Loss = {total_loss:.4f}")

        if epoch % save_every == 0:
            ckpt = output_dir / f"dec_epoch_{epoch}.pt"
            torch.save({"model_state_dict": model.state_dict()}, ckpt)
            print(f"Saved checkpoint → {ckpt}")

    final_path = output_dir / "dec_final.pt"
    torch.save({"model_state_dict": model.state_dict()}, final_path)
    print(f"Saved final DEC model → {final_path}")

    return model


# Standalone DEC 入口
if __name__ == "__main__":
    import yaml
    from dataset.data import FeatureConfig
    from dataset.utils import prepare_dataset

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
        enable_reconstruction=True,
        use_missing_embedding=True,
        use_mask_embedding=True,
    )
    model.load_state_dict(torch.load(cfg["paths"]["model_path"]))

    print("======================= 开始 DEC 微调 =======================")
    train_dec(model, dataset, cfg)
    print("======================= DEC 微调完成 =======================")
