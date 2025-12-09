import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path
from .Modules.mrm import apply_mrm_mask, compute_mrm_loss
from .dataset.data import collate_patient_sequences
from .scripts.plot_loss import save_loss_curve

def train_mrm_pretrain(model, dataset, config):
    """
    完整版 MRM 预训练流程（包含模型保存），参考原 trainer.py。

    保存内容：
    - 每 save_every 轮保存一次 checkpoint
    - 最终保存 mrm_pretrained.pt
    """

    device = torch.device(config["training"]["device"])
    batch_size = config["training"]["batch_size"]
    lr = config["mrm"]["lr"]
    n_epochs = config["mrm"]["n_epochs"]
    mask_ratio = config["mrm"]["mask_ratio"]

    # 输出目录
    output_dir = Path(config["paths"]["output_dir"]) / "mrm_pretrain"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 每 N 轮保存一次
    save_every = config["mrm"].get("save_every", 10)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_patient_sequences)

    # 只训练 encoder + reconstruction head
    optimizer = torch.optim.Adam(
        list(model.row_encoder.parameters()) +
        list(model.time_encoder.parameters()) +
        list(model.reconstruction_head.parameters()),
        lr=lr,
    )

    # LR_schedular
    power = 1.2
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda epoch: (1 - epoch / n_epochs) ** power
    )

    loss_history = []  # 存每个 epoch 的 loss
    model.train()
    for epoch in range(1, n_epochs + 1):
        total_loss = 0.0

        for batch in tqdm(loader, desc=f"MRM Pre-train Epoch {epoch}"):
            x_cont = batch["x_cont"].to(device)
            x_cat = batch["x_cat"].to(device)
            mask = batch["mask"].to(device)
            times = batch["times"]
            missing_mask_cont = batch.get("missing_cont")
            missing_mask_cat = batch.get("missing_cat")

            # -------- MRM 随机 mask --------
            x_cont_masked, x_cat_masked, mrm_mask_cont, mrm_mask_cat = apply_mrm_mask(
                x_cont, x_cat, mask,
                missing_mask_cont, missing_mask_cat,
                mrm_mask_ratio=mask_ratio,
            )

            # -------- 编码 --------
            h = model.encode(
                x_cont_masked, x_cat_masked, mask, times,
                missing_mask_cont=missing_mask_cont,
                missing_mask_cat=missing_mask_cat,
                mrm_mask_cont=mrm_mask_cont,
                mrm_mask_cat=mrm_mask_cat,
            )

            # -------- 重建 --------
            cont_preds, cat_preds = model.reconstruct(h)

            # -------- MRM loss（只计算 MRM mask 位置） --------
            loss = compute_mrm_loss(
                cont_preds, cat_preds,
                x_cont, x_cat,
                mrm_mask_cont, mrm_mask_cat,
                missing_mask_cont, missing_mask_cat,
                mask,
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += float(loss.detach().cpu())

        loss_history.append(total_loss)
        print(f"[MRM Epoch {epoch}] Loss = {total_loss:.4f}")

        lr_scheduler.step()

        # ---------- 保存 checkpoint ----------
        if epoch % save_every == 0:
            ckpt_path = output_dir / f"mrm_epoch_{epoch}.pt"
            torch.save(model.state_dict(), ckpt_path)
            print(f"Saved checkpoint to {ckpt_path}")
            # ---------- 保存 loss 曲线图 ----------
            fig_path = output_dir / f"loss_curve_epoch_{epoch}.png"
            save_loss_curve(loss_history, fig_path)
            print(f"Saved loss curve to {fig_path}")

    # ---------- 保存最终模型 ----------
    final_path = output_dir / "mrm_pretrained.pt"
    torch.save(model.state_dict(), final_path)
    print(f"Saved final pretrained model to {final_path}")

    return model


# =====================
# Standalone 训练入口
# =====================
if __name__ == "__main__":
    import yaml
    from dataset.data import FeatureConfig
    from dataset.utils import prepare_dataset
    from models import TimeSeriesClusteringModel

    # 1. 加载 YAML 配置
    cfg = yaml.safe_load(open("config.yaml", encoding="utf-8"))

    # 2. 构建 feature 配置
    feature_cfg = FeatureConfig(
        patient_id_col=cfg["feature"]["patient_id_col"],
        time_col=cfg["feature"]["time_col"],
        cont_cols=cfg["feature"]["cont_cols"],
        cat_cols=cfg["feature"]["cat_cols"],
        static_cont_cols=cfg["feature"]["static_cont_cols"],
        static_cat_cols=cfg["feature"]["static_cat_cols"],
    )

    # 3. 准备 dataset
    dataset = prepare_dataset(
        cfg["paths"]["static_csv"],
        cfg["paths"]["events_csv"],
        feature_cfg,
        cfg["model"]["max_seq_len"],
    )

    # 4. 模型构建
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

    print("======================= 开始 MRM 预训练 =======================")
    train_mrm_pretrain(model, dataset, cfg)
    print("======================= 预训练完成 =======================")