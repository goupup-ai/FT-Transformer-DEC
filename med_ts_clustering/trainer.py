from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .models import TimeSeriesClusteringModel
from .Modules.dec import target_distribution, dec_loss
from .Modules.mrm import apply_mrm_mask, compute_mrm_loss
from .dataset.utils import prepare_dataset
from .dataset.data import FeatureConfig, MedicalTimeSeriesDataset, collate_patient_sequences


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


def train_dec(
    static_csv_path: str,
    events_csv_path: str,
    feature_cfg: FeatureConfig,
    *,
    n_clusters: int,
    d_model: int = 128,
    max_seq_len: int = 128,
    batch_size: int = 32,
    n_epochs: int = 20,
    device: Optional[str] = None,
    lr: float = 1e-4,
    ft_kwargs: Optional[Dict] = None,
    time_transformer_cfg: Optional[Dict] = None,
    output_dir: str = "outputs_dec",
    save_every: int = 1,
):
    """End-to-end training of FT-Transformer + Time Transformer + DEC."""
    device_t = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    dataset = prepare_dataset(static_csv_path, events_csv_path, feature_cfg, max_seq_len)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_patient_sequences,
    )

    n_cont_features = (
        len(feature_cfg.cont_cols) + len(feature_cfg.static_cont_cols)
    )
    n_cat_features = (
        len(feature_cfg.cat_cols) + len(feature_cfg.static_cat_cols)
    )
    # for FT-Transformer we need cardinalities list for categorical features
    # here we do not distinguish dynamic/static categorical for the encoder
    cat_cardinalities: List[int] = []
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
        time_transformer_cfg=time_transformer_cfg,
        ft_kwargs=ft_kwargs,
    ).to(device_t)

    # KMeans initialization
    init_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_patient_sequences,
    )
    initialize_kmeans(model, init_loader, device_t)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    model.train()
    for epoch in range(1, n_epochs + 1):
        total_loss = 0.0
        n_batches = 0
        for batch in tqdm(dataloader, desc=f"Epoch {epoch}"):
            x_cont = batch["x_cont"].to(device_t)  # (B, T, Cc)
            x_cat = batch["x_cat"].to(device_t)  # (B, T, Ck)
            mask = batch["mask"].to(device_t)  # (B, T)
            times = batch["times"]

            h, q = model(x_cont, x_cat, mask, times)  # h: (B,T,D), q:(B,T,K)
            valid_mask = mask  # True for valid
            q_valid = q[valid_mask]  # (N_valid, K)

            p = target_distribution(q_valid.detach())
            loss = dec_loss(q_valid, p)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += float(loss.detach().cpu())
            n_batches += 1

        avg_loss = total_loss / max(1, n_batches)
        print(f"[Epoch {epoch}] DEC loss = {avg_loss:.4f}")

        # Optional: save checkpoint every `save_every` epochs
        if epoch % max(1, save_every) == 0:
            ckpt_path = output_path / f"model_epoch_{epoch}.pt"
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "feature_cfg": feature_cfg.__dict__,
                },
                ckpt_path,
            )

    return model, dataset


def train_mrm_pretrain(
    model: TimeSeriesClusteringModel,
    dataloader: DataLoader,
    device: torch.device,
    n_epochs: int = 50,
    lr: float = 1e-4,
    mrm_mask_ratio: float = 0.2,
    save_every: Optional[int] = None,
    output_path: Optional[Path] = None,
) -> None:
    """Stage 1: MRM self-supervised pre-training."""
    if model.reconstruction_head is None:
        raise RuntimeError("Reconstruction head must be enabled for MRM pre-training.")
    
    # Only train encoder and reconstruction head, not DEC
    optimizer = torch.optim.Adam(
        list(model.row_encoder.parameters()) +
        list(model.time_encoder.parameters()) +
        list(model.reconstruction_head.parameters()),
        lr=lr
    )
    
    model.train()
    for epoch in range(1, n_epochs + 1):
        total_loss = 0.0
        n_batches = 0
        
        for batch in tqdm(dataloader, desc=f"MRM Pre-train Epoch {epoch}"):
            x_cont = batch["x_cont"].to(device)  # (B, T, Cc)
            x_cat = batch["x_cat"].to(device)  # (B, T, Ck)
            mask = batch["mask"].to(device)  # (B, T)
            times = batch["times"]
            missing_mask_cont = batch.get("missing_cont", torch.zeros_like(x_cont, dtype=torch.bool)).to(device)
            missing_mask_cat = batch.get("missing_cat", torch.zeros_like(x_cat, dtype=torch.bool)).to(device)
            
            # Apply MRM masking
            x_cont_masked, x_cat_masked, mrm_mask_cont, mrm_mask_cat = apply_mrm_mask(
                x_cont, x_cat, mask,
                missing_mask_cont, missing_mask_cat,
                mrm_mask_ratio=mrm_mask_ratio,
            )
            
            # Forward pass
            h = model.encode(
                x_cont_masked, x_cat_masked, mask, times,
                missing_mask_cont=missing_mask_cont,
                missing_mask_cat=missing_mask_cat,
                mrm_mask_cont=mrm_mask_cont,
                mrm_mask_cat=mrm_mask_cat,
            )  # (B, T, D)
            
            # Reconstruction
            cont_preds, cat_preds = model.reconstruct(h)  # cont: (B, T, Cc), cat: list
            
            # Compute loss (only on MRM-masked fields, not originally missing)
            loss = compute_mrm_loss(
                cont_preds, cat_preds,
                x_cont, x_cat,  # original targets
                mrm_mask_cont, mrm_mask_cat,
                missing_mask_cont, missing_mask_cat,
                mask,
            )
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += float(loss.detach().cpu())
            n_batches += 1
        
        avg_loss = total_loss / max(1, n_batches)
        print(f"[MRM Epoch {epoch}] Loss = {avg_loss:.4f}")
        
        if save_every and epoch % save_every == 0 and output_path:
            ckpt_path = output_path / f"mrm_pretrain_epoch_{epoch}.pt"
            torch.save(model.state_dict(), ckpt_path)
            print(f"Saved checkpoint to {ckpt_path}")


def train_mrm_dec_pipeline(
    static_csv_path: str,
    events_csv_path: str,
    feature_cfg: FeatureConfig,
    *,
    n_clusters: int,
    d_model: int = 128,
    max_seq_len: int = 128,
    batch_size: int = 32,
    # Stage 1: MRM pre-training
    mrm_n_epochs: int = 50,
    mrm_lr: float = 1e-4,
    mrm_mask_ratio: float = 0.2,
    # Stage 2: KMeans init (no training, just initialization)
    # Stage 3: DEC fine-tuning
    dec_n_epochs: int = 50,
    dec_lr: float = 1e-4,
    device: Optional[str] = None,
    ft_kwargs: Optional[Dict] = None,
    time_transformer_cfg: Optional[Dict] = None,
    output_dir: str = "outputs_mrm_dec",
    save_every: int = 10,
):
    """Complete three-stage training pipeline: MRM pre-training -> KMeans init -> DEC fine-tuning.
    
    Stage 1: MRM self-supervised pre-training
    Stage 2: KMeans initialization of DEC cluster centers
    Stage 3: DEC clustering fine-tuning
    """
    device_t = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    
    # Prepare dataset
    dataset = prepare_dataset(static_csv_path, events_csv_path, feature_cfg, max_seq_len)
    
    # Compute feature dimensions
    n_cont_features = (
        len(feature_cfg.cont_cols) + len(feature_cfg.static_cont_cols)
    )
    cat_cardinalities: List[int] = []
    for col in feature_cfg.cat_cols:
        cat_cardinalities.append(
            int(dataset.artifacts.cat_vocab_maps[col].__len__())
        )
    for col in feature_cfg.static_cat_cols:
        cat_cardinalities.append(
            int(dataset.artifacts.static_cat_vocab_maps[col].__len__())
        )
    
    # Create model with reconstruction head enabled
    model = TimeSeriesClusteringModel(
        n_cont_features=n_cont_features,
        cat_cardinalities=cat_cardinalities,
        d_model=d_model,
        n_clusters=n_clusters,
        time_transformer_cfg=time_transformer_cfg,
        ft_kwargs=ft_kwargs,
        use_missing_embedding=True,
        use_mask_embedding=True,
        enable_reconstruction=True,  # Enable for MRM
    ).to(device_t)
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # ========================================================================
    # Stage 1: MRM Self-Supervised Pre-training
    # ========================================================================
    print("=" * 80)
    print("Stage 1: MRM Self-Supervised Pre-training")
    print("=" * 80)
    
    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_patient_sequences,
    )
    
    train_mrm_pretrain(
        model, train_loader, device_t,
        n_epochs=mrm_n_epochs,
        lr=mrm_lr,
        mrm_mask_ratio=mrm_mask_ratio,
        save_every=save_every,
        output_path=output_path,
    )
    
    # Save MRM pre-trained model
    mrm_ckpt_path = output_path / "mrm_pretrained.pt"
    torch.save(model.state_dict(), mrm_ckpt_path)
    print(f"Saved MRM pre-trained model to {mrm_ckpt_path}")
    
    # ========================================================================
    # Stage 2: KMeans Initialization
    # ========================================================================
    print("=" * 80)
    print("Stage 2: KMeans Initialization of DEC Cluster Centers")
    print("=" * 80)
    
    init_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_patient_sequences,
    )
    initialize_kmeans(model, init_loader, device_t)
    print("KMeans initialization completed.")
    
    # ========================================================================
    # Stage 3: DEC Fine-tuning
    # ========================================================================
    print("=" * 80)
    print("Stage 3: DEC Clustering Fine-tuning")
    print("=" * 80)
    
    # Freeze reconstruction head
    if model.reconstruction_head is not None:
        for param in model.reconstruction_head.parameters():
            param.requires_grad = False
        print("Reconstruction head frozen.")
    
    # Optimizer for DEC fine-tuning: encoder + DEC centers
    # Use smaller LR for encoder (fine-tuning), normal LR for DEC centers
    optimizer = torch.optim.Adam(
        [
            {"params": model.row_encoder.parameters(), "lr": dec_lr * 0.1},  # Smaller LR for encoder
            {"params": model.time_encoder.parameters(), "lr": dec_lr * 0.1},
            {"params": model.clustering.parameters(), "lr": dec_lr},  # Normal LR for DEC
        ],
        lr=dec_lr
    )
    
    model.train()
    for epoch in range(1, dec_n_epochs + 1):
        total_loss = 0.0
        n_batches = 0
        
        for batch in tqdm(train_loader, desc=f"DEC Fine-tune Epoch {epoch}"):
            x_cont = batch["x_cont"].to(device_t)  # (B, T, Cc)
            x_cat = batch["x_cat"].to(device_t)  # (B, T, Ck)
            mask = batch["mask"].to(device_t)  # (B, T)
            times = batch["times"]
            missing_mask_cont = batch.get("missing_cont", torch.zeros_like(x_cont, dtype=torch.bool)).to(device_t)
            missing_mask_cat = batch.get("missing_cat", torch.zeros_like(x_cat, dtype=torch.bool)).to(device_t)
            
            # Forward pass (no MRM masking in DEC stage)
            h, q = model(
                x_cont, x_cat, mask, times,
                missing_mask_cont=missing_mask_cont,
                missing_mask_cat=missing_mask_cat,
            )  # h: (B, T, D), q: (B, T, K)
            
            valid_mask = mask  # True for valid
            q_valid = q[valid_mask]  # (N_valid, K)
            
            # DEC loss
            p = target_distribution(q_valid.detach())
            loss = dec_loss(q_valid, p)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += float(loss.detach().cpu())
            n_batches += 1
        
        avg_loss = total_loss / max(1, n_batches)
        print(f"[DEC Epoch {epoch}] Loss = {avg_loss:.4f}")
        
        if epoch % max(1, save_every) == 0:
            ckpt_path = output_path / f"dec_finetune_epoch_{epoch}.pt"
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "feature_cfg": feature_cfg.__dict__,
                    "epoch": epoch,
                },
                ckpt_path,
            )
            print(f"Saved checkpoint to {ckpt_path}")
    
    # Save final model
    final_ckpt_path = output_path / "final_model.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "feature_cfg": feature_cfg.__dict__,
        },
        final_ckpt_path,
    )
    print(f"Saved final model to {final_ckpt_path}")
    
    return model, dataset


def export_cluster_assignments(
    model: TimeSeriesClusteringModel,
    dataset: MedicalTimeSeriesDataset,
    *,
    feature_cfg: FeatureConfig,
    max_seq_len: int,
    batch_size: int = 32,
    device: Optional[str] = None,
    output_csv: str = "cluster_assignments.csv",
):
    """Export final cluster_id per event row."""
    device_t = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = model.to(device_t)
    model.eval()

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_patient_sequences,
    )

    records = []
    embeddings_list = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Export clusters"):
            x_cont = batch["x_cont"].to(device_t)
            x_cat = batch["x_cat"].to(device_t)
            mask = batch["mask"].to(device_t)
            times_list = batch["times"]
            patient_ids = batch["patient_ids"]
            missing_mask_cont = batch.get("missing_cont", torch.zeros_like(x_cont, dtype=torch.bool)).to(device_t)
            missing_mask_cat = batch.get("missing_cat", torch.zeros_like(x_cat, dtype=torch.bool)).to(device_t)

            h, q = model(
                x_cont, x_cat, mask, times_list,
                missing_mask_cont=missing_mask_cont,
                missing_mask_cat=missing_mask_cat,
            )
            # predicted cluster id = argmax over K
            cluster_ids = q.argmax(dim=-1).cpu().numpy()  # (B, T)
            mask_np = mask.cpu().numpy()

            B, T = cluster_ids.shape
            for i in range(B):
                pid = patient_ids[i]
                times = times_list[i]
                for t_idx in range(min(len(times), T)):
                    if not mask_np[i, t_idx]:
                        continue
                    records.append(
                        {
                            feature_cfg.patient_id_col: pid,
                            feature_cfg.time_col: times[t_idx],
                            "cluster_id": int(cluster_ids[i, t_idx]),
                        }
                    )
                    # Store embedding for visualization
                    embeddings_list.append(h[i, t_idx].cpu().numpy())

    df = pd.DataFrame.from_records(records)
    df.to_csv(output_csv, index=False)
    print(f"Saved cluster assignments to {output_csv}")
    
    # Optionally save embeddings
    if embeddings_list:
        embeddings_np = np.array(embeddings_list)
        embeddings_path = output_csv.replace(".csv", "_embeddings.npy")
        np.save(embeddings_path, embeddings_np)
        print(f"Saved embeddings to {embeddings_path}")
    
    return df



