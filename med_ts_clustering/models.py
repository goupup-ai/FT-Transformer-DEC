from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from .Modules.ft_transformer import FTTransformer
from .Modules.time_transformer import TimeTransformer
from .Modules.dec import ClusteringLayer
from .Modules.mrm import MissingEmbedding, MaskEmbedding, ReconstructionHead


class RowEncoderFTTransformer(nn.Module):
    """Wrap FTTransformer to work on (B, T, *) inputs.

    We treat each (patient, time) row as an independent sample for the FT-Transformer,
    then reshape back to (B, T, d_model).
    
    Supports missing and mask embeddings for MRM pre-training.
    """

    def __init__(
        self,
        n_cont_features: int,
        cat_cardinalities: List[int],
        d_model: int,
        ft_kwargs: Optional[Dict] = None,
        use_missing_embedding: bool = True,
        use_mask_embedding: bool = True,
    ) -> None:
        super().__init__()
        ft_kwargs = dict(ft_kwargs or {})
        # Backbone (FTTransformerBackbone) expects at least d_block and n_blocks.
        # Here we set sensible defaults if the user didn't provide them.
        ft_kwargs.setdefault("d_block", d_model)
        ft_kwargs.setdefault("n_blocks", 3)
        ft_kwargs.setdefault("d_out", d_model)
        ft_kwargs.setdefault("ffn_d_hidden", None)
        ft_kwargs.setdefault("ffn_d_hidden_multiplier", 4 / 3)
        ft_kwargs.setdefault("attention_n_heads", 8)
        ft_kwargs.setdefault("attention_dropout", 0.1)
        ft_kwargs.setdefault("ffn_dropout", 0.1)
        ft_kwargs.setdefault("residual_dropout", 0.0)

        self.ft = FTTransformer(
            n_cont_features=n_cont_features,
            cat_cardinalities=cat_cardinalities,
            **ft_kwargs,
        )
        self.d_model = d_model
        self.n_cont_features = n_cont_features
        self.n_cat_features = len(cat_cardinalities)
        self.use_missing_embedding = use_missing_embedding
        self.use_mask_embedding = use_mask_embedding
        
        if use_missing_embedding:
            self.missing_embedding = MissingEmbedding(d_model)
        else:
            self.missing_embedding = None
            
        if use_mask_embedding:
            self.mask_embedding = MaskEmbedding(d_model)
        else:
            self.mask_embedding = None

    def forward(
        self,
        x_cont: Tensor,
        x_cat: Tensor,
        mask: Tensor,
        missing_mask_cont: Optional[Tensor] = None,
        missing_mask_cat: Optional[Tensor] = None,
        mrm_mask_cont: Optional[Tensor] = None,
        mrm_mask_cat: Optional[Tensor] = None,
    ) -> Tensor:
        """Encode per-row features.

        Args:
            x_cont: (B, T, Cc)
            x_cat: (B, T, Ck)
            mask:  (B, T) bool, True=valid
            missing_mask_cont: (B, T, Cc) bool, True where originally missing
            missing_mask_cat: (B, T, Ck) bool, True where originally missing
            mrm_mask_cont: (B, T, Cc) bool, True where MRM masked
            mrm_mask_cat: (B, T, Ck) bool, True where MRM masked
        Returns:
            z: (B, T, d_model)
        """
        B, T, _ = x_cont.shape
        # flatten valid positions
        x_cont_flat = x_cont.reshape(B * T, -1)
        x_cat_flat = x_cat.reshape(B * T, -1)
        z_flat = self.ft(x_cont_flat, x_cat_flat)  # (B*T, d_model)
        z = z_flat.reshape(B, T, self.d_model)
        
        # Add missing and mask embeddings at row level
        # We compute a weighted sum based on the proportion of missing/masked fields
        if self.missing_embedding is not None and missing_mask_cont is not None:
            # Compute missing ratio per row
            missing_ratio_cont = (
                missing_mask_cont.float().mean(dim=-1, keepdim=True)
                if missing_mask_cont.shape[-1] > 0
                else torch.zeros(B, T, 1, device=z.device)
            )
            missing_ratio_cat = (
                missing_mask_cat.float().mean(dim=-1, keepdim=True)
                if missing_mask_cat is not None and missing_mask_cat.shape[-1] > 0
                else torch.zeros(B, T, 1, device=z.device)
            )
            # Average ratio across cont and cat
            total_features = self.n_cont_features + self.n_cat_features
            if total_features > 0:
                missing_ratio = (
                    missing_ratio_cont * self.n_cont_features +
                    missing_ratio_cat * self.n_cat_features
                ) / total_features
            else:
                missing_ratio = torch.zeros(B, T, 1, device=z.device)
            # Add missing embedding weighted by ratio
            missing_emb = self.missing_embedding(z.shape) * missing_ratio
            z = z + missing_emb
        
        if self.mask_embedding is not None and mrm_mask_cont is not None:
            # Compute mask ratio per row
            mask_ratio_cont = (
                mrm_mask_cont.float().mean(dim=-1, keepdim=True)
                if mrm_mask_cont.shape[-1] > 0
                else torch.zeros(B, T, 1, device=z.device)
            )
            mask_ratio_cat = (
                mrm_mask_cat.float().mean(dim=-1, keepdim=True)
                if mrm_mask_cat is not None and mrm_mask_cat.shape[-1] > 0
                else torch.zeros(B, T, 1, device=z.device)
            )
            # Average ratio across cont and cat
            total_features = self.n_cont_features + self.n_cat_features
            if total_features > 0:
                mask_ratio = (
                    mask_ratio_cont * self.n_cont_features +
                    mask_ratio_cat * self.n_cat_features
                ) / total_features
            else:
                mask_ratio = torch.zeros(B, T, 1, device=z.device)
            # Add mask embedding weighted by ratio
            mask_emb = self.mask_embedding(z.shape) * mask_ratio
            z = z + mask_emb
        
        # we keep padded positions as is; mask will be used later
        return z


class TimeSeriesClusteringModel(nn.Module):
    """End-to-end model: FT-Transformer row encoder + Time Transformer + DEC.
    
    Supports MRM pre-training and DEC fine-tuning.
    """

    def __init__(
        self,
        *,
        n_cont_features: int,
        cat_cardinalities: List[int],
        d_model: int,
        n_clusters: int,
        time_transformer_cfg: Optional[Dict] = None,
        ft_kwargs: Optional[Dict] = None,
        use_missing_embedding: bool = True,
        use_mask_embedding: bool = True,
        enable_reconstruction: bool = False,
    ) -> None:
        super().__init__()
        self.row_encoder = RowEncoderFTTransformer(
            n_cont_features=n_cont_features,
            cat_cardinalities=cat_cardinalities,
            d_model=d_model,
            ft_kwargs=ft_kwargs,
            use_missing_embedding=use_missing_embedding,
            use_mask_embedding=use_mask_embedding,
        )
        tt_cfg = dict(time_transformer_cfg or {})
        tt_cfg.setdefault("d_model", d_model)
        self.time_encoder = TimeTransformer(**tt_cfg)
        self.clustering = ClusteringLayer(
            n_clusters=n_clusters, embedding_dim=d_model
        )
        
        # Reconstruction head for MRM pre-training
        self.enable_reconstruction = enable_reconstruction
        if enable_reconstruction:
            self.reconstruction_head = ReconstructionHead(
                d_model=d_model,
                n_cont_features=n_cont_features,
                cat_cardinalities=cat_cardinalities,
            )
        else:
            self.reconstruction_head = None

    def encode(
        self,
        x_cont: Tensor,
        x_cat: Tensor,
        mask: Tensor,
        times: Optional[List] = None,
        missing_mask_cont: Optional[Tensor] = None,
        missing_mask_cat: Optional[Tensor] = None,
        mrm_mask_cont: Optional[Tensor] = None,
        mrm_mask_cat: Optional[Tensor] = None,
    ) -> Tensor:
        """Return contextualized per-row embeddings h_t (B, T, d_model)."""
        z = self.row_encoder(
            x_cont, x_cat, mask,
            missing_mask_cont=missing_mask_cont,
            missing_mask_cat=missing_mask_cat,
            mrm_mask_cont=mrm_mask_cont,
            mrm_mask_cat=mrm_mask_cat,
        )  # (B, T, d_model)
        h = self.time_encoder(z, mask=mask, times=times)  # (B, T, d_model)
        return h

    def forward(
        self,
        x_cont: Tensor,
        x_cat: Tensor,
        mask: Tensor,
        times: Optional[List] = None,
        missing_mask_cont: Optional[Tensor] = None,
        missing_mask_cat: Optional[Tensor] = None,
        mrm_mask_cont: Optional[Tensor] = None,
        mrm_mask_cat: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """Return h and soft cluster assignments q for all valid positions.

        Returns:
            h: (B, T, d_model)
            q: (B, T, K) with padded positions arbitrary (should be ignored by mask)
        """
        h = self.encode(
            x_cont, x_cat, mask, times,
            missing_mask_cont=missing_mask_cont,
            missing_mask_cat=missing_mask_cat,
            mrm_mask_cont=mrm_mask_cont,
            mrm_mask_cat=mrm_mask_cat,
        )
        B, T, D = h.shape
        K = self.clustering.n_clusters

        h_flat = h.reshape(B * T, D)
        q_flat = self.clustering(h_flat)  # (B*T, K)
        q = q_flat.reshape(B, T, K)
        return h, q
    
    def reconstruct(
        self,
        h: Tensor,
    ) -> Tuple[Tensor, List[Tensor]]:
        """Reconstruct features from embeddings (for MRM pre-training).
        
        Args:
            h: (B, T, d_model) embeddings
        
        Returns:
            cont_preds: (B, T, n_cont_features)
            cat_preds: list of (B, T, card_i) logits
        """
        if self.reconstruction_head is None:
            raise RuntimeError("Reconstruction head not enabled. Set enable_reconstruction=True.")
        return self.reconstruction_head(h)



