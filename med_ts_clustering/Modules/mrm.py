"""MRM (Masked Row Modeling) modules for self-supervised pre-training."""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
from torch import Tensor


def apply_mrm_mask(
    x_cont: Tensor,
    x_cat: Tensor,
    mask: Tensor,
    missing_mask_cont: Tensor,
    missing_mask_cat: Tensor,
    mrm_mask_ratio: float = 0.2,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Apply MRM masking to input features.
    
    Args:
        x_cont: (B, T, Cc) continuous features
        x_cat: (B, T, Ck) categorical features
        mask: (B, T) bool, True for valid positions
        missing_mask_cont: (B, T, Cc) bool, True where values are originally missing
        missing_mask_cat: (B, T, Ck) bool, True where values are originally missing
        mrm_mask_ratio: ratio of fields to mask (0.15-0.30)
    
    Returns:
        x_cont_masked: (B, T, Cc) with masked values filled (median/0)
        x_cat_masked: (B, T, Ck) with masked values filled (0)
        mrm_mask_cont: (B, T, Cc) bool, True where MRM masked
        mrm_mask_cat: (B, T, Ck) bool, True where MRM masked
    """
    B, T, Cc = x_cont.shape
    _, _, Ck = x_cat.shape
    device = x_cont.device
    
    # Only mask valid positions
    valid_positions = mask  # (B, T)
    
    # Initialize MRM masks
    mrm_mask_cont = torch.zeros_like(missing_mask_cont, dtype=torch.bool)
    mrm_mask_cat = torch.zeros_like(missing_mask_cat, dtype=torch.bool)
    
    # Copy inputs for masking
    x_cont_masked = x_cont.clone()
    x_cat_masked = x_cat.clone()
    
    # For each valid row, randomly mask fields
    for b in range(B):
        for t in range(T):
            if not valid_positions[b, t]:
                continue
            
            # Total number of fields in this row
            n_fields = Cc + Ck
            
            # Number of fields to mask (excluding already missing ones)
            n_to_mask = max(1, int(n_fields * mrm_mask_ratio))
            
            # Create candidate indices (exclude already missing fields)
            cont_candidates = torch.arange(Cc, device=device)[
                ~missing_mask_cont[b, t]
            ]
            cat_candidates = torch.arange(Ck, device=device)[
                ~missing_mask_cat[b, t]
            ]
            
            # Randomly select fields to mask
            total_candidates = len(cont_candidates) + len(cat_candidates)
            if total_candidates > 0:
                # Sample from available candidates
                n_cont_to_mask = 0
                n_cat_to_mask = 0
                
                if len(cont_candidates) > 0 and len(cat_candidates) > 0:
                    # Both types available, randomly split
                    n_cont_to_mask = torch.randint(
                        0, min(len(cont_candidates), n_to_mask) + 1, (1,)
                    ).item()
                    n_cat_to_mask = min(n_to_mask - n_cont_to_mask, len(cat_candidates))
                    n_cont_to_mask = min(n_cont_to_mask, len(cont_candidates))
                elif len(cont_candidates) > 0:
                    n_cont_to_mask = min(n_to_mask, len(cont_candidates))
                elif len(cat_candidates) > 0:
                    n_cat_to_mask = min(n_to_mask, len(cat_candidates))
                
                if len(cont_candidates) > 0 and n_cont_to_mask > 0:
                    cont_selected = cont_candidates[
                        torch.randperm(len(cont_candidates), device=device)[:n_cont_to_mask]
                    ]
                    mrm_mask_cont[b, t, cont_selected] = True
                    # Fill masked continuous values with 0 (or median if available)
                    x_cont_masked[b, t, cont_selected] = 0.0
                
                if len(cat_candidates) > 0 and n_cat_to_mask > 0:
                    cat_selected = cat_candidates[
                        torch.randperm(len(cat_candidates), device=device)[:n_cat_to_mask]
                    ]
                    mrm_mask_cat[b, t, cat_selected] = True
                    # Fill masked categorical values with 0
                    x_cat_masked[b, t, cat_selected] = 0
    
    return x_cont_masked, x_cat_masked, mrm_mask_cont, mrm_mask_cat


class MissingEmbedding(nn.Module):
    """Learnable embedding for missing values."""
    
    def __init__(self, d_model: int):
        super().__init__()
        self.embedding = nn.Parameter(torch.randn(d_model) * 0.02)
    
    def forward(self, shape: Tuple[int, ...]) -> Tensor:
        """Return embedding broadcasted to shape.
        
        Args:
            shape: (..., d_model) target shape
        
        Returns:
            embedding: (..., d_model)
        """
        return self.embedding.expand(*shape)


class MaskEmbedding(nn.Module):
    """Learnable embedding for MRM masked values."""
    
    def __init__(self, d_model: int):
        super().__init__()
        self.embedding = nn.Parameter(torch.randn(d_model) * 0.02)
    
    def forward(self, shape: Tuple[int, ...]) -> Tensor:
        """Return embedding broadcasted to shape.
        
        Args:
            shape: (..., d_model) target shape
        
        Returns:
            embedding: (..., d_model)
        """
        return self.embedding.expand(*shape)


class ReconstructionHead(nn.Module):
    """Reconstruction head for MRM pre-training.
    
    Reconstructs masked continuous and categorical features.
    """
    
    def __init__(
        self,
        d_model: int,
        n_cont_features: int,
        cat_cardinalities: list[int],
    ):
        super().__init__()
        self.n_cont_features = n_cont_features
        self.cat_cardinalities = cat_cardinalities
        
        # Continuous decoders: one per continuous feature
        if n_cont_features > 0:
            self.cont_decoders = nn.ModuleList([
                nn.Linear(d_model, 1) for _ in range(n_cont_features)
            ])
        else:
            self.cont_decoders = nn.ModuleList()
        
        # Categorical decoders: one per categorical feature
        if cat_cardinalities:
            self.cat_decoders = nn.ModuleList([
                nn.Linear(d_model, card) for card in cat_cardinalities
            ])
        else:
            self.cat_decoders = nn.ModuleList()
    
    def forward(
        self,
        h: Tensor,  # (B, T, d_model)
    ) -> Tuple[Tensor, Tensor]:
        """Reconstruct features from embeddings.
        
        Returns:
            cont_preds: (B, T, n_cont_features)
            cat_preds: (B, T, n_cat_features) logits for each categorical feature
        """
        B, T, D = h.shape
        
        # Continuous predictions
        cont_preds = []
        for decoder in self.cont_decoders:
            pred = decoder(h)  # (B, T, 1)
            cont_preds.append(pred)
        cont_preds = torch.cat(cont_preds, dim=-1) if cont_preds else torch.empty(
            B, T, 0, device=h.device
        )
        
        # Categorical predictions (logits)
        cat_preds = []
        for decoder in self.cat_decoders:
            logits = decoder(h)  # (B, T, card)
            cat_preds.append(logits)
        # Stack along a new dimension: (B, T, n_cat_features, card_i)
        # For simplicity, we'll return as list or flatten
        # Actually, we need to handle variable cardinalities
        # Return as list for now, caller will handle per-feature loss
        return cont_preds, cat_preds


def compute_mrm_loss(
    cont_preds: Tensor,  # (B, T, Cc)
    cat_preds: list[Tensor],  # list of (B, T, card_i)
    cont_targets: Tensor,  # (B, T, Cc)
    cat_targets: Tensor,  # (B, T, Ck)
    mrm_mask_cont: Tensor,  # (B, T, Cc)
    mrm_mask_cat: Tensor,  # (B, T, Ck)
    missing_mask_cont: Tensor,  # (B, T, Cc)
    missing_mask_cat: Tensor,  # (B, T, Ck)
    mask: Tensor,  # (B, T)
) -> Tensor:
    """Compute MRM reconstruction loss.
    
    Only compute loss for MRM-masked fields (not originally missing).
    
    Args:
        cont_preds: predicted continuous values
        cat_preds: list of predicted categorical logits
        cont_targets: ground truth continuous values
        cat_targets: ground truth categorical indices
        mrm_mask_cont: MRM mask for continuous features
        mrm_mask_cat: MRM mask for categorical features
        missing_mask_cont: original missing mask for continuous
        missing_mask_cat: original missing mask for categorical
        mask: valid position mask
    
    Returns:
        loss: scalar tensor
    """
    device = cont_preds.device
    losses = []
    
    # Continuous loss: MSE only on MRM-masked (not originally missing)
    if cont_preds.shape[-1] > 0:
        # Only compute on MRM-masked AND not originally missing
        valid_cont_mask = mrm_mask_cont & (~missing_mask_cont) & mask.unsqueeze(-1)
        if valid_cont_mask.any():
            cont_loss = nn.functional.mse_loss(
                cont_preds[valid_cont_mask],
                cont_targets[valid_cont_mask],
                reduction='mean'
            )
            losses.append(cont_loss)
    
    # Categorical loss: CrossEntropy per feature
    if cat_preds:
        for i, cat_logits in enumerate(cat_preds):
            # cat_logits: (B, T, card_i)
            # cat_targets[:, :, i]: (B, T)
            # Only compute on MRM-masked AND not originally missing
            valid_cat_mask = (
                mrm_mask_cat[:, :, i] & 
                (~missing_mask_cat[:, :, i]) & 
                mask
            )
            if valid_cat_mask.any():
                # Flatten for cross entropy
                logits_flat = cat_logits.reshape(-1, cat_logits.shape[-1])
                targets_flat = cat_targets[:, :, i].reshape(-1)
                mask_flat = valid_cat_mask.reshape(-1)
                
                # Compute loss only on valid positions
                if mask_flat.any():
                    cat_loss = nn.functional.cross_entropy(
                        logits_flat[mask_flat],
                        targets_flat[mask_flat],
                        reduction='mean'
                    )
                    losses.append(cat_loss)
    
    if not losses:
        # Return zero loss if nothing to compute
        return torch.tensor(0.0, device=device, requires_grad=True)
    
    return sum(losses) / len(losses)  # Average over feature types

