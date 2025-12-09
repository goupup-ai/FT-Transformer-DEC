from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor


class TimePositionalEncoding(nn.Module):
    """Learnable positional + (optional) continuous time embedding."""

    def __init__(
        self,
        d_model: int,
        max_len: int,
        use_time_scalar: bool = True,
    ) -> None:
        super().__init__()
        self.position_emb = nn.Embedding(max_len, d_model)
        self.use_time_scalar = use_time_scalar
        if use_time_scalar:
            self.time_proj = nn.Linear(1, d_model)
        else:
            self.time_proj = None

    def forward(
        self,
        x: Tensor,  # (B, T, d_model)
        times: Optional[list] = None,
    ) -> Tensor:
        B, T, d_model = x.shape
        device = x.device
        positions = torch.arange(T, device=device).unsqueeze(0).expand(B, T)
        pos_emb = self.position_emb(positions)  # (B, T, d_model)
        out = x + pos_emb

        if self.use_time_scalar and times is not None:
            # times: list of np.ndarray, length B, each (T_valid,)
            # we assume already truncated/padded; pad last time for padded positions
            time_tensor = torch.zeros(B, T, 1, device=device)
            for i, ts in enumerate(times):
                ts = torch.as_tensor(ts, dtype=torch.float32, device=device)
                L = min(len(ts), T)
                time_tensor[i, :L, 0] = ts[:L]
                if L < T and L > 0:
                    time_tensor[i, L:, 0] = ts[L - 1]
            time_emb = self.time_proj(time_tensor)  # (B, T, d_model)
            out = out + time_emb
        return out


class TimeTransformer(nn.Module):
    """Transformer encoder over time for per-row embeddings.

    Input:  z  of shape (B, T, d_model), mask (B, T) where True=valid, False=pad.
    Output: h of shape (B, T, d_model).
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int = 4,
        n_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        max_len: int = 512,
        use_time_scalar: bool = True,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.pos_encoding = TimePositionalEncoding(
            d_model=d_model,
            max_len=max_len,
            use_time_scalar=use_time_scalar,
        )

    def forward(
        self,
        z: Tensor,  # (B, T, d_model)
        mask: Tensor,  # (B, T) bool, True=valid
        times: Optional[list] = None,
    ) -> Tensor:
        # src_key_padding_mask: (B, T) with True for PAD; invert mask
        src_key_padding_mask = ~mask
        x = self.pos_encoding(z, times=times)
        h = self.encoder(x, src_key_padding_mask=src_key_padding_mask)
        return h



