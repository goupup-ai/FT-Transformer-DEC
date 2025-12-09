from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
from torch import Tensor
from sklearn.cluster import KMeans


class ClusteringLayer(nn.Module):
    """DEC clustering layer with Student t-distribution."""

    def __init__(self, n_clusters: int, embedding_dim: int, alpha: float = 1.0):
        super().__init__()
        self.n_clusters = n_clusters
        self.embedding_dim = embedding_dim
        self.alpha = alpha
        self.cluster_centers = nn.Parameter(torch.Tensor(n_clusters, embedding_dim))
        nn.init.xavier_uniform_(self.cluster_centers.data)

    def forward(self, z: Tensor) -> Tensor:
        """Compute soft assignment q.

        Args:
            z: (N, D) embeddings
        Returns:
            q: (N, K)
        """
        # pairwise squared distance to cluster centers
        diff = z.unsqueeze(1) - self.cluster_centers  # (N, K, D)
        dist_sq = (diff**2).sum(dim=2)  # (N, K)
        q = 1.0 / (1.0 + dist_sq / self.alpha)
        q = q ** ((self.alpha + 1.0) / 2.0)
        q = q / q.sum(dim=1, keepdim=True)
        return q

    @torch.no_grad()
    def init_from_kmeans(self, embeddings: Tensor) -> None:
        """Initialize cluster centers with KMeans on given embeddings."""
        device = self.cluster_centers.device
        x = embeddings.detach().cpu().numpy()
        km = KMeans(n_clusters=self.n_clusters, n_init=20)
        km.fit(x)
        self.cluster_centers.data.copy_(torch.from_numpy(km.cluster_centers_).to(device))


def target_distribution(q: Tensor) -> Tensor:
    """Compute DEC target distribution p from soft assignments q."""
    # q: (N, K)
    # weight_ik = q_ik^2 / sum_i q_ik
    weight = q**2 / q.sum(0, keepdim=True)
    # normalize per sample i (row-wise), keeping p row-stochastic
    # note: weight.sum(1) has shape (N,) so it broadcasts correctly over weight.t() of shape (K, N)
    return (weight.t() / weight.sum(1)).t()


def dec_loss(q: Tensor, p: Tensor) -> Tensor:
    """KL divergence between target distribution p and q."""
    return torch.nn.functional.kl_div(
        q.log(), p, reduction="batchmean"
    )  # KL(p || q) ~ sum p * log(p/q)



