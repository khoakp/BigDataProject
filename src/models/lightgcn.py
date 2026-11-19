"""LightGCN propagation over the pre-normalized D06 graph."""

import torch
from torch import nn
from torch.nn import functional as F

from src.graph.normalize import SparseAdjacency


def to_sparse_tensor(adjacency: SparseAdjacency, device: torch.device) -> torch.Tensor:
    """Convert D06 float64 COO to float32 without renormalizing weights."""
    sources, targets, weights = zip(*adjacency.entries) if adjacency.entries else ((), (), ())
    indices = torch.tensor((sources, targets), dtype=torch.long, device=device)
    values = torch.tensor(weights, dtype=torch.float32, device=device)
    return torch.sparse_coo_tensor(indices, values, adjacency.shape).coalesce()


class LightGCN(nn.Module):
    def __init__(self, num_nodes: int, embedding_dim: int, layers: int):
        super().__init__()
        if min(num_nodes, embedding_dim) <= 0 or layers < 0:
            raise ValueError("Expected positive node count/dimension and nonnegative layers")
        self.embedding = nn.Embedding(num_nodes, embedding_dim)
        nn.init.normal_(self.embedding.weight, std=0.1)
        self.layers = layers

    def forward(self, adjacency: torch.Tensor) -> torch.Tensor:
        """Mean of the initial embedding and each propagation layer."""
        current = self.embedding.weight
        if adjacency.layout != torch.sparse_coo or adjacency.shape != (len(current), len(current)):
            raise ValueError("Expected a square COO adjacency matching the node embedding table")
        total = current
        for _ in range(self.layers):
            current = torch.sparse.mm(adjacency, current)
            total = total + current
        return total / (self.layers + 1)


def pairwise_loss(embeddings: torch.Tensor, triples: torch.Tensor) -> torch.Tensor:
    """Unregularized BPR term on (user, positive movie, unobserved movie)."""
    users, positive, negative = triples.unbind(dim=1)
    margin = (embeddings[users] * (embeddings[positive] - embeddings[negative])).sum(dim=1)
    return F.softplus(-margin).mean()
