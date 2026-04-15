"""Graph Convolutional Network for node-level delay-risk prediction."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, BatchNorm


class GCNModel(nn.Module):
    """Multi-layer GCN with batch-norm and skip connections.

    Parameters
    ----------
    in_dim : int
        Number of input features per node.
    hidden_dim : int
        Width of hidden GCN layers.
    out_dim : int
        Number of output classes.
    num_layers : int
        Total GCN layers (≥ 2).
    dropout : float
        Dropout rate applied after each hidden layer.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 64,
        out_dim: int = 2,
        num_layers: int = 3,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.dropout = dropout

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        # First layer
        self.convs.append(GCNConv(in_dim, hidden_dim))
        self.bns.append(BatchNorm(hidden_dim))

        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
            self.bns.append(BatchNorm(hidden_dim))

        # Output layer
        self.convs.append(GCNConv(hidden_dim, out_dim))

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """Forward pass through the GCN stack."""
        for i, (conv, bn) in enumerate(zip(self.convs[:-1], self.bns)):
            h = conv(x, edge_index, edge_weight=edge_weight)
            h = bn(h)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
            # Skip connection when dims match
            if h.shape == x.shape:
                h = h + x
            x = h

        # Final layer – no activation, raw logits
        x = self.convs[-1](x, edge_index, edge_weight=edge_weight)
        return x

    def reset_parameters(self) -> None:
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()
