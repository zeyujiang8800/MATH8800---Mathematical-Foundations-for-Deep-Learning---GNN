"""Simple MLP baseline for node classification (ignores graph structure)."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPBaseline(nn.Module):
    """Two-hidden-layer MLP operating on node features only.

    Parameters
    ----------
    in_dim : int
        Number of input features per node.
    hidden_dim : int
        Width of hidden layers.
    out_dim : int
        Number of output classes (2 for binary delay-risk).
    dropout : float
        Dropout probability.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 64,
        out_dim: int = 2,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, out_dim)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:  # noqa: ANN003
        """Forward pass.  Extra ``kwargs`` (``edge_index``, etc.) are
        accepted but ignored so the model has the same call signature as
        the GCN.
        """
        h = self.fc1(x)
        h = self.bn1(h)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)

        h = self.fc2(h)
        h = self.bn2(h)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)

        return self.fc3(h)

    def reset_parameters(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
