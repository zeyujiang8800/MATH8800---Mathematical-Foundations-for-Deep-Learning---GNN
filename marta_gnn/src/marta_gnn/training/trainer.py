"""Training loop with early stopping for node-classification models."""

import copy
import logging
import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data

from ..models.gcn_model import GCNModel
from ..models.mlp_baseline import MLPBaseline

logger = logging.getLogger(__name__)


def model_forward(model: nn.Module, data: Data) -> torch.Tensor:
    """Unified forward that works for both MLP and GCN."""
    if isinstance(model, GCNModel):
        edge_weight = data.edge_attr[:, 0] if data.edge_attr is not None and data.edge_attr.numel() > 0 else None
        return model(data.x, data.edge_index, edge_weight=edge_weight)
    return model(data.x)


def build_model(cfg: dict[str, Any], in_dim: int, out_dim: int = 2) -> nn.Module:
    """Instantiate a model based on ``cfg["model"]["type"]``."""
    mcfg = cfg.get("model", {})
    model_type = mcfg.get("type", "gcn").lower()
    hidden = mcfg.get("hidden_dim", 64)
    dropout = mcfg.get("dropout", 0.3)

    if model_type == "mlp":
        return MLPBaseline(in_dim, hidden, out_dim, dropout)
    elif model_type == "gcn":
        num_layers = mcfg.get("num_layers", 3)
        return GCNModel(in_dim, hidden, out_dim, num_layers, dropout)
    else:
        raise ValueError(f"Unknown model type: {model_type!r}")


def _compute_regression_metrics(
    probs: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor,
) -> dict[str, float]:
    """Compute MAE, MSE, RMSE between predicted probability and true label."""
    prob_pos = probs[mask, 1]
    y = labels[mask].float()
    diff = prob_pos - y
    mae = diff.abs().mean().item()
    mse = (diff ** 2).mean().item()
    rmse = math.sqrt(mse)
    return {"mae": mae, "mse": mse, "rmse": rmse}


class Trainer:
    """Train and evaluate a node-classification model with early stopping."""

    def __init__(self, cfg: dict[str, Any]) -> None:
        self.cfg = cfg
        mcfg = cfg.get("model", {})
        self.lr: float = mcfg.get("learning_rate", 1e-3)
        self.wd: float = mcfg.get("weight_decay", 5e-4)
        self.epochs: int = mcfg.get("epochs", 200)
        self.patience: int = mcfg.get("patience", 20)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def train(self, model: nn.Module, data: Data) -> dict[str, list[float]]:
        """Run the full training loop.

        Returns a history dict with keys: train_loss, val_loss,
        train_acc, val_acc, train_mae, val_mae, train_mse, val_mse,
        train_rmse, val_rmse.
        """
        model = model.to(self.device)
        data = data.to(self.device)

        # Handle class imbalance with weighted loss
        class_counts = torch.bincount(data.y[data.train_mask], minlength=2).float()
        weight = (class_counts.sum() / (2 * class_counts.clamp(min=1))).to(self.device)
        criterion = nn.CrossEntropyLoss(weight=weight)

        optimizer = torch.optim.Adam(
            model.parameters(), lr=self.lr, weight_decay=self.wd
        )

        history: dict[str, list[float]] = {
            "train_loss": [], "val_loss": [],
            "train_acc": [], "val_acc": [],
            "train_mae": [], "val_mae": [],
            "train_mse": [], "val_mse": [],
            "train_rmse": [], "val_rmse": [],
        }

        best_val_loss = float("inf")
        best_state = None
        patience_counter = 0

        for epoch in range(1, self.epochs + 1):
            # --- train ---
            model.train()
            optimizer.zero_grad()
            out = model_forward(model, data)
            loss = criterion(out[data.train_mask], data.y[data.train_mask])
            loss.backward()
            optimizer.step()

            # --- metrics (no grad) ---
            model.eval()
            with torch.no_grad():
                out = model_forward(model, data)
                probs = torch.softmax(out, dim=1)
                val_loss = criterion(out[data.val_mask], data.y[data.val_mask]).item()

                train_acc = self._accuracy(out, data.y, data.train_mask)
                val_acc = self._accuracy(out, data.y, data.val_mask)
                train_reg = _compute_regression_metrics(probs, data.y, data.train_mask)
                val_reg = _compute_regression_metrics(probs, data.y, data.val_mask)

            history["train_loss"].append(loss.item())
            history["val_loss"].append(val_loss)
            history["train_acc"].append(train_acc)
            history["val_acc"].append(val_acc)
            for k in ("mae", "mse", "rmse"):
                history[f"train_{k}"].append(train_reg[k])
                history[f"val_{k}"].append(val_reg[k])

            logger.info(
                "Epoch %3d | loss=%.4f val_loss=%.4f | acc=%.3f val_acc=%.3f "
                "| MAE=%.4f MSE=%.4f RMSE=%.4f",
                epoch, loss.item(), val_loss, train_acc, val_acc,
                val_reg["mae"], val_reg["mse"], val_reg["rmse"],
            )

            # --- early stopping ---
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = copy.deepcopy(model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    logger.info("Early stopping at epoch %d", epoch)
                    break

        if best_state is not None:
            model.load_state_dict(best_state)
        return history

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _accuracy(
        logits: torch.Tensor,
        labels: torch.Tensor,
        mask: torch.Tensor,
    ) -> float:
        preds = logits[mask].argmax(dim=1)
        correct = (preds == labels[mask]).sum().item()
        total = mask.sum().item()
        return correct / max(total, 1)
