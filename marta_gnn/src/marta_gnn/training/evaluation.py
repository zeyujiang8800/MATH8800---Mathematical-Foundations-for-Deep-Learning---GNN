"""Model evaluation: accuracy, precision, recall, F1, confusion matrix, ROC, MAE/MSE/RMSE."""

import logging
import math
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch_geometric.data import Data

from .trainer import model_forward

logger = logging.getLogger(__name__)


def evaluate(
    model: nn.Module,
    data: Data,
    mask: torch.Tensor | None = None,
) -> dict[str, Any]:
    """Compute classification metrics on the nodes selected by *mask*.

    If *mask* is ``None``, ``data.test_mask`` is used.

    Returns a dict with keys: accuracy, precision, recall, f1,
    roc_auc, mae, mse, rmse, confusion_matrix, classification_report.
    """
    device = next(model.parameters()).device
    data = data.to(device)
    if mask is None:
        mask = data.test_mask

    model.eval()
    with torch.no_grad():
        logits = model_forward(model, data)
        probs = torch.softmax(logits, dim=1)

    preds = logits[mask].argmax(dim=1).cpu().numpy()
    y_true = data.y[mask].cpu().numpy()
    prob_pos = probs[mask, 1].cpu().numpy()

    acc = accuracy_score(y_true, preds)
    prec = precision_score(y_true, preds, zero_division=0)
    rec = recall_score(y_true, preds, zero_division=0)
    f1 = f1_score(y_true, preds, zero_division=0)
    f1_macro = f1_score(y_true, preds, average="macro", zero_division=0)

    mae = mean_absolute_error(y_true, prob_pos)
    mse = mean_squared_error(y_true, prob_pos)
    rmse = math.sqrt(mse)

    try:
        auc = roc_auc_score(y_true, prob_pos)
    except ValueError:
        auc = float("nan")  # only one class present in split

    cm = confusion_matrix(y_true, preds, labels=[0, 1])
    report = classification_report(y_true, preds, labels=[0, 1], target_names=["on-time", "at-risk"], zero_division=0)

    results = {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "f1_macro": f1_macro,
        "roc_auc": auc,
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "confusion_matrix": cm,
        "classification_report": report,
        "y_true": y_true,
        "y_pred": preds,
        "y_prob": prob_pos,
    }
    logger.info(
        "Test metrics — acc=%.3f  prec=%.3f  rec=%.3f  f1=%.3f  auc=%.3f  MAE=%.4f  MSE=%.4f  RMSE=%.4f",
        acc, prec, rec, f1, auc, mae, mse, rmse,
    )
    return results
