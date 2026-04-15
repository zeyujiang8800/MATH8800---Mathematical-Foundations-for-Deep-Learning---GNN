"""Plotting utilities for training curves, confusion matrix, graph layout, ROC."""

import logging
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay

logger = logging.getLogger(__name__)

# Consistent style for all plots
_STYLE_DEFAULTS = dict(
    figure_facecolor="white",
    axes_grid=True,
    grid_alpha=0.25,
)

def _apply_style() -> None:
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.grid": True,
        "grid.alpha": 0.25,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.size": 10,
    })

_apply_style()


def plot_training_curves(history: dict[str, list[float]], save_path: str | None = None) -> plt.Figure:
    """Plot loss, accuracy, MAE, MSE, RMSE curves for train / val."""
    panels = [
        ("train_loss", "val_loss", "Loss"),
        ("train_acc", "val_acc", "Accuracy"),
    ]
    # Include regression metrics if present in history
    for key, label in [("mae", "MAE"), ("mse", "MSE"), ("rmse", "RMSE")]:
        if f"train_{key}" in history:
            panels.append((f"train_{key}", f"val_{key}", label))

    n = len(panels)
    cols = min(n, 3)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows), squeeze=False)

    for idx, (train_key, val_key, ylabel) in enumerate(panels):
        ax = axes[idx // cols][idx % cols]
        epochs = range(1, len(history[train_key]) + 1)
        ax.plot(epochs, history[train_key], label="train", linewidth=1.5)
        ax.plot(epochs, history[val_key], label="val", linewidth=1.5)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{ylabel} Curves")
        ax.legend()
        best_idx = int(np.argmin(history[val_key]) if ylabel != "Accuracy" else np.argmax(history[val_key]))
        ax.axvline(best_idx + 1, color="grey", linestyle=":", alpha=0.5)

    # Hide unused subplots
    for idx in range(n, rows * cols):
        axes[idx // cols][idx % cols].set_visible(False)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Saved training curves to %s", save_path)
    return fig


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: list[str] | None = None,
    ax: plt.Axes | None = None,
    save_path: str | None = None,
) -> plt.Figure:
    """Plot a confusion matrix heatmap."""
    labels = labels or ["on-time", "at-risk"]
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 4))
    else:
        fig = ax.get_figure()
    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, display_labels=labels, ax=ax, cmap="Blues"
    )
    ax.set_title("Confusion Matrix")
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_roc_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    save_path: str | None = None,
) -> plt.Figure:
    """Plot the ROC curve for the positive (at-risk) class."""
    fig, ax = plt.subplots(figsize=(5, 4))
    RocCurveDisplay.from_predictions(y_true, y_prob, ax=ax, name="GCN")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3, label="chance")
    ax.set_title("ROC Curve – At-Risk Class")
    ax.legend()
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_graph_layout(
    data: Any,
    predictions: np.ndarray | None = None,
    show_edges: bool = True,
    save_path: str | None = None,
) -> plt.Figure:
    """Scatter-plot stops colored by predicted delay risk, with edges.

    Uses latitude / longitude (first two features in ``data.x``).
    """
    import torch
    x_np = data.x.cpu().numpy() if isinstance(data.x, torch.Tensor) else np.array(data.x)
    lats = x_np[:, 0]
    lons = x_np[:, 1]

    if predictions is None:
        colors = data.y.cpu().numpy() if hasattr(data, "y") and data.y is not None else np.zeros(len(lats))
    else:
        colors = np.asarray(predictions)

    fig, ax = plt.subplots(figsize=(8, 7))

    # Draw edges as light lines
    if show_edges and hasattr(data, "edge_index") and data.edge_index.numel() > 0:
        ei = data.edge_index.cpu().numpy()
        # Subsample edges if too many (for readability)
        n_edges = ei.shape[1]
        max_draw = 2000
        if n_edges > max_draw:
            idx = np.random.default_rng(0).choice(n_edges, max_draw, replace=False)
            ei = ei[:, idx]
        for i in range(ei.shape[1]):
            s, d = ei[0, i], ei[1, i]
            ax.plot([lons[s], lons[d]], [lats[s], lats[d]], color="lightgrey", linewidth=0.3, zorder=1)

    # Use discrete colormap for binary predictions
    unique_vals = np.unique(colors)
    is_binary = set(unique_vals).issubset({0, 1, 0.0, 1.0})
    if is_binary:
        from matplotlib.colors import ListedColormap
        cmap = ListedColormap(["#2ca02c", "#d62728"])  # green=on-time, red=at-risk
        scatter = ax.scatter(
            lons, lats, c=colors, cmap=cmap, vmin=0, vmax=1, s=24, alpha=0.85,
            edgecolors="k", linewidths=0.3, zorder=2,
        )
        cbar = fig.colorbar(scatter, ax=ax, label="Delay Risk", shrink=0.8, ticks=[0, 1])
        cbar.ax.set_yticklabels(["On-Time", "At-Risk"])
    else:
        scatter = ax.scatter(
            lons, lats, c=colors, cmap="RdYlGn_r", s=24, alpha=0.85,
            edgecolors="k", linewidths=0.3, zorder=2,
        )
        fig.colorbar(scatter, ax=ax, label="Delay Risk", shrink=0.8)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("MARTA Stop Delay Risk Map")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_delay_distribution(
    delays: np.ndarray,
    threshold: int = 300,
    save_path: str | None = None,
) -> plt.Figure:
    """Histogram of delay values with the risk threshold marked."""
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(delays, bins=50, color="steelblue", edgecolor="white", alpha=0.85)
    ax.axvline(threshold, color="red", linestyle="--", linewidth=1.5, label=f"Threshold ({threshold}s)")
    pct_delayed = 100.0 * np.mean(delays > threshold)
    ax.set_xlabel("Arrival Delay (seconds)")
    ax.set_ylabel("Count")
    ax.set_title(f"Distribution of Arrival Delays ({pct_delayed:.1f}% exceed threshold)")
    ax.legend()
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_model_comparison(results: dict[str, dict[str, float]], save_path: str | None = None) -> plt.Figure:
    """Bar chart comparing metrics across models.

    Parameters
    ----------
    results : dict
        ``{model_name: {metric_name: value, ...}, ...}``
    """
    class_metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    error_metrics = ["mae", "mse", "rmse"]
    has_error = any(m in results[next(iter(results))] for m in error_metrics)

    model_names = list(results.keys())

    if has_error:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))
    else:
        fig, ax1 = plt.subplots(figsize=(8, 4))

    # Classification metrics
    x = np.arange(len(class_metrics))
    width = 0.8 / len(model_names)
    for i, name in enumerate(model_names):
        vals = [results[name].get(m, 0) for m in class_metrics]
        ax1.bar(x + i * width, vals, width, label=name)
    ax1.set_xticks(x + width * (len(model_names) - 1) / 2)
    ax1.set_xticklabels([m.replace("_", " ").title() for m in class_metrics])
    ax1.set_ylim(0, 1.05)
    ax1.set_ylabel("Score")
    ax1.set_title("Classification Metrics")
    ax1.legend()

    # Error metrics
    if has_error:
        x2 = np.arange(len(error_metrics))
        width2 = 0.8 / len(model_names)
        for i, name in enumerate(model_names):
            vals = [results[name].get(m, 0) for m in error_metrics]
            ax2.bar(x2 + i * width2, vals, width2, label=name)
        ax2.set_xticks(x2 + width2 * (len(model_names) - 1) / 2)
        ax2.set_xticklabels([m.upper() for m in error_metrics])
        ax2.set_ylabel("Error")
        ax2.set_title("Regression Metrics (lower is better)")
        ax2.legend()

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig
