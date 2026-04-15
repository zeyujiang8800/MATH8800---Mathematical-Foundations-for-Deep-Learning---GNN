"""CLI entry-point: build data → train → evaluate → report."""

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

from marta_gnn.config import load_config
from marta_gnn.data.dataset_builder import DatasetBuilder
from marta_gnn.training.evaluation import evaluate
from marta_gnn.training.trainer import Trainer, build_model
from marta_gnn.visualization.plots import (
    plot_confusion_matrix,
    plot_roc_curve,
    plot_training_curves,
    plot_graph_layout,
)

logger = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="marta_gnn",
        description="MARTA GNN – transit delay risk prediction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
examples:
  python -m marta_gnn.main                  # full training on synthetic data
  python -m marta_gnn.main --demo           # quick 50-epoch demo
  python -m marta_gnn.main --model mlp      # train MLP baseline instead
  python -m marta_gnn.main --live           # use real MARTA feeds
""",
    )
    parser.add_argument("--config", type=str, default=None, help="path to config.yaml")
    parser.add_argument("--model", choices=["gcn", "mlp"], default=None, help="model type (overrides config)")
    parser.add_argument("--epochs", type=int, default=None, help="max training epochs (overrides config)")
    parser.add_argument("--seed", type=int, default=None, help="random seed (overrides config)")

    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--mock", action="store_true", default=False, help="use synthetic data (default)")
    mode.add_argument("--live", action="store_true", default=False, help="use real MARTA GTFS-rt feeds")

    parser.add_argument("--demo", action="store_true", default=False, help="quick demo: --mock --epochs 50")
    return parser.parse_args()


def main(config_path: str | None = None, **overrides: Any) -> None:
    cfg = load_config(config_path)

    # Apply CLI overrides
    if overrides.get("model"):
        cfg["model"]["type"] = overrides["model"]
    if overrides.get("epochs"):
        cfg["model"]["epochs"] = overrides["epochs"]
    if overrides.get("seed"):
        cfg["training"]["seed"] = overrides["seed"]
    if overrides.get("live"):
        cfg["data"]["use_mock"] = False
    if overrides.get("demo"):
        cfg["data"]["use_mock"] = True
        cfg["model"]["epochs"] = min(cfg["model"].get("epochs", 200), 50)
        cfg["model"]["patience"] = min(cfg["model"].get("patience", 20), 10)

    logger.info("Configuration loaded")

    # ---- data ----
    builder = DatasetBuilder(cfg)
    data = builder.build()
    logger.info(
        "Dataset ready — %d nodes, %d edges, feature dim=%d",
        data.num_nodes,
        data.edge_index.shape[1],
        data.x.shape[1],
    )

    # ---- model ----
    in_dim = data.x.shape[1]
    model = build_model(cfg, in_dim=in_dim)
    logger.info("Model: %s", model.__class__.__name__)

    # ---- train ----
    trainer = Trainer(cfg)
    history = trainer.train(model, data)

    # ---- evaluate ----
    results = evaluate(model, data)
    print("\n" + results["classification_report"])

    # ---- plots ----
    out_dir = Path(cfg.get("data", {}).get("processed_dir", "data/processed"))
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_training_curves(history, save_path=str(out_dir / "training_curves.png"))
    plot_confusion_matrix(
        results["y_true"], results["y_pred"],
        save_path=str(out_dir / "confusion_matrix.png"),
    )
    plot_roc_curve(
        results["y_true"], results["y_prob"],
        save_path=str(out_dir / "roc_curve.png"),
    )
    plot_graph_layout(data, results["y_pred"], save_path=str(out_dir / "stop_map.png"))

    logger.info("Done. Plots saved to %s", out_dir)


if __name__ == "__main__":
    args = _parse_args()
    main(
        config_path=args.config,
        model=args.model,
        epochs=args.epochs,
        seed=args.seed,
        live=args.live,
        demo=args.demo,
    )
