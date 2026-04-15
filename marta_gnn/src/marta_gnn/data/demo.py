"""Pre-built demo dataset that works without any config or external data."""

from typing import Any

import torch
from torch_geometric.data import Data

from .dataset_builder import DatasetBuilder
from .mock_data import generate_mock_data


_DEMO_CFG: dict[str, Any] = {
    "data": {"use_mock": True, "mock_num_stops": 200, "mock_num_routes": 10, "mock_num_trips": 50, "processed_dir": "data/processed"},
    "features": {"time_bins": 24, "delay_threshold_seconds": 300, "historical_window_days": 7},
    "model": {"type": "gcn", "hidden_dim": 64, "num_layers": 3, "dropout": 0.3, "learning_rate": 0.001, "weight_decay": 0.0005, "epochs": 50, "patience": 10, "batch_size": 32},
    "training": {"train_ratio": 0.6, "val_ratio": 0.2, "test_ratio": 0.2, "seed": 42},
    "logging": {"level": "WARNING", "file": ""},
}


def load_demo_dataset(cfg: dict[str, Any] | None = None) -> tuple[Data, dict]:
    """Return ``(data, mock_tables)`` ready for training – zero config needed.

    Useful for notebooks and quick experiments::

        from marta_gnn.data.demo import load_demo_dataset
        data, tables = load_demo_dataset()
    """
    if cfg is None:
        cfg = _DEMO_CFG.copy()
        # Deep-copy mutable sections
        cfg = {k: (dict(v) if isinstance(v, dict) else v) for k, v in _DEMO_CFG.items()}

    mock = generate_mock_data(cfg)
    builder = DatasetBuilder(cfg)
    tables = {
        "stops": mock["stops"],
        "routes": mock["routes"],
        "trips": mock["trips"],
        "stop_times": mock["stop_times"],
    }
    data = builder.build(tables=tables, realtime=mock.get("realtime"))
    return data, mock
