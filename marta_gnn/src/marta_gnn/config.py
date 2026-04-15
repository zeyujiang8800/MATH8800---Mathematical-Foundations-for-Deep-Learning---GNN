"""Configuration loader for MARTA GNN project."""

import logging
import os
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

_DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent.parent / "config.yaml"


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load YAML configuration from *path*, falling back to the repo default.

    Environment variable overrides:
        MARTA_API_KEY  -> realtime.api_key
        MARTA_USE_MOCK -> data.use_mock  (``"true"`` / ``"false"``)
    """
    path = Path(path) if path else _DEFAULT_CONFIG_PATH
    if not path.exists():
        logger.warning("Config file %s not found – using built-in defaults", path)
        return _defaults()

    with open(path, "r", encoding="utf-8") as fh:
        cfg: dict[str, Any] = yaml.safe_load(fh) or {}

    # Environment overrides
    api_key = os.environ.get("MARTA_API_KEY")
    if api_key:
        cfg.setdefault("realtime", {})["api_key"] = api_key

    use_mock_env = os.environ.get("MARTA_USE_MOCK")
    if use_mock_env is not None:
        cfg.setdefault("data", {})["use_mock"] = use_mock_env.lower() == "true"

    _apply_defaults(cfg)
    _setup_logging(cfg)
    return cfg


def _defaults() -> dict[str, Any]:
    return {
        "gtfs": {
            "static_url": "https://itsmarta.com/google_transit_feed/google_transit.zip",
            "static_dir": "data/gtfs_static",
        },
        "realtime": {
            "trip_updates_url": "",
            "vehicle_positions_url": "",
            "api_key": "",
        },
        "data": {
            "processed_dir": "data/processed",
            "use_mock": True,
            "mock_num_stops": 200,
            "mock_num_routes": 10,
            "mock_num_trips": 50,
        },
        "features": {
            "time_bins": 24,
            "delay_threshold_seconds": 300,
            "historical_window_days": 7,
        },
        "model": {
            "type": "gcn",
            "hidden_dim": 64,
            "num_layers": 3,
            "dropout": 0.3,
            "learning_rate": 0.001,
            "weight_decay": 0.0005,
            "epochs": 200,
            "patience": 20,
            "batch_size": 32,
        },
        "training": {
            "train_ratio": 0.6,
            "val_ratio": 0.2,
            "test_ratio": 0.2,
            "seed": 42,
        },
        "logging": {
            "level": "INFO",
            "file": "logs/marta_gnn.log",
        },
    }


def _apply_defaults(cfg: dict[str, Any]) -> None:
    defaults = _defaults()
    for section, values in defaults.items():
        if section not in cfg:
            cfg[section] = values
        elif isinstance(values, dict):
            for k, v in values.items():
                cfg[section].setdefault(k, v)


def _setup_logging(cfg: dict[str, Any]) -> None:
    log_cfg = cfg.get("logging", {})
    level = getattr(logging, log_cfg.get("level", "INFO").upper(), logging.INFO)
    log_file = log_cfg.get("file")

    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_path))

    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        handlers=handlers,
        force=True,
    )
