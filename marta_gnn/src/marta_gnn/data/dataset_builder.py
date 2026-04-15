"""End-to-end dataset builder: GTFS → graph → features → labels → splits."""

import logging
from typing import Any

import numpy as np
import torch
from torch_geometric.data import Data

from .feature_engineering import FeatureEngineer
from .graph_builder import GraphBuilder
from .label_generation import LabelGenerator
from .mock_data import generate_mock_data

logger = logging.getLogger(__name__)


class DatasetBuilder:
    """Orchestrate the full data pipeline."""

    def __init__(self, cfg: dict[str, Any]) -> None:
        self.cfg = cfg
        self.graph_builder = GraphBuilder(cfg)
        self.feature_eng = FeatureEngineer(cfg)
        self.label_gen = LabelGenerator(cfg)

    def build(
        self,
        tables: dict | None = None,
        realtime: "pd.DataFrame | None" = None,
    ) -> Data:
        """Build a single ``Data`` object ready for training.

        If *tables* is ``None`` and ``data.use_mock`` is true in the
        config, synthetic data is generated automatically.
        """
        import pandas as pd  # local to avoid circular at module level

        use_mock: bool = self.cfg.get("data", {}).get("use_mock", True)

        if tables is None:
            if use_mock:
                logger.info("Using mock data (set data.use_mock=false for live feeds)")
                mock = generate_mock_data(self.cfg)
                tables = {
                    "stops": mock["stops"],
                    "routes": mock["routes"],
                    "trips": mock["trips"],
                    "stop_times": mock["stop_times"],
                }
                if realtime is None:
                    realtime = mock.get("realtime")
            else:
                raise ValueError(
                    "No tables provided and use_mock is false. "
                    "Pass tables from GTFSLoader.load_all()."
                )

        data = self.graph_builder.build(tables)
        data = self.feature_eng.enrich(data, tables, realtime)
        data = self.label_gen.generate(data, realtime)

        data = self._add_splits(data)
        return data

    # ------------------------------------------------------------------
    # Train / val / test masks
    # ------------------------------------------------------------------

    def _add_splits(self, data: Data) -> Data:
        """Attach boolean masks ``train_mask``, ``val_mask``, ``test_mask``.

        Uses stratified splitting so each fold has proportional class
        representation — critical for imbalanced datasets.
        """
        t_cfg = self.cfg.get("training", {})
        seed = t_cfg.get("seed", 42)
        train_r = t_cfg.get("train_ratio", 0.6)
        val_r = t_cfg.get("val_ratio", 0.2)

        n = data.num_nodes
        labels = data.y.numpy() if hasattr(data, "y") and data.y is not None else np.zeros(n)
        rng = np.random.default_rng(seed)

        train_mask = torch.zeros(n, dtype=torch.bool)
        val_mask = torch.zeros(n, dtype=torch.bool)
        test_mask = torch.zeros(n, dtype=torch.bool)

        # Stratify by class label
        for cls in np.unique(labels):
            cls_idx = np.where(labels == cls)[0]
            rng.shuffle(cls_idx)
            cls_n = len(cls_idx)
            t_end = max(1, int(cls_n * train_r))
            v_end = t_end + max(1, int(cls_n * val_r))
            train_mask[cls_idx[:t_end]] = True
            val_mask[cls_idx[t_end:v_end]] = True
            test_mask[cls_idx[v_end:]] = True

        data.train_mask = train_mask
        data.val_mask = val_mask
        data.test_mask = test_mask

        logger.info(
            "Splits: train=%d, val=%d, test=%d",
            train_mask.sum().item(),
            val_mask.sum().item(),
            test_mask.sum().item(),
        )
        return data
