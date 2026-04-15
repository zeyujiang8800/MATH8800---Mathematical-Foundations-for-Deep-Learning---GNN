"""Generate binary delay-risk labels for each stop."""

import logging
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data

logger = logging.getLogger(__name__)


class LabelGenerator:
    """Attach ``data.y`` labels (0 = on-time, 1 = at-risk) to graph nodes."""

    def __init__(self, cfg: dict[str, Any]) -> None:
        self.threshold: int = cfg.get("features", {}).get(
            "delay_threshold_seconds", 300
        )

    def generate(
        self,
        data: Data,
        realtime: pd.DataFrame | None = None,
    ) -> Data:
        """Label each stop based on realtime delay observations.

        A stop is labeled *1* (at-risk) if its **median** arrival delay
        exceeds ``delay_threshold_seconds``.

        When *realtime* is ``None`` (pure static mode), labels are set to
        all zeros so the pipeline can still run.
        """
        stop2idx: dict[str, int] = data.stop2idx  # type: ignore[attr-defined]
        num_nodes = data.num_nodes

        labels = np.zeros(num_nodes, dtype=np.int64)

        if realtime is not None and len(realtime) > 0:
            for sid, idx in stop2idx.items():
                sub = realtime[realtime["stop_id"] == sid]
                if sub.empty:
                    continue
                median_delay = sub["arrival_delay"].astype(float).median()
                if median_delay > self.threshold:
                    labels[idx] = 1

        data.y = torch.tensor(labels, dtype=torch.long)
        n_pos = int(labels.sum())
        logger.info(
            "Labels: %d at-risk (%.1f%%), %d on-time",
            n_pos,
            100.0 * n_pos / max(num_nodes, 1),
            num_nodes - n_pos,
        )
        return data
