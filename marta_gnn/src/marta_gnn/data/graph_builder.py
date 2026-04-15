"""Build a PyG-compatible graph from GTFS static + realtime data."""

import logging
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data

logger = logging.getLogger(__name__)


class GraphBuilder:
    """Construct a heterogeneous stop-level graph.

    Nodes  = stops
    Edges  = (stop_i → stop_j) if they appear consecutively on any trip
    """

    def __init__(self, cfg: dict[str, Any]) -> None:
        self.cfg = cfg

    def build(self, tables: dict[str, pd.DataFrame]) -> Data:
        """Return a ``torch_geometric.data.Data`` object.

        Parameters
        ----------
        tables : dict
            Must contain ``"stops"`` and ``"stop_times"``; may contain
            ``"trips"`` and ``"routes"``.
        """
        stops = tables["stops"].copy()
        stop_times = tables["stop_times"].copy()

        # --- node mapping ---
        stop_ids = stops["stop_id"].unique().tolist()
        stop2idx: dict[str, int] = {sid: i for i, sid in enumerate(stop_ids)}
        num_nodes = len(stop2idx)
        logger.info("Graph has %d nodes (stops)", num_nodes)

        # --- edges: consecutive stops on same trip ---
        edge_src: list[int] = []
        edge_dst: list[int] = []
        edge_attr_list: list[list[float]] = []

        stop_times = stop_times.sort_values(["trip_id", "stop_sequence"])
        for trip_id, group in stop_times.groupby("trip_id"):
            sids = group["stop_id"].tolist()
            arr_times = group["arrival_time"].tolist()
            for i in range(len(sids) - 1):
                src_idx = stop2idx.get(sids[i])
                dst_idx = stop2idx.get(sids[i + 1])
                if src_idx is None or dst_idx is None:
                    continue
                # bidirectional
                edge_src.extend([src_idx, dst_idx])
                edge_dst.extend([dst_idx, src_idx])
                # edge weight = travel time between consecutive stops (seconds)
                t0 = arr_times[i] if arr_times[i] is not None else 0
                t1 = arr_times[i + 1] if arr_times[i + 1] is not None else 0
                travel = max(float(t1) - float(t0), 0.0)
                edge_attr_list.extend([[travel], [travel]])

        edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long)
        edge_attr = torch.tensor(edge_attr_list, dtype=torch.float) if edge_attr_list else torch.zeros((0, 1))
        logger.info("Graph has %d edges", edge_index.shape[1])

        # --- basic node features (lat, lon) ---
        stops_indexed = stops.set_index("stop_id")
        x_list: list[list[float]] = []
        for sid in stop_ids:
            row = stops_indexed.loc[sid]
            lat = float(row.get("stop_lat", 0) or 0)
            lon = float(row.get("stop_lon", 0) or 0)
            x_list.append([lat, lon])
        x = torch.tensor(x_list, dtype=torch.float)

        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        data.num_nodes = num_nodes
        data.stop_ids = stop_ids
        data.stop2idx = stop2idx
        return data


def deduplicate_edges(edge_index: torch.Tensor) -> torch.Tensor:
    """Remove duplicate edges from a ``[2, E]`` tensor."""
    pairs = set()
    keep: list[int] = []
    for i in range(edge_index.shape[1]):
        pair = (edge_index[0, i].item(), edge_index[1, i].item())
        if pair not in pairs:
            pairs.add(pair)
            keep.append(i)
    return edge_index[:, keep]
