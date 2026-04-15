"""Feature engineering: enrich the stop-level graph with realtime and static features."""

import logging
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Attach node-level features derived from GTFS static + realtime data."""

    def __init__(self, cfg: dict[str, Any]) -> None:
        self.time_bins: int = cfg.get("features", {}).get("time_bins", 24)

    def enrich(
        self,
        data: Data,
        tables: dict[str, pd.DataFrame],
        realtime: pd.DataFrame | None = None,
    ) -> Data:
        """Add features to *data.x* in-place and return it.

        Feature vector per stop (appended to existing lat/lon):
            0  degree           – number of edges incident to the stop
            1  n_routes         – distinct routes serving the stop
            2  n_trips          – total trips visiting the stop per day
            3  avg_headway      – mean headway (sec) between trips at the stop
            4  mean_delay       – mean observed arrival delay (from RT)
            5  std_delay        – std-dev of arrival delay
            6  max_delay        – max observed delay
            7  frac_delayed     – fraction of observations exceeding threshold
            8  time_bin_sin     – sin-encoded median departure hour
            9  time_bin_cos     – cos-encoded median departure hour
        """
        stop_ids: list[str] = data.stop_ids  # type: ignore[attr-defined]
        stop2idx: dict[str, int] = data.stop2idx  # type: ignore[attr-defined]
        num_nodes = data.num_nodes

        stop_times = tables["stop_times"]
        trips = tables.get("trips", pd.DataFrame())

        # --- degree (vectorised) ---
        degree = np.bincount(data.edge_index[1].numpy(), minlength=num_nodes).astype(np.float32)

        # --- route/trip counts & headway per stop (vectorised groupby) ---
        n_routes = np.zeros(num_nodes, dtype=np.float32)
        n_trips_feat = np.zeros(num_nodes, dtype=np.float32)
        avg_headway = np.zeros(num_nodes, dtype=np.float32)
        median_hour = np.full(num_nodes, 12.0, dtype=np.float32)

        st = stop_times.copy()
        if "route_id" not in st.columns and len(trips) > 0:
            st = st.merge(trips[["trip_id", "route_id"]], on="trip_id", how="left")

        # Map stop_id → node index for the whole frame at once
        st["_idx"] = st["stop_id"].map(stop2idx)
        st = st.dropna(subset=["_idx"])
        st["_idx"] = st["_idx"].astype(int)

        grp = st.groupby("_idx")
        n_trips_feat[grp.ngroup().unique()] = 0  # reset
        counts = grp.size()
        n_trips_feat[counts.index] = counts.values.astype(np.float32)

        if "route_id" in st.columns:
            rc = grp["route_id"].nunique()
            n_routes[rc.index] = rc.values.astype(np.float32)

        # Headway: mean diff of sorted arrival times per stop
        def _headway(arr):
            s = arr.dropna().sort_values()
            return s.diff().mean() if len(s) > 1 else 0.0
        hw = grp["arrival_time"].apply(_headway)
        avg_headway[hw.index] = hw.values.astype(np.float32)

        # Median departure hour
        med_dep = grp["departure_time"].median()
        valid = med_dep.dropna()
        median_hour[valid.index] = (valid.values / 3600.0).astype(np.float32) % 24

        time_sin = np.sin(2 * np.pi * median_hour / 24)
        time_cos = np.cos(2 * np.pi * median_hour / 24)

        # --- realtime delay stats (vectorised groupby) ---
        mean_delay = np.zeros(num_nodes, dtype=np.float32)
        std_delay = np.zeros(num_nodes, dtype=np.float32)
        max_delay = np.zeros(num_nodes, dtype=np.float32)
        frac_delayed = np.zeros(num_nodes, dtype=np.float32)

        if realtime is not None and len(realtime) > 0:
            threshold = 300
            rt = realtime.copy()
            rt["_idx"] = rt["stop_id"].map(stop2idx)
            rt = rt.dropna(subset=["_idx"])
            rt["_idx"] = rt["_idx"].astype(int)
            rt["arrival_delay"] = rt["arrival_delay"].astype(float)

            rg = rt.groupby("_idx")["arrival_delay"]
            mean_delay[rg.mean().index] = rg.mean().values.astype(np.float32)
            std_delay[rg.std().index] = rg.std().fillna(0).values.astype(np.float32)
            max_delay[rg.max().index] = rg.max().values.astype(np.float32)
            fd = rg.apply(lambda x: (x > threshold).mean())
            frac_delayed[fd.index] = fd.values.astype(np.float32)

        # --- assemble ---
        new_feats = np.stack(
            [
                degree, n_routes, n_trips_feat, avg_headway,
                mean_delay, std_delay, max_delay, frac_delayed,
                time_sin, time_cos,
            ],
            axis=1,
        )  # (N, 10)

        new_feats_t = torch.tensor(new_feats, dtype=torch.float)
        data.x = torch.cat([data.x, new_feats_t], dim=1)
        logger.info("Node feature dim after enrichment: %d", data.x.shape[1])
        return data
