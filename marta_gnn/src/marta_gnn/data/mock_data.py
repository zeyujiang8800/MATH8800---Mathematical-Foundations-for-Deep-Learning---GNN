"""Generate synthetic MARTA-like GTFS data for testing and demo purposes."""

import logging
import random
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def generate_mock_data(cfg: dict[str, Any]) -> dict[str, pd.DataFrame]:
    """Return a dict mirroring the output of ``GTFSLoader.load_all()``,
    plus a ``"realtime"`` key containing synthetic delay observations.

    The synthetic graph is a plausible bus/rail network: stops placed on a
    grid, routes connecting chains of stops, trips scheduled across the day.
    """
    data_cfg = cfg.get("data", {})
    seed = cfg.get("training", {}).get("seed", 42)
    rng = np.random.default_rng(seed)
    random.seed(seed)

    num_stops = data_cfg.get("mock_num_stops", 200)
    num_routes = data_cfg.get("mock_num_routes", 10)
    num_trips = data_cfg.get("mock_num_trips", 50)

    stops = _make_stops(num_stops, rng)
    routes = _make_routes(num_routes)
    trips, stop_times = _make_trips_and_stop_times(
        routes, stops, num_trips, rng
    )
    realtime = _make_realtime(stop_times, cfg, rng)

    logger.info(
        "Generated mock data: %d stops, %d routes, %d trips, %d stop_times, %d RT updates",
        len(stops), len(routes), len(trips), len(stop_times), len(realtime),
    )

    return {
        "stops": stops,
        "routes": routes,
        "trips": trips,
        "stop_times": stop_times,
        "realtime": realtime,
    }


# ------------------------------------------------------------------
# Generators
# ------------------------------------------------------------------

def _make_stops(n: int, rng: np.random.Generator) -> pd.DataFrame:
    """Generate *n* stops scattered around Atlanta (lat ~33.75, lon ~-84.39)."""
    lats = rng.normal(33.75, 0.06, size=n)
    lons = rng.normal(-84.39, 0.06, size=n)
    return pd.DataFrame(
        {
            "stop_id": [f"STOP_{i:04d}" for i in range(n)],
            "stop_name": [f"Mock Stop {i}" for i in range(n)],
            "stop_lat": lats,
            "stop_lon": lons,
            "wheelchair_boarding": rng.choice([0, 1, 2], size=n).tolist(),
        }
    )


def _make_routes(n: int) -> pd.DataFrame:
    route_types = [3] * (n - 2) + [1, 1]  # mostly bus, a couple of rail
    return pd.DataFrame(
        {
            "route_id": [f"ROUTE_{i:03d}" for i in range(n)],
            "route_short_name": [f"R{i}" for i in range(n)],
            "route_long_name": [f"Mock Route {i}" for i in range(n)],
            "route_type": route_types[:n],
        }
    )


def _make_trips_and_stop_times(
    routes: pd.DataFrame,
    stops: pd.DataFrame,
    num_trips: int,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    stop_ids = stops["stop_id"].tolist()
    route_ids = routes["route_id"].tolist()

    trip_rows: list[dict[str, Any]] = []
    st_rows: list[dict[str, Any]] = []

    for t in range(num_trips):
        route_id = route_ids[t % len(route_ids)]
        trip_id = f"TRIP_{t:04d}"
        direction = int(rng.integers(0, 2))
        trip_rows.append(
            {
                "trip_id": trip_id,
                "route_id": route_id,
                "direction_id": direction,
                "service_id": "WK",
            }
        )

        # Each trip visits 8-20 stops in order
        n_stops_in_trip = int(rng.integers(8, 21))
        selected = rng.choice(len(stop_ids), size=n_stops_in_trip, replace=False)
        selected.sort()

        start_sec = int(rng.integers(5 * 3600, 23 * 3600))  # 05:00–23:00
        for seq, idx in enumerate(selected):
            arr = start_sec + seq * int(rng.integers(60, 300))
            dep = arr + int(rng.integers(10, 60))
            st_rows.append(
                {
                    "trip_id": trip_id,
                    "stop_id": stop_ids[idx],
                    "stop_sequence": seq,
                    "arrival_time": arr,
                    "departure_time": dep,
                }
            )

    trips = pd.DataFrame(trip_rows)
    stop_times = pd.DataFrame(st_rows)
    return trips, stop_times


def _make_realtime(
    stop_times: pd.DataFrame,
    cfg: dict[str, Any],
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Simulate realtime delay observations for every stop-time."""
    threshold = cfg.get("features", {}).get("delay_threshold_seconds", 300)

    n = len(stop_times)
    # Most arrivals on time, ~40% delayed (ensures ~25-30% of stops at-risk)
    delays = rng.exponential(scale=60, size=n)
    # Randomly flip some to larger delays
    big_mask = rng.random(n) < 0.4
    delays[big_mask] = rng.exponential(scale=threshold * 1.5, size=big_mask.sum())
    delays = delays.astype(int)

    rt = stop_times[["trip_id", "stop_id", "stop_sequence"]].copy()
    rt["arrival_delay"] = delays
    rt["departure_delay"] = (delays + rng.integers(-10, 30, size=n)).clip(min=0)
    rt["timestamp"] = rng.integers(1_700_000_000, 1_700_100_000, size=n)
    # Add a route_id by merging back to trip
    return rt
