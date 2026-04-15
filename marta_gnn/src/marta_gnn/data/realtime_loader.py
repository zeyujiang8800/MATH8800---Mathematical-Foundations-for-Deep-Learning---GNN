"""Load MARTA GTFS-realtime feeds (trip updates & vehicle positions)."""

import logging
import time
from typing import Any

import pandas as pd
import requests

logger = logging.getLogger(__name__)

# Lazy import so the rest of the project still works if protobuf isn't installed.
_gtfs_rt = None


def _ensure_gtfs_rt() -> Any:
    global _gtfs_rt
    if _gtfs_rt is None:
        from google.transit import gtfs_realtime_pb2  # type: ignore[import-untyped]

        _gtfs_rt = gtfs_realtime_pb2
    return _gtfs_rt


class RealtimeLoader:
    """Fetch and parse MARTA GTFS-realtime protobuf feeds."""

    def __init__(self, cfg: dict[str, Any]) -> None:
        rt = cfg.get("realtime", {})
        self.trip_updates_url: str = rt.get("trip_updates_url", "")
        self.vehicle_positions_url: str = rt.get("vehicle_positions_url", "")
        self.api_key: str = rt.get("api_key", "")
        self._timeout = 30

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fetch_trip_updates(self) -> pd.DataFrame:
        """Return a DataFrame with one row per stop-time update.

        Columns: trip_id, route_id, stop_id, stop_sequence,
                 arrival_delay, departure_delay, timestamp
        """
        pb2 = _ensure_gtfs_rt()
        feed = self._fetch_feed(self.trip_updates_url)
        rows: list[dict[str, Any]] = []
        for entity in feed.entity:
            if not entity.HasField("trip_update"):
                continue
            tu = entity.trip_update
            trip_id = tu.trip.trip_id
            route_id = tu.trip.route_id
            ts = tu.timestamp or int(time.time())
            for stu in tu.stop_time_update:
                rows.append(
                    {
                        "trip_id": trip_id,
                        "route_id": route_id,
                        "stop_id": stu.stop_id,
                        "stop_sequence": stu.stop_sequence,
                        "arrival_delay": stu.arrival.delay if stu.HasField("arrival") else 0,
                        "departure_delay": stu.departure.delay if stu.HasField("departure") else 0,
                        "timestamp": ts,
                    }
                )
        df = pd.DataFrame(rows)
        logger.info("Fetched %d stop-time updates from realtime feed", len(df))
        return df

    def fetch_vehicle_positions(self) -> pd.DataFrame:
        """Return a DataFrame of current vehicle positions.

        Columns: vehicle_id, trip_id, route_id, lat, lon, speed,
                 bearing, timestamp
        """
        feed = self._fetch_feed(self.vehicle_positions_url)
        rows: list[dict[str, Any]] = []
        for entity in feed.entity:
            if not entity.HasField("vehicle"):
                continue
            v = entity.vehicle
            rows.append(
                {
                    "vehicle_id": v.vehicle.id,
                    "trip_id": v.trip.trip_id,
                    "route_id": v.trip.route_id,
                    "lat": v.position.latitude,
                    "lon": v.position.longitude,
                    "speed": v.position.speed if v.position.speed else 0.0,
                    "bearing": v.position.bearing if v.position.bearing else 0.0,
                    "timestamp": v.timestamp or int(time.time()),
                }
            )
        df = pd.DataFrame(rows)
        logger.info("Fetched %d vehicle positions", len(df))
        return df

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _fetch_feed(self, url: str) -> Any:
        pb2 = _ensure_gtfs_rt()
        headers: dict[str, str] = {}
        if self.api_key:
            headers["api_key"] = self.api_key
        logger.debug("Fetching realtime feed from %s", url)
        resp = requests.get(url, headers=headers, timeout=self._timeout)
        resp.raise_for_status()
        feed = pb2.FeedMessage()
        feed.ParseFromString(resp.content)
        return feed
