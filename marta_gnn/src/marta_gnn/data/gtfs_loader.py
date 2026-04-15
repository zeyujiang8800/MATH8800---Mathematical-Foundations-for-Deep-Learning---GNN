"""Load MARTA GTFS static data (stops, routes, trips, stop_times)."""

import io
import logging
import zipfile
from pathlib import Path
from typing import Any

import pandas as pd
import requests

logger = logging.getLogger(__name__)


class GTFSLoader:
    """Download and parse MARTA GTFS static feed."""

    # Files we care about inside the GTFS zip
    _REQUIRED_FILES = ["stops.txt", "routes.txt", "trips.txt", "stop_times.txt"]
    _OPTIONAL_FILES = ["shapes.txt", "calendar.txt", "calendar_dates.txt"]

    def __init__(self, cfg: dict[str, Any]) -> None:
        self.url: str = cfg["gtfs"]["static_url"]
        self.static_dir = Path(cfg["gtfs"]["static_dir"])
        self.static_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def download(self, force: bool = False) -> Path:
        """Download the GTFS zip to *static_dir* and extract it.

        Returns the directory containing the extracted text files.
        """
        zip_path = self.static_dir / "google_transit.zip"
        if zip_path.exists() and not force:
            logger.info("GTFS zip already cached at %s", zip_path)
        else:
            logger.info("Downloading GTFS static feed from %s …", self.url)
            resp = requests.get(self.url, timeout=120)
            resp.raise_for_status()
            zip_path.write_bytes(resp.content)
            logger.info("Saved %d bytes to %s", len(resp.content), zip_path)

        self._extract(zip_path)
        return self.static_dir

    def load_stops(self) -> pd.DataFrame:
        """Return ``stops.txt`` as a DataFrame."""
        return self._read("stops.txt")

    def load_routes(self) -> pd.DataFrame:
        """Return ``routes.txt`` as a DataFrame."""
        return self._read("routes.txt")

    def load_trips(self) -> pd.DataFrame:
        """Return ``trips.txt`` as a DataFrame."""
        return self._read("trips.txt")

    def load_stop_times(self) -> pd.DataFrame:
        """Return ``stop_times.txt`` as a DataFrame with parsed times."""
        df = self._read("stop_times.txt")
        for col in ("arrival_time", "departure_time"):
            if col in df.columns:
                df[col] = df[col].apply(_parse_gtfs_time)
        return df

    def load_all(self) -> dict[str, pd.DataFrame]:
        """Convenience: load all required tables into a dict."""
        return {
            "stops": self.load_stops(),
            "routes": self.load_routes(),
            "trips": self.load_trips(),
            "stop_times": self.load_stop_times(),
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _extract(self, zip_path: Path) -> None:
        with zipfile.ZipFile(zip_path, "r") as zf:
            names = zf.namelist()
            for fname in self._REQUIRED_FILES:
                if fname not in names:
                    raise FileNotFoundError(
                        f"GTFS zip is missing required file: {fname}"
                    )
            zf.extractall(self.static_dir)
        logger.info("Extracted GTFS to %s", self.static_dir)

    def _read(self, filename: str) -> pd.DataFrame:
        path = self.static_dir / filename
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found – call download() first or check static_dir"
            )
        df = pd.read_csv(path, dtype=str)
        logger.debug("Loaded %s: %d rows, %d cols", filename, len(df), len(df.columns))
        return df


def _parse_gtfs_time(raw: str | float) -> int | None:
    """Convert a GTFS time string ``HH:MM:SS`` to seconds since midnight.

    GTFS allows hours ≥ 24 for trips past midnight.  Returns *None* for
    missing / unparseable values.
    """
    if pd.isna(raw):
        return None
    try:
        parts = str(raw).strip().split(":")
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
    except (ValueError, IndexError):
        return None
