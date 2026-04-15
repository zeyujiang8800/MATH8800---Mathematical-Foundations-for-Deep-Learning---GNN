"""Tests for GTFS static loader (parsing helpers)."""

from marta_gnn.data.gtfs_loader import _parse_gtfs_time


class TestParseGtfsTime:
    def test_normal_time(self):
        assert _parse_gtfs_time("08:30:00") == 8 * 3600 + 30 * 60

    def test_midnight_crossing(self):
        assert _parse_gtfs_time("25:10:00") == 25 * 3600 + 10 * 60

    def test_none_input(self):
        assert _parse_gtfs_time(None) is None
        assert _parse_gtfs_time(float("nan")) is None

    def test_bad_format(self):
        assert _parse_gtfs_time("bad") is None
