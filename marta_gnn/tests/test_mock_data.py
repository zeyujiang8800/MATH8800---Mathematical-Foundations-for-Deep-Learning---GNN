"""Tests for mock data generation."""

from marta_gnn.data.mock_data import generate_mock_data


class TestMockData:
    def test_generates_expected_keys(self, cfg):
        data = generate_mock_data(cfg)
        assert set(data.keys()) >= {"stops", "routes", "trips", "stop_times", "realtime"}

    def test_stop_count_matches_config(self, cfg):
        data = generate_mock_data(cfg)
        assert len(data["stops"]) == cfg["data"]["mock_num_stops"]

    def test_route_count_matches_config(self, cfg):
        data = generate_mock_data(cfg)
        assert len(data["routes"]) == cfg["data"]["mock_num_routes"]

    def test_realtime_has_delays(self, cfg):
        data = generate_mock_data(cfg)
        assert "arrival_delay" in data["realtime"].columns
        assert len(data["realtime"]) > 0
