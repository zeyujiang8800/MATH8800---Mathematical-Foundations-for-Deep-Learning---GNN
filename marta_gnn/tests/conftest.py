"""Shared fixtures for all tests."""

import pytest

from marta_gnn.config import load_config
from marta_gnn.data.mock_data import generate_mock_data


@pytest.fixture()
def cfg():
    """Return a default configuration dict."""
    cfg = load_config()
    cfg["data"]["use_mock"] = True
    cfg["model"]["epochs"] = 5  # fast for tests
    cfg["model"]["patience"] = 3
    cfg["data"]["mock_num_stops"] = 30
    cfg["data"]["mock_num_routes"] = 3
    cfg["data"]["mock_num_trips"] = 10
    return cfg


@pytest.fixture()
def mock_tables(cfg):
    """Return mock GTFS tables dict."""
    return generate_mock_data(cfg)
