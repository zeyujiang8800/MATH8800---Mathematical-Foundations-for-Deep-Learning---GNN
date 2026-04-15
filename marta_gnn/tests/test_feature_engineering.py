"""Tests for feature engineering."""

from marta_gnn.data.graph_builder import GraphBuilder
from marta_gnn.data.feature_engineering import FeatureEngineer


class TestFeatureEngineer:
    def test_enrich_adds_features(self, cfg, mock_tables):
        builder = GraphBuilder(cfg)
        data = builder.build(mock_tables)
        assert data.x.shape[1] == 2  # initially lat/lon

        eng = FeatureEngineer(cfg)
        data = eng.enrich(data, mock_tables, mock_tables.get("realtime"))

        # Should now have 2 (lat,lon) + 10 new features = 12
        assert data.x.shape[1] == 12

    def test_enrich_without_realtime(self, cfg, mock_tables):
        builder = GraphBuilder(cfg)
        data = builder.build(mock_tables)
        eng = FeatureEngineer(cfg)
        data = eng.enrich(data, mock_tables, realtime=None)
        assert data.x.shape[1] == 12  # delay stats will be zero
