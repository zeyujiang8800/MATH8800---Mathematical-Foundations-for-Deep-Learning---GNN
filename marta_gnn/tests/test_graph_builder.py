"""Tests for graph construction."""

import torch

from marta_gnn.data.graph_builder import GraphBuilder, deduplicate_edges


class TestGraphBuilder:
    def test_builds_graph_from_mock(self, cfg, mock_tables):
        builder = GraphBuilder(cfg)
        data = builder.build(mock_tables)

        assert data.num_nodes > 0
        assert data.edge_index.shape[0] == 2
        assert data.edge_index.shape[1] > 0
        assert data.x.shape[0] == data.num_nodes
        assert data.x.shape[1] == 2  # lat, lon

    def test_stop2idx_complete(self, cfg, mock_tables):
        builder = GraphBuilder(cfg)
        data = builder.build(mock_tables)
        assert len(data.stop2idx) == data.num_nodes

    def test_edge_indices_in_range(self, cfg, mock_tables):
        builder = GraphBuilder(cfg)
        data = builder.build(mock_tables)
        assert data.edge_index.min() >= 0
        assert data.edge_index.max() < data.num_nodes


class TestDeduplicateEdges:
    def test_removes_dupes(self):
        ei = torch.tensor([[0, 1, 0], [1, 2, 1]])
        deduped = deduplicate_edges(ei)
        assert deduped.shape[1] == 2
