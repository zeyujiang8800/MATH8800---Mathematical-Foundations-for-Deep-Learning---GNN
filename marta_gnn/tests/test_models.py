"""Tests for MLP and GCN models."""

import torch

from marta_gnn.data.dataset_builder import DatasetBuilder
from marta_gnn.models.gcn_model import GCNModel
from marta_gnn.models.mlp_baseline import MLPBaseline
from marta_gnn.training.trainer import build_model


class TestMLPBaseline:
    def test_forward_shape(self):
        model = MLPBaseline(in_dim=12, hidden_dim=32, out_dim=2)
        x = torch.randn(10, 12)
        out = model(x)
        assert out.shape == (10, 2)

    def test_accepts_extra_kwargs(self):
        model = MLPBaseline(in_dim=12, hidden_dim=32, out_dim=2)
        x = torch.randn(10, 12)
        ei = torch.tensor([[0, 1], [1, 0]])
        out = model(x, edge_index=ei)
        assert out.shape == (10, 2)


class TestGCNModel:
    def test_forward_shape(self):
        model = GCNModel(in_dim=12, hidden_dim=32, out_dim=2, num_layers=3)
        x = torch.randn(10, 12)
        ei = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]])
        out = model(x, ei)
        assert out.shape == (10, 2)


class TestBuildModel:
    def test_build_gcn(self, cfg):
        cfg["model"]["type"] = "gcn"
        model = build_model(cfg, in_dim=12)
        assert isinstance(model, GCNModel)

    def test_build_mlp(self, cfg):
        cfg["model"]["type"] = "mlp"
        model = build_model(cfg, in_dim=12)
        assert isinstance(model, MLPBaseline)
