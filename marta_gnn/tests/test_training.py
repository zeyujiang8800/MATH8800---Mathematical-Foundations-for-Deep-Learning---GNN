"""Tests for training loop and evaluation."""

from marta_gnn.data.dataset_builder import DatasetBuilder
from marta_gnn.training.evaluation import evaluate
from marta_gnn.training.trainer import Trainer, build_model


class TestTrainer:
    def test_train_gcn_runs(self, cfg):
        cfg["model"]["type"] = "gcn"
        cfg["model"]["epochs"] = 3
        builder = DatasetBuilder(cfg)
        data = builder.build()
        model = build_model(cfg, in_dim=data.x.shape[1])
        trainer = Trainer(cfg)
        history = trainer.train(model, data)

        assert "train_loss" in history
        assert len(history["train_loss"]) > 0

    def test_train_mlp_runs(self, cfg):
        cfg["model"]["type"] = "mlp"
        cfg["model"]["epochs"] = 3
        builder = DatasetBuilder(cfg)
        data = builder.build()
        model = build_model(cfg, in_dim=data.x.shape[1])
        trainer = Trainer(cfg)
        history = trainer.train(model, data)

        assert len(history["val_acc"]) > 0


class TestEvaluation:
    def test_evaluate_returns_metrics(self, cfg):
        cfg["model"]["type"] = "gcn"
        cfg["model"]["epochs"] = 3
        builder = DatasetBuilder(cfg)
        data = builder.build()
        model = build_model(cfg, in_dim=data.x.shape[1])
        trainer = Trainer(cfg)
        trainer.train(model, data)

        results = evaluate(model, data)
        assert "accuracy" in results
        assert "f1" in results
        assert "confusion_matrix" in results
        assert 0.0 <= results["accuracy"] <= 1.0
