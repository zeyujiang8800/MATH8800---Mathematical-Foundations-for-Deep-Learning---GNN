"""Tests for the full dataset builder pipeline."""

from marta_gnn.data.dataset_builder import DatasetBuilder


class TestDatasetBuilder:
    def test_build_mock_data(self, cfg):
        builder = DatasetBuilder(cfg)
        data = builder.build()

        assert data.num_nodes > 0
        assert hasattr(data, "y")
        assert data.y.shape[0] == data.num_nodes
        assert hasattr(data, "train_mask")
        assert hasattr(data, "val_mask")
        assert hasattr(data, "test_mask")

    def test_masks_are_disjoint(self, cfg):
        builder = DatasetBuilder(cfg)
        data = builder.build()

        overlap = data.train_mask & data.val_mask
        assert overlap.sum() == 0
        overlap2 = data.val_mask & data.test_mask
        assert overlap2.sum() == 0

    def test_masks_cover_all(self, cfg):
        builder = DatasetBuilder(cfg)
        data = builder.build()
        total = data.train_mask.sum() + data.val_mask.sum() + data.test_mask.sum()
        assert total == data.num_nodes
