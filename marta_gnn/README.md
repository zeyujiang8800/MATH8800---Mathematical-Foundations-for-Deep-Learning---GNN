# MARTA GNN – Stop-Level Delay Risk Prediction

Predict delay risk at MARTA transit stops using a Graph Convolutional Network over GTFS data.

## Quick Start

```bash
pip install -r requirements.txt
python -m marta_gnn.main              # trains on synthetic data, saves plots
python -m marta_gnn.main --demo       # quick 50-epoch demo run
```

The notebook runs out of the box with `jupyter notebook notebooks/demo.ipynb`.

## CLI

```
python -m marta_gnn.main [OPTIONS]

Options:
  --config PATH   Path to config YAML          (default: config.yaml)
  --model TYPE    "gcn" or "mlp"               (default: from config)
  --epochs N      Override training epochs      (default: from config)
  --mock / --live Use synthetic or real feeds   (default: mock)
  --demo          Shorthand: --mock --epochs 50
  --seed N        Random seed                   (default: 42)
```

### Live MARTA feeds

```bash
export MARTA_API_KEY="your-key-here"
python -m marta_gnn.main --live
```

### Tests

```bash
pytest -v
```

## Configuration (`config.yaml`)

| Section    | Key                       | Default | Description                       |
|------------|---------------------------|---------|-----------------------------------|
| `data`     | `use_mock`                | `true`  | Synthetic data vs live feeds      |
| `model`    | `type`                    | `gcn`   | `"gcn"` or `"mlp"`               |
| `model`    | `hidden_dim`              | `64`    | Hidden layer width                |
| `model`    | `num_layers`              | `3`     | GCN depth                         |
| `model`    | `epochs`                  | `200`   | Max training epochs               |
| `model`    | `patience`                | `20`    | Early stopping patience            |
| `features` | `delay_threshold_seconds` | `300`   | Delay cutoff for "at-risk" (sec) |
| `training` | `seed`                    | `42`    | Random seed                        |

## Graph Construction

- **Nodes** – each MARTA stop
- **Edges** – bidirectional links between consecutive stops on any trip
- **Edge weights** – travel time (seconds)
- **Node features** (12-dim) – lat, lon, degree, route count, trip count, avg headway, mean/std/max delay, frac delayed, cyclical hour encoding
- **Labels** – binary: `1` if median arrival delay > threshold

## Models

| Model | Description |
|-------|-------------|
| **GCN** | 3-layer GCN with batch-norm, dropout, and residual connections |
| **MLP** | 2-hidden-layer feedforward baseline (ignores graph structure) |

Both use weighted cross-entropy, Adam, and early stopping on validation loss.

## License

MIT
