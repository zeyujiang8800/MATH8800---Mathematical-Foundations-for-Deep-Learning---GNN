# GNN MARTA Transit – Stop-Level Delay Prediction

Graph Neural Network pipeline for predicting stop-level delay risk on the MARTA transit network using static GTFS and GTFS-realtime data.

## Project Structure

```
data/
  raw_gtfs/          # Static GTFS .txt files
  raw_realtime/      # GTFS-realtime protobuf snapshots
  processed/         # Feature matrices, labels, etc.
notebooks/           # Jupyter notebooks for exploration
src/
  gtfs_loader.py     # Step 1 – Load & validate static GTFS
  graph_builder.py   # Step 2 – Build transit graph
  realtime_loader.py # Step 3 – Ingest realtime feeds
  feature_engineering.py # Step 4 – Static & dynamic features
  dataset_builder.py # Step 6 – PyG Data objects
  models.py          # Step 7 – MLP & GCN baselines
  train.py           # Step 8 – Training loop
  evaluate.py        # Step 8 – Evaluation & metrics
  utils.py           # Shared utilities
```

## Quick Start

```bash
pip install -r requirements.txt
python -m src.create_sample_gtfs   # generate sample GTFS data
```

## Status

- [x] Step 1: Static GTFS ingestion
- [ ] Step 2: Graph construction
- [ ] Step 3: Realtime ingestion
- [ ] Step 4: Feature engineering
- [ ] Step 5: Label generation
- [ ] Step 6: Dataset builder
- [ ] Step 7: Baseline models
- [ ] Step 8: Training & evaluation
- [ ] Step 9: Notebook demo
