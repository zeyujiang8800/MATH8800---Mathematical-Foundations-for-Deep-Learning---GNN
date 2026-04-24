# GNN w/Cora + GNN w/MARTA Transit – Stop-Level Delay Prediction

Graph Neural Network pipeline for predicting stop-level delay risk on the MARTA transit network using static GTFS and GTFS-realtime data. For the MARTA example, we use a GTFS-inspired mock transit dataset generated to resemble a transit stop network. Nodes represent stops, edges represent connectivity between stops, and node features encode stop-level and delay-related attributes. A binary label is assigned based on whether a stop exceeds a delay threshold of 300 seconds. This example is intended to demonstrate how GNNs can be applied to transportation networks, rather than to claim deployment-ready performance on live MARTA data.

Please see the gnn.ipynb and demo.ipynb notebook files for an interactive example. The notebooks can be ran smoothly with a run-all button

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
