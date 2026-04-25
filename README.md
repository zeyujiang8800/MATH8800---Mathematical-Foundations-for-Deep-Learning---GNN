# GNN w/Cora + GNN w/MARTA Transit – Stop-Level Delay Prediction

## Project Overview

Many real-world datasets are represented as graphs. In a graph, objects are represented as **nodes**, and relationships between objects are represented as **edges**. Examples include citation networks, transportation systems, social networks, molecular structures, and knowledge graphs.

Traditional neural networks such as CNNs and RNNs are designed for regular grid or sequence data. On the other hand, graphs have unordered neighborhoods and a non-static neighbors. GNNs solve this problem by allowing each node to update its representation using information from its neighboring nodes.

This project focuses on the following ideas:
- Graph Convolutional Networks
- Node classification
- Comparison between GNNs and non-graph baselines


The goal of this project is to showcase how to implement the mathematical foundations of GNNs and practical applications through two examples:

1. **Cora citation network node classification**
2. **MARTA stop delay-risk prediction using a GTFS-inspired transit graph**


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
