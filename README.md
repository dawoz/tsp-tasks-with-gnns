# TSP Tasks with Graph Neural Networks

This project investigates the impact of edge weights and features on the performance of Graph Neural Networks (GNNs) when solving TSP-related tasks.
This work has been developed as an outcome of the GNN course held at the Bertinoro International Spring School 2024.

## Problem Overview

The **Travelling Salesman Problem** (TSP) is a classic combinatorial optimization challenge where the objective is to find the shortest possible route that visits each city once and returns to the starting point. The problem is NP-complete, making it computationally challenging as the number of cities increases.

This project focuses on testing various GNNs to handle TSP tasks by leveraging node and edge features and investigating how these features affect model performance.

## Tasks

The following three tasks were tackled:

1. **Cost Regression**: predict the optimal tour cost for a given TSP instance;
2. **Tour Prediction**: predict the probability of edges being part of the optimal tour;
3. **Heuristic Tour Generation**: generate feasible TSP solutions using reinforcement learning.

## Models

Several GNN architectures were tested for these tasks:

- **Graph Convolutional Network** (GCN);
- **Graph Isomorphism Network** (GIN);
- **Graph Attention Network** (GAT);
- **Transformer** (used as a baseline model).

Each model was tested in three settings:
1. **Node features only**;
2. **With edge weights**;
3. **With edge features**.

## Implementation Details

The project is implemented using **PyTorch** with the **PyTorch Geometric** library for GNN operations and **Salina** for agent-based reinforcement learning tasks.

## Documentation

For a detailed analysis of the tasks, models, experiments, and results, please refer to [paper](docs/TSP_GNNs.pdf).