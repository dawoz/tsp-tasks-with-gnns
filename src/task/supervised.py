import dataclasses
from typing import Literal

import torch
from torch_geometric.utils import to_dense_adj, to_undirected

from src.concorde_utils import concorde_solve
from src.problem import Component
from src.task.generation import (Generation, TSPBaseProblem, TSPCostEvaluator,
                                 TSPEncoder, TSPGenerator)
from src.trainer import Trainer

########################################################################################################################
# Arguements
########################################################################################################################

@dataclasses.dataclass
class CostRegression(Generation):
    """Train a model to predict the optimal tour cost of a TSP instance"""

    problem: Literal['cost-regression'] = 'cost-regression'
    """Frozen parameter to indicate the problem type"""

    # override the problem parameters
    problem_size: int = 100
    """Size of the problem to solve"""
    batch_size: int = 4
    """Batch size for training/evaluation"""
    p_runs: Literal[1] = 1
    """Parallel runs for each instance"""
    tot_batches: int = 1_000
    """Number of batches to train on"""
    lr: float = 1e-4
    """Learning rate for training"""
    clip_grad: float = 1.0
    """Clip gradients to this value"""

    # override the wandb logging parameters
    project_name: str = 'TSP_regression_optimal_tour_cost'


@dataclasses.dataclass
class TourPrediction(CostRegression):
    """Train a model to predict the optimal tour of a TSP instance"""

    problem: Literal['tour-prediction'] = 'tour-prediction'
    """Frozen parameter to indicate the problem type"""

    # override the wandb logging parameters
    project_name: str = 'TSP_optimal_tour_prediction'


########################################################################################################################
# Components
########################################################################################################################

class ConcordeSolver(Component):
    """Agent adapter of the Concorde solver."""
    @staticmethod
    def _tours_to_adj(tours):
        offset = tours.shape[1] * torch.arange(tours.shape[0], device=tours.device)
        rolled = torch.stack([tours, tours.roll(-1, -1)], 1) + offset[:, None, None]
        edge_index = rolled.permute(1, 0, 2).reshape(2, -1)
        edge_index = to_undirected(edge_index)
        batch = torch.arange(tours.shape[0], device=tours.device).repeat_interleave(tours.shape[1])
        return to_dense_adj(edge_index, batch=batch)

    def forward(self):
        data = self.get("data")
        tours = []
        tours = torch.stack([concorde_solve(xy) for xy in data], dim=-1).unsqueeze(-1)

        # (problem_size, batch_size, p_runs)
        self.set("action", tours)

        tours = tours.squeeze(-1).t().to(data.device)
        adj = self._tours_to_adj(tours)
        self.set("adj", adj)


class CostPredictor(TSPEncoder):
    """Predict the optimal tour cost from the graph embeddings."""
    def __init__(self, *, args):
        super().__init__(args=args)
        self.cost_predictor = torch.nn.Linear(args.embedding_dim, 1)

    def forward(self):
        data = self.get("data")
        embeddings = self.net(data)
        graph_embedding = embeddings.sum(1)
        pred_cost = self.cost_predictor(graph_embedding).squeeze(-1)
        self.set("pred_cost", pred_cost)


class AdjPredictor(TSPEncoder):
    """Predict the adjacency matrix from the graph embeddings."""
    def forward(self):
        data = self.get("data")
        embeddings = self.net(data)

        # todo variations: MLP on concatenation of node embeddings, MLP on edge embeddings, etc
        # todo edge embeddings

        # cosine similarity
        cosine = torch.nn.functional.cosine_similarity(embeddings.unsqueeze(1), embeddings.unsqueeze(2), dim=-1)
        self.set("pred_adj", cosine)


########################################################################################################################
# Problems
########################################################################################################################

class TSPOptimalTourCostRegression(TSPBaseProblem):
    """Problem to train a model to predict the optimal tour cost of a TSP instance."""
    def __init__(self, *, args):
        super().__init__(
            args=args,
            preprocess_classes=[TSPGenerator, ConcordeSolver, CostPredictor],
            step_classes=[TSPCostEvaluator],
        )


class TSPOptimalTourPrediction(TSPBaseProblem):
    """Problem to train a model to predict the optimal tour of a TSP instance."""
    def __init__(self, *, args):
        super().__init__(
            args=args,
            preprocess_classes=[TSPGenerator, ConcordeSolver, AdjPredictor],
        )


########################################################################################################################
# Trainers
########################################################################################################################

class OptimalTourCostRegressionTrainer(Trainer):
    """Trainer to train a model to predict the optimal tour cost of a TSP instance."""
    def __init__(self, problem):
        super().__init__(
            problem,
            preprocess_required_fields=["data", "pred_cost"],
            step_required_fields=["cost"],
        )

    def train_step(self):
        pred_cost = self.get_train("pred_cost")  # (batch_size, p_runs)
        cost = self.get_train("cost").sum(0).squeeze(-1)  # (batch_size, p_runs)

        loss = torch.nn.functional.mse_loss(pred_cost, cost)  # MSE loss
        relative_error = ((pred_cost - cost).abs() / cost).mean()  # relative error

        return {"loss": loss, "relative_error": relative_error}

    def compute_metrics(self):
        pred_cost = self.get_eval("pred_cost")  # (batch_size, p_runs)
        cost = self.get_eval("cost").sum(0).flatten()  # (batch_size, p_runs)

        loss = torch.nn.functional.mse_loss(pred_cost, cost)  # MSE loss
        relative_error = ((pred_cost - cost).abs() / cost).mean()  # relative error

        return {"loss": loss, "relative_error": relative_error}


class OptimalTourPredictionTrainer(Trainer):
    """Trainer to train a model to predict the optimal tour of a TSP instance."""
    def __init__(self, problem):
        super().__init__(
            problem, preprocess_required_fields=["data", "adj", "pred_adj"]
        )

    def train_step(self):
        pred_adj = self.get_train("pred_adj")  # (batch_size, problem_size, problem_size)
        adj = self.get_train("adj")  # (batch_size, problem_size, problem_size)

        loss = torch.linalg.norm(pred_adj - adj, ord="fro", dim=(1, 2)).mean()  # Frobenius norm loss
        num_error = torch.linalg.norm(pred_adj.round() - adj, ord=1, dim=(1, 2)).mean() / 2  # number of errors

        # todo add cost gap
        
        # todo add mean of feasible tours predicted

        return {"loss": loss, "num_error": num_error}

    def compute_metrics(self):
        # (batch_size, problem_size, problem_size)
        pred_adj = self.get_eval("pred_adj")
        adj = self.get_eval("adj")  # (batch_size, problem_size, problem_size)

        loss = torch.linalg.norm(pred_adj - adj, ord="fro", dim=(1, 2)).mean()  # Frobenius norm loss
        num_error = torch.linalg.norm(pred_adj.round() - adj, ord=1, dim=(1, 2)).mean() / 2  # number of errors

        return {"loss": loss, "num_error": num_error}
