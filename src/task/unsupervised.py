import dataclasses
from typing import Literal

import torch

from src.problem import Component
from src.task.generation import (Generation, TSPActor, TSPBaseProblem,
                                 TSPCostEvaluator, TSPGenerator, TSPMasker)
from src.task.supervised import AdjPredictor
from src.trainer import Trainer

########################################################################################################################
# Arguments
########################################################################################################################

@dataclasses.dataclass
class UnsupervisedGeneration(Generation):
    """Train a model to predict the optimal tour cost of a TSP instance with an unsupervised approach."""

    problem: Literal['unsupervised'] = 'unsupervised'
    """Frozen parameter to indicate the problem type"""

    # override the wandb logging parameters
    project_name: str = 'TSP_generation_unsupervised'
    """Project name for wandb"""
    
    row_wise_coef: float = 10.0
    """Coefficient for the row-wise constraint"""
    
    no_self_loops_coef: float = 0.1
    """Coefficient for the no self-loops constraint"""


########################################################################################################################
# Components
########################################################################################################################

# PREPROCESS COMPONENTS

class AdjToHeathMap(Component):
    """Convert the adjacency matrix to a heathmap."""
    def forward(self, **kwargs):
        pred_adj = self.get("pred_adj")
        norm_adj = pred_adj.softmax(-1)
        self.set("pred_adj", pred_adj)
        heathmap = torch.matmul(norm_adj, norm_adj.transpose(1,2).roll(-1,1))
        self.set("heathmap", heathmap)

# STEP COMPONENTS

class TSPHeathMapDecoder(Component):
    """Computes the policy"""
    def forward(self, t, **kwargs):
        heathmap = self.get('heathmap')
        heathmap = heathmap + heathmap.transpose(1, 2)
        heathmap = heathmap.repeat_interleave(self.args.p_runs, 0)
        # (batch_size * p_runs, problem_size, problem_size)
        if t == 0:
            logits = torch.ones(heathmap.shape[:-1], device=self.args.device)
        else:
            action = self.get(("action", t - 1)).view(-1, 1, 1)  # (batch_size * p_runs, 1, 1)
            logits = heathmap.gather(-2, action).squeeze(-1)  # (batch_size * p_runs, problem_size)
            mask = self.get("mask").view(self.args.batch_size * self.args.p_runs, -1)  # (batch_size * p_runs, problem_size)
            logits = logits.masked_fill(mask, -10000)

        logits = torch.log_softmax(logits, dim=-1)
        logits = logits.reshape(self.args.batch_size, self.args.p_runs, -1)
        self.set(("logits", t), logits)

########################################################################################################################
# Problems
########################################################################################################################

class TSPUnsupervisedNonAutoregressive(TSPBaseProblem):
    """Problem to train a model to predict the adjacency matrix of a TSP instance using a non-autoregressive unsupervised approach."""
    def __init__(self, *, args):
        super().__init__(args=args,
                         preprocess_classes=[TSPGenerator, AdjPredictor, AdjToHeathMap],
                         step_classes=[TSPHeathMapDecoder, TSPActor, TSPMasker, TSPCostEvaluator])


########################################################################################################################
# Trainers
########################################################################################################################

class UnsupervisedGenerationTrainer(Trainer):
    """Trainer to train a model to predict the optimal tour of a TSP instance using a non-autoregressive unsupervised approach."""
    def __init__(self, problem):
        super().__init__(
            problem,
            preprocess_required_fields=["data", "pred_adj", "heathmap"],
            step_required_fields=["cost"]
        )

    def train_step(self):
        args = self.problem.args
        data = self.get_train("data")
        heathmap = self.get_train("heathmap")
        pred_adj = self.get_train("pred_adj")
        cost_step = self.get_train("cost")  # (problem_size, batch_size, p_runs)
        distance = torch.cdist(data, data, p=2)

        l1 = ((pred_adj.sum(-1) - 1) ** 2).sum(-1)
        l2 = torch.einsum('bii->b', heathmap)  # trace
        l3 = (distance * heathmap).sum(-1).sum(-1)
        loss = (args.row_wise_coef * l1 + args.no_self_loops_coef * l2 + l3).mean()

        cost = cost_step.sum(0)
        cost_mean = cost.mean()
        cost_std = cost.std()
        cost_min = cost.min(-1).values.mean()

        return {
            "loss": loss,
            "cost_mean": cost_mean,
            "cost_std": cost_std,
            "cost_min": cost_min
            }

    def compute_metrics(self):
        cost = self.get_eval("cost").sum(0)
        
        cost_mean = cost.mean()
        cost_std = cost.std()
        cost_min = cost.min(-1).values.mean()
        
        return {
            "cost_mean": cost_mean,
            "cost_std": cost_std,
            "cost_min": cost_min
        }