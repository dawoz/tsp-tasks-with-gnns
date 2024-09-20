import dataclasses
import enum
from typing import Literal, Optional

import torch

from src.models import GNN, PointerNet, Transformer
from src.problem import Component, Problem
from src.trainer import Trainer


########################################################################################################################
# Components
########################################################################################################################

# PREPROCESS COMPONENTS

class TSPGenerator(Component):
    """Generates a batch of random Euclidean TSP instances in [0,1]^2"""

    def forward(self, **kwargs):
        data = torch.rand(
            self.args.batch_size,
            self.args.problem_size,
            2,
            device=self.args.device,
            generator=self.gen,
        )
        self.set("data", data)


class TSPEncoder(Component):
    """Encodes each node of the graph into an embedding"""

    def __init__(self, *, args):
        super().__init__(args=args)
        net_constructor = {
            GNN: lambda: GNN(
                embedding_dim=args.embedding_dim,
                num_layers=args.num_layers,
                gnn_layer_class=args.gnn_layer_class,
                add_residual_connections=args.add_residual_connections,
                activation_function=args.activation_function,
                batch_norm=args.batch_norm,
                add_self_loops=args.add_self_loops,
                edge_embeddings=args.edge_embeddings,
                edge_weights=args.edge_weights,
            ),
            Transformer: lambda: Transformer(
                embedding_dim=args.embedding_dim,
                feed_forward_dim=args.feed_forward_dim,
                num_attention_heads=args.num_attention_heads,
                num_attention_layers=args.num_attention_layers,
            ),
        }.get(args.net_class, None)
        assert net_constructor is not None, f"Unknown net class: {args.net_class}"
        self.net = net_constructor()

    def forward(self, **kwargs):
        data = self.get("data")
        embeddings = self.net(data)
        # (batch_size, problem_size, embedding_dim)
        self.set("embeddings", embeddings)


class TSPEdgeLogits(Component):
    """Computes the logits for the edges of the graph"""

    def forward(self, **kwargs):
        embeddings = self.get("embeddings")
        logits_edges = torch.einsum("bid, bjd -> bij", embeddings, embeddings)

        # set diagonal to -inf
        diag = torch.diag(torch.ones(self.args.problem_size, dtype=bool))
        logits_edges[:, diag] = -10000

        # copy upper triangular part to lower triangular part
        triu = torch.triu(torch.ones(self.args.problem_size, self.args.problem_size, dtype=bool), diagonal=1)
        logits_edges[:, triu.t()] = logits_edges[:, triu]

        self.set("logits_edges", logits_edges)  # (batch_size, problem_size, problem_size)


# STEP COMPONENTS

class TSPDecoder(Component):
    """Computes the policy"""

    def __init__(self, *, args):
        super().__init__(args=args)
        self.prob_calculator = PointerNet(args.embedding_dim)

    def forward(self, t, **kwargs):
        # (batch_size * p_runs, problem_size, embedding_dim)
        embeddings = self.get("embeddings").repeat_interleave(self.args.p_runs, 0)
        if t == 0:
            hidden = None
            input = torch.zeros(
                self.args.batch_size * self.args.p_runs,
                self.args.embedding_dim,
                device=self.args.device,
            )
            mask = None
        else:
            hidden = self.get("hidden")
            # (batch_size * p_runs)
            action = self.get(("action", t - 1)).view(-1)
            input = embeddings[range(action.shape[0]), action]
            mask = self.get("mask").reshape(self.args.batch_size * self.args.p_runs, -1)

        logits, hidden = self.prob_calculator(input, embeddings, hidden, mask)
        logits = logits.reshape(self.args.batch_size, self.args.p_runs, -1)
        self.set(("logits", t), logits)
        self.set("hidden", hidden)


class TSPEdgeLogitsDecoder(Component):
    """Computes the policy"""

    def forward(self, t, **kwargs):
        logits_edges = self.get("logits_edges").repeat_interleave(self.args.p_runs, 0)
        # (batch_size * p_runs, problem_size, problem_size)
        if t == 0:
            logits = torch.ones(logits_edges.shape[:-1], device=self.args.device)
        else:
            action = self.get(("action", t - 1)).view(-1, 1, 1)  # (batch_size * p_runs, 1, 1)
            logits = logits_edges.gather(-2, action).squeeze(-1)  # (batch_size * p_runs, problem_size)
            mask = self.get("mask").view(self.args.batch_size * self.args.p_runs, -1)  # (batch_size * p_runs, problem_size)
            logits = logits.masked_fill(mask, -10000)

        logits = torch.log_softmax(logits, dim=-1)
        logits = logits.reshape(self.args.batch_size, self.args.p_runs, -1)
        self.set(("logits", t), logits)


class TSPActor(Component):
    """Takes the action, given the policy"""

    def forward(self, t, greedy=False, **kwargs):
        logits = self.get(('logits', t))
        if greedy:
            action = logits.max(-1)[0]
        else:
            probs = logits.exp()
            action = probs.view(-1, logits.shape[-1]).multinomial(1, generator=self.gen).view(logits.shape[:-1]) # (batch_size, p_runs)
        self.set(('action', t), action)


class TSPMasker(Component):
    """Updates the mask given the action at current time step"""

    def forward(self, t, **kwargs):
        if t == 0:
            mask = torch.zeros(
                self.args.batch_size,
                self.args.p_runs,
                self.args.problem_size,
                dtype=torch.bool,
                device=self.args.device,
            )
        else:
            mask = self.get("mask")

        # get chosen node and update
        action = self.get(("action", t))  # (batch_size, p_runs)
        mask = mask.scatter(-1, action.unsqueeze(-1), True)
        self.set("mask", mask)
        

class TSPCostEvaluator(Component):
    """Computes the cost of the action"""

    def forward(self, t, **kwargs):
        x = self.get("data")
        if t > 0:
            i = self.get(("action", t - 1)).reshape(-1)
            j = self.get(("action", t)).reshape(-1)
            b = torch.arange(self.args.batch_size).repeat_interleave(self.args.p_runs, 0)
            cost = torch.linalg.norm(x[b, i] - x[b, j], dim=-1)
            cost = cost.reshape(self.args.batch_size, self.args.p_runs)
            self.set(("cost", t), cost)
        if t + 1 == self.args.problem_size:
            i = self.get(("action", 0)).reshape(-1)
            j = self.get(("action", t)).reshape(-1)
            b = torch.arange(self.args.batch_size).repeat_interleave(self.args.p_runs, 0)
            cost = torch.linalg.norm(x[b, i] - x[b, j], dim=-1)
            cost = cost.reshape(self.args.batch_size, self.args.p_runs)
            self.set(("cost", 0), cost)


########################################################################################################################
# Problems
########################################################################################################################

class TSPBaseProblem(Problem):
    """Base problem for TSP tasks."""
    def stop_condition(self):
        return dict(n_steps=self.args.problem_size)


class TSPGeneration(TSPBaseProblem):
    def __init__(self, *, args):
        super().__init__(
            args=args,
            preprocess_classes=[TSPGenerator, TSPEncoder],
            step_classes=[TSPDecoder, TSPActor, TSPMasker, TSPCostEvaluator],
        )


class TSPGenerationEdgeLogits(TSPBaseProblem):
    def __init__(self, *, args):
        super().__init__(
            args=args,
            preprocess_classes=[TSPGenerator, TSPEncoder, TSPEdgeLogits],
            step_classes=[TSPEdgeLogitsDecoder, TSPActor, TSPMasker, TSPCostEvaluator],
        )


########################################################################################################################
# Trainers
########################################################################################################################

class GenerationTrainer(Trainer):
    """Trainer for the TSP generation problem with RL"""
    def __init__(self, problem):
        super().__init__(
            problem,
            preprocess_required_fields=["embeddings", "data"],
            step_required_fields=["action", "logits", "cost"],
        )

    def train_step(self):
        action = self.get_train("action")  # (problem_size, batch_size, p_runs)
        logit = self.get_train("logits")  # (problem_size, batch_size, p_runs, problem_size)
        cost_step = self.get_train("cost")  # (problem_size, batch_size, p_runs)

        logit_action = (logit.gather(-1, action.unsqueeze(-1)).squeeze(-1).sum(0))  # chosen action logit
        cost = cost_step.sum(0)  # tour cost
        baseline = cost.mean(-1, keepdim=True)  # POMO baseline

        loss = ((cost - baseline) * logit_action).mean()  # REINFORCE loss
        cost_mean = cost.mean()  # average cost
        cost_std = cost.std()  # cost standard deviation
        cost_min = cost.min(-1).values.mean()  # average minimum cost

        return {
            "loss": loss,
            "cost_mean": cost_mean,
            "cost_std": cost_std,
            "cost_min": cost_min,
        }

    def compute_metrics(self):
        cost_step = self.get_eval("cost")  # (problem_size, batch_size, p_runs)
        cost = cost_step.sum(0)  # tour cost
        cost_mean = cost.mean()  # average cost
        cost_std = cost.std()  # cost standard deviation
        cost_min = cost.min(-1).values.mean()  # average minimum cost

        return {"cost_mean": cost_mean, "cost_std": cost_std, "cost_min": cost_min}


########################################################################################################################
# Arguments
########################################################################################################################

class Method(enum.Enum):
    pointer_net = TSPGeneration
    edge_logits = TSPGenerationEdgeLogits
    

@dataclasses.dataclass
class Generation:
    """Train a model to generate heuristic solutions for the TSP with RL"""

    problem: Literal['generation'] = 'generation'
    """Frozen parameter to indicate the problem type"""
    config_file: Optional[str] = None
    """Path to a YAML file with a list of configurations. If not provided, run a single experiment with the provided arguments."""

    # problem parameters
    problem_size: int = 100
    """Size of the problem to solve"""
    batch_size: int = 32
    """Batch size for training/evaluation"""
    p_runs: int = 20
    """Parallel runs for each instance"""
    tot_batches: int = 10_000
    """Number of batches to train on"""
    tot_batches_eval: int = 4
    """Number of batches to do evaluation"""
    lr: float = 1e-3
    """Learning rate for training"""
    clip_grad: float = 0.5
    """Clip gradients to this value"""
    seed: int = 1234
    """Seed for training"""
    seed_eval: int = 42
    """Seed for evaluation"""
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    """Device to use for training"""

    # wandb logging parameters
    track: bool = True
    """Track results with wandb"""
    log_step: int = 20
    """Log every log_step batches"""
    project_name: str = 'TSP_generation_RL'
    """Name of the wandb project"""
    run_name: Optional[str] = None
    """Name of the run on wandb"""
    save_dir: Optional[str] = None
    """Path for checkpoint. If None, no checkpoint is saved."""

    # generic model parameters
    embedding_dim: int = 128
    """Dimension of the embedding vectors"""
    net_class: str = 'Transformer'
    """Class of the neural network"""
    method: Method = Method.pointer_net
    """Method to use for the generation problem"""

    # params for the transformer
    feed_forward_dim: Optional[int] = None
    """Dimension of the feed forward layers in the encoder"""
    num_attention_layers: Optional[int] = None
    """Number of encoder layers"""
    num_attention_heads: Optional[int] = None
    """Number of attention heads"""

    # params for the GNN
    gnn_layer_class: Optional[str] = None
    """Class of the GNN layer"""
    num_layers: Optional[int] = None
    """Number of GNN layers"""
    activation_function: Optional[str] = None
    """Activation function for the GNN layers"""
    add_residual_connections: Optional[bool] = None
    """Add residual connections to the GNN layers"""
    batch_norm: Optional[bool] = None
    """Add batch normalization to the GNN layers"""
    add_self_loops: Optional[bool] = None
    """Add batch normalization to the GNN layers"""
    edge_embeddings: Optional[bool] = None
    """Add edge embeddings to the GNN layers"""
    edge_weights: Optional[bool] = None
    """Add edge weights to the GNN layers"""
