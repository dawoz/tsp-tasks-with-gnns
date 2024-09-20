import os
import time
from typing import Union

import colorama
import torch
import tyro
from torch_geometric.nn import GCNConv

import wandb
from src.args import load_config, override_args
from src.gnn import GATWConv, GATv2WConv, GINEWConv
from src.models import GNN, Transformer
from src.task.generation import Generation, GenerationTrainer
from src.task.supervised import (CostRegression,
                                 OptimalTourCostRegressionTrainer,
                                 OptimalTourPredictionTrainer, TourPrediction,
                                 TSPOptimalTourCostRegression,
                                 TSPOptimalTourPrediction)
from src.task.unsupervised import (TSPUnsupervisedNonAutoregressive,
                                   UnsupervisedGeneration,
                                   UnsupervisedGenerationTrainer)


def set_model_classes(args):
    """Set the model classes based on the arguments"""
    
    if args.net_class == 'Transformer':
        args.net_class = Transformer
        args.feed_forward_dim = 128 if args.feed_forward_dim is None else args.feed_forward_dim
        args.num_attention_layers = 3 if args.num_attention_layers is None else args.num_attention_layers
        args.num_attention_heads = 8 if args.num_attention_heads is None else args.num_attention_heads
    elif args.net_class == 'GNN':
        args.net_class = GNN
        if args.gnn_layer_class == 'GCNConv':
            args.gnn_layer_class = GCNConv
        elif args.gnn_layer_class == 'GATWConv':
            args.gnn_layer_class = GATWConv
        elif args.gnn_layer_class == 'GATv2WConv':
            args.gnn_layer_class = GATv2WConv
        elif args.gnn_layer_class == 'GINEWConv':
            args.gnn_layer_class = GINEWConv
        else:
            raise ValueError(f'Unknown GNN layer class: {args.gnn_layer_class}')
        args.num_layers = 3 if args.num_layers is None else args.num_layers
        if args.activation_function == 'tanh':
            args.activation_function = torch.nn.functional.tanh
        elif args.activation_function == 'relu':
            args.activation_function = torch.nn.functional.relu
        elif args.activation_function == 'sigmoid':
            args.activation_function = torch.nn.functional.sigmoid
        elif args.activation_function == 'leaky_relu':
            args.activation_function = torch.nn.functional.leaky_relu
        elif args.activation_function == 'selu':
            args.activation_function = torch.nn.functional.selu
        else:
            raise ValueError(f'Unknown activation function: {args.activation_function}')
        args.add_residual_connections = True if args.add_residual_connections is None else args.add_residual_connections
        args.batch_norm = True if args.batch_norm is None else args.batch_norm
        args.add_self_loops = True if args.add_self_loops is None else args.add_self_loops
        assert not (args.edge_embeddings and args.gnn_layer_class == GCNConv), 'Edge embeddings are not supported with GCNConv'
        args.edge_embeddings = False if args.edge_embeddings is None else args.edge_embeddings
        args.edge_weights = True if args.edge_weights is None else args.edge_weights
    else:
        raise ValueError(f'Unknown net class: {args.net_class}')


if __name__ == '__main__':
    cli_args = tyro.cli(Union[Generation, CostRegression, TourPrediction, UnsupervisedGeneration])

    if cli_args.problem == 'generation':
        problem_class, trainer_class = cli_args.method.value, GenerationTrainer
    elif cli_args.problem == 'cost-regression':
        problem_class, trainer_class = TSPOptimalTourCostRegression, OptimalTourCostRegressionTrainer
    elif cli_args.problem == 'tour-prediction':
        problem_class, trainer_class = TSPOptimalTourPrediction, OptimalTourPredictionTrainer
    elif cli_args.problem == 'unsupervised':
        problem_class, trainer_class = TSPUnsupervisedNonAutoregressive, UnsupervisedGenerationTrainer
    else:
        raise ValueError(f'Unknown problem type: {cli_args.problem}')

    # if a config file is provided with a list of experiments, run them all
    if cli_args.config_file is not None:
        config_list = load_config(cli_args.config_file)
        assert isinstance(config_list, list), 'YAML file must contain a list of configurations'
        print(colorama.Fore.CYAN + f'Config file with {colorama.Fore.RED + colorama.Style.BRIGHT}{len(config_list)}{colorama.Fore.CYAN + colorama.Style.NORMAL} experiments found' + colorama.Fore.RESET)
    # if no config file is provided, run a single experiment with the provided arguments
    else:
        config_list = [{}]

    # run all experiments
    for i, config in enumerate(config_list):
        if cli_args.config_file is not None:
            print('\n' + colorama.Fore.YELLOW + f'========== Running experiment {colorama.Fore.RED + colorama.Style.BRIGHT}{i + 1}/{len(config_list)}{colorama.Fore.YELLOW + colorama.Style.NORMAL} ==========' + colorama.Fore.RESET)

        args = override_args(cli_args, config)
        set_model_classes(args)
        problem = problem_class(args=args)

        wandb.init(
            project=args.project_name,
            name=args.run_name if args.run_name is not None else f'{problem.__class__.__name__}{args.problem_size}_{args.seed}_{args.net_class.__name__}',
            mode='online' if args.track else 'disabled',
            config=vars(args))

        trainer = trainer_class(problem)
        trainer.train()

        wandb.finish()

        if args.save_dir is not None:
            os.makedirs(args.save_dir, exist_ok=True)
            save_path = f'{args.save_dir}/{args.problem}_{time.strftime("%Y%m%dT%H%M%S")}.pkl'
            torch.save({'args': problem.args, 'state_dict': problem.state_dict()}, save_path)
            print(f'Saved: {save_path}')