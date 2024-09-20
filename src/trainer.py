from abc import ABC, abstractmethod
from collections import defaultdict

import tabulate
import torch
from salina import Workspace
from tqdm import tqdm

import wandb


class Trainer(ABC):
    """Abstract class for training a problem with a given agent"""
    def __init__(self, problem, *, preprocess_required_fields=None, step_required_fields=None):
        self.problem = problem
        self.train_workspace = None
        self.eval_workspace = None
        self.optimizer = None
        self.preprocess_required_fields = set(preprocess_required_fields or [])
        self.step_required_fields = set(step_required_fields or [])

    def get_train(self, key):
        """Gets the value of a key in the train workspace"""
        return self.train_workspace[key]
    
    def get_eval(self, key):
        """Gets the value of a key in the evaluation workspace"""
        return self.eval_workspace[key]

    @abstractmethod
    def train_step(self):
        """Perform a training step and return a dictionary of train metrics. Must contain 'loss'"""
        pass

    @abstractmethod
    def compute_metrics(self):
        """Compute metrics from the evaluation workspace. Access the workspace with self.get_eval(key)"""
        pass
    
    def evaluate(self):
        """Evaluation function of the method on unseen instances. The procedure populates self.eval_workspace and it is accessible with self.get_eval(key) in self.compute_metrics()"""
        args = self.problem.args
        assert args.seed != args.seed_eval, "Evaluation seed must be different in order to have unseen instances"
        
        self.problem.to(args.device)
        self.problem.seed(args.seed_eval)
        self.problem.eval()
        eval_data = defaultdict(list)
        
        for _ in range(args.tot_batches_eval):
            workspace = Workspace()
            with torch.no_grad():
                self.problem.preprocess_agent(workspace)
                assert self.preprocess_required_fields.issubset(workspace.keys()), f"Preprocess agent must provide required fields: {', '.join(self.preprocess_required_fields)}"
                self.problem.step_agent(workspace, **self.problem.stop_condition())
                assert self.step_required_fields.issubset(workspace.keys()), f"Step agent must provide required fields: {', '.join(self.step_required_fields)}"
            
            for key in workspace.keys():
                eval_data[key].append(workspace[key].to('cpu'))
            
        eval_data = {key: torch.cat(tensor, 0) if key in self.preprocess_required_fields # there dim 0 is batch_dim
                        else torch.cat(tensor, 1)  # there (during step) dim 0 is temporal_dim
                     for key, tensor in eval_data.items() if key in self.preprocess_required_fields or key in self.step_required_fields}
        self.eval_workspace = Workspace(eval_data)
    
    def train(self):
        """Train the problem with the agent for a given number of batches. Logs the training and evaluation metrics to wandb.
        The procedure populates self.train_workspace and it is accessible with self.get_train(key) in self.train_step().
        Every metric in train and evaluation dictionaries are logged to wandb and printed."""
        args = self.problem.args
        
        self.problem.to(args.device)
        self.problem.seed(args.seed)
        self.train_workspace = Workspace()
        self.optimizer = torch.optim.Adam(self.problem.parameters(), lr=args.lr)

        for step in tqdm(range(args.tot_batches), ascii="·██"):
            self.problem.train()
            
            # forward pass
            self.problem.preprocess_agent(self.train_workspace)
            assert self.preprocess_required_fields.issubset(self.train_workspace.keys()), f"Preprocess agent must provide required fields: {', '.join(self.preprocess_required_fields)}"
            self.problem.step_agent(self.train_workspace, **self.problem.stop_condition())
            assert self.step_required_fields.issubset(self.train_workspace.keys()), f"Step agent must provide required fields: {', '.join(self.step_required_fields)}"
            
            # compute loss and training data
            train_output_dict = self.train_step()
            assert 'loss' in train_output_dict, "'loss' must be provided in train step output dictionary"
            loss = train_output_dict['loss']
            
            # optimization
            self.optimizer.zero_grad(True)
            loss.backward()
            if args.clip_grad:
                torch.nn.utils.clip_grad_norm_(self.problem.parameters(), args.clip_grad)
            self.optimizer.step()
            
            # log
            if step % args.log_step == 0 or step + 1 == args.tot_batches:
                self.evaluate()
                eval_output_dict = self.compute_metrics()
                              
                table = tabulate.tabulate([[key.replace("_", " ").capitalize(), value.item()] for key, value in train_output_dict.items()], tablefmt='psql')
                line = table.split('\n')[-1].replace('-+-', '---')
                print(line + '\n' + ('| Train' + ' ' * len(line))[:len(line)-1] + '|\n' + table)
                wandb.log({f'train/{key}': value for key, value in train_output_dict.items()}, commit=not bool(eval_output_dict))
                    
                if eval_output_dict:
                    table = tabulate.tabulate([[key.replace("_", " ").capitalize(), value.item()] for key, value in eval_output_dict.items()], tablefmt='psql')
                    line = table.split('\n')[-1].replace('-+-', '---')
                    print(line + '\n' + ('| Evaluation' + ' ' * len(line))[:len(line)-1] + '|\n' + table)
                    wandb.log({f'eval/{key}': value for key, value in eval_output_dict.items()}, commit=True)

            self.train_workspace.clear()
