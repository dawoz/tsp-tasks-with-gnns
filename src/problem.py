from abc import ABC, abstractmethod

import torch
from salina import Agent
from salina.agents import Agents, TemporalAgent


class Component(Agent):
    def __init__(self, *, args=None):
        assert type(self) != Component, f"{self.__class__.__name__} is an abstract class and should not be instantiated"
        assert args is not None
        super().__init__()
        self.args = args
        self.gen = torch.Generator(args.device)

    def seed(self, seed):
        self.gen.manual_seed(seed)    


class Problem(ABC, Agent):
    def __init__(self, *, args=None, preprocess_classes=None, step_classes=None):
        assert type(self) != Problem, f"{self.__class__.__name__} is an abstract class and should not be instantiated"
        assert args is not None
        super().__init__()
        if preprocess_classes is None:
            preprocess_classes = []
        if step_classes is None:
            step_classes = []
        self.args = args
        self.preprocess_agent = Agents(*(agent_class(args=args) for agent_class in preprocess_classes))
        self.step_agent = TemporalAgent(Agents(*(agent_class(args=args) for agent_class in step_classes)))
            
    def seed(self, seed):
        self.preprocess_agent.seed(seed)
        self.step_agent.seed(seed)
            
    def num_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    @abstractmethod
    def stop_condition(self):
        pass

