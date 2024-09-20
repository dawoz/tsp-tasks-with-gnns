import torch
from concorde.concorde import Concorde
from concorde.problem import Problem


def concorde_solve(xy):
    """Solve TSP using Concorde solver"""
    problem = Problem.from_coordinates(xy[:, 0].tolist(), xy[:, 1].tolist(), norm='GEO')
    concorde = Concorde()
    solution = concorde.solve(problem)
    tour = torch.tensor(solution.tour).long()
    return tour