from typing import *

import numpy as np
import requests
import scipy
import torch
from concorde.tsp import TSPSolver
from tqdm.autonotebook import tqdm

from ml.tspsolver import DefaultTSPSolver
import matplotlib.pyplot as plt


class BeamSearchSolver(DefaultTSPSolver):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.adjacency_matrix = torch.softmax(self._build_adjacency_matrix(
            self.city_locations
        ), dim=1)

    def partial_calc_path_cost(self, a: int, b: int):
        return scipy.spatial.distance.euclidean(
                self.city_locations[a], self.city_locations[b])

    def _eval_beam_search(self, u: int, ll: int, branch: List[int], visited: Set[int], g: List[int], beam_size: int, cost_so_far: float = 0):
        visited.add(u)
        branch.append(u)
        if len(branch) == ll:  # if length of branch equals length of string, print the branch
            yield branch.copy()
        else:
            neighbours = [n for n in g if n not in visited]
            travel_costs = [cost_so_far + self.partial_calc_path_cost(branch[-1], n) for n in neighbours]
            neighbours_and_cost = sorted(zip(neighbours, travel_costs), key=lambda pair: pair[1])
            
            for n, cost in neighbours_and_cost[:beam_size]:
                yield from self._eval_beam_search(n, ll, branch, visited, g, beam_size, cost_so_far+cost)
        # backtrack
        visited.remove(u)
        branch.remove(u)

    def run(self,):
        print(self.adjacency_matrix.shape)
        display_path_cost = lambda cost: f"best path cost so far: {cost:.3f}"

        best_path_cost = float('inf')
        branch = []
        visited = set()
        for starting_node in range(len(self.adjacency_matrix)):
            path_generator = self._eval_beam_search(
                starting_node, len(self.adjacency_matrix), branch, visited,
                list(range(len(self.adjacency_matrix))), 20
            )
            for path in (t := tqdm(path_generator, desc=display_path_cost(best_path_cost))):
                cost = self._calculate_path_cost(path)
                
                if cost < best_path_cost:
                    best_path_cost = cost
                    self._send_result(path)

                    t.set_description(display_path_cost(best_path_cost))
                    t.refresh()  # to show immediately the update

        return best_path_cost


# conda install -c anaconda cython
# git clone https: // github.com / jvkersch / pyconcorde
# pip install - e pyconcorde

if __name__ == "__main__":
    # solver = BeamSearchSolver(address="10.90.185.46", port="8000")
    solver = BeamSearchSolver()
    solver.run()

    # def beam_search(nodes: List[int], beam_size: int):
    #     branch = []
    #     visited = set()

    #     def bfs(u, ll, branch, visited, g, beam_size):
    #         visited.add(u)
    #         branch.append(u)
    #         if len(branch) == ll:  # if length of branch equals length of string, print the branch
    #             yield "-".join(map(str, branch))
    #         else:
    #             neighbours = [n for n in g if n not in visited]
    #             neighbours.sort()
    #             for n in neighbours[:beam_size]:
    #                 yield from bfs(n, ll, branch, visited, g, beam_size)
    #         # backtrack
    #         visited.remove(u)
    #         branch.remove(u)

    #     for i in nodes:  # use every character as a source
    #         yield from bfs(i, len(nodes), branch, visited, nodes, beam_size)
    # import time
    # start = time.time()
    # i = 0
    # for path in beam_search(list(range(10)), beam_size=3):

    #     # print(f'{time.time() - start: 0.6f}', path)
    #     # start = time.time()
    #     i += 1
    # print(i)
