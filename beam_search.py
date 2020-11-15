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

    def _eval_dfs(self, u: int, ll: int, branch: List[int], visited: Set[int], g: List[int], beam_size: int, cost_so_far: float = 0, best_cost: float = float('inf')):
        visited[u] = True
        branch.append(u)
        if len(branch) == ll:  # if length of branch equals length of string, print the branch
            yield branch
        else:
            neighbours = [n for n in g if not visited[n]]
            travel_costs = [cost_so_far + self.partial_calc_path_cost(branch[-1], n) for n in neighbours]
            neighbours_and_cost = sorted(zip(neighbours, travel_costs), key=lambda pair: pair[1])
            
            for n, cost in neighbours_and_cost[:beam_size]:
                yield from self._eval_dfs(n, ll, branch, visited, g, beam_size, cost_so_far+cost)
        # backtrack
        visited[u] = False
        branch.remove(u)

    def beam_search_decoder(self,data, k):
        sequences = [[list(), 0.0]]
        # walk over each step in sequence
        for row in data:
            all_candidates = list()
            # expand each current candidate
            for i in range(len(sequences)):
                seq, score = sequences[i]
                for j in range(len(row)):
                    candidate = [seq + [j], score - np.log(row[j])]
                    all_candidates.append(candidate)
            # order all candidates by score
            ordered = sorted(all_candidates, key=lambda tup:tup[1])
            # select k best
            sequences = ordered[:k]
        return [s[0] for s in sequences]

    def run(self,):
        print(self.adjacency_matrix.shape)
        display_path_cost = lambda cost: f"best path cost so far: {cost:.3f}"

        best_path_cost = float('inf')
        branch = []
        visited = [False for _ in range(self.adjacency_matrix.shape[0])]
        for starting_node in range(len(self.adjacency_matrix)):
            # path_generator = self._eval_dfs(
            #     starting_node, len(self.adjacency_matrix), branch, visited,
            #     list(range(len(self.adjacency_matrix))), 1, best_cost = best_path_cost
            # )
            path_generator = self.beam_search_decoder(self.adjacency_matrix, 250)
            for path in (t := tqdm(path_generator, desc=display_path_cost(best_path_cost))):
                cost = self._calculate_path_cost(path)
                
                if cost < best_path_cost:
                    best_path_cost = cost
                    print(cost)
                    self._send_result(path)

                    t.set_description(display_path_cost(best_path_cost))
                    # t.refresh()  # to show immediately the update

        return best_path_cost


# conda install -c anaconda cython
# git clone https: // github.com / jvkersch / pyconcorde
# pip install - e pyconcorde

if __name__ == "__main__":
    # solver = BeamSearchSolver(address="10.90.185.46", port="8000")
    solver = BeamSearchSolver()
    solver.run()
