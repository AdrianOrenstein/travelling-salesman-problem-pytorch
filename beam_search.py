from typing import *

import numpy as np
import requests
import scipy
import torch
from concorde.tsp import TSPSolver
from tqdm.autonotebook import tqdm

from ml.tspsolver import DefaultTSPSolver
from train_heuristic.data_generation import ConcordeSolver
import matplotlib.pyplot as plt
import concurrent.futures

from functools import partial


class BeamSearchSolver(DefaultTSPSolver):
    def partial_calc_path_cost(self, a: int, b: int):
        return scipy.spatial.distance.euclidean(
            self.city_locations[a], self.city_locations[b])

    def _eval_dfs(self, u: int, ll: int, branch: List[int], visited: Set[int], g: List[int], beam_size: int, cost_so_far: float = 0, best_cost: float = float('inf')):
        visited[u] = True
        branch.append(u)
        if len(branch) == ll:  # if length of branch equals length of string, print the branch
            yield branch.copy()
        else:
            neighbours = [n for n in g if not visited[n]]
            travel_costs = [
                cost_so_far + self.partial_calc_path_cost(branch[-1], n) for n in neighbours
            ]
            neighbours_and_cost = sorted(
                zip(neighbours, travel_costs), key=lambda pair: pair[1], reverse=False
            )

            for n, cost in neighbours_and_cost[:beam_size]:
                yield from self._eval_dfs(n, ll, branch, visited, g, beam_size, cost_so_far + cost)
        # backtrack
        visited[u] = False
        branch.remove(u)

    def _gen_candidates(self, sequence_i, row):
        candidates = []
        seq, score = sequence_i
        for j in range(len(row)):
            if j not in sequence_i[0]:
                if seq == []:
                    candidate = [seq + [j], 0]
                else:
                    candidate = [seq + [j], score + self.partial_calc_path_cost(seq[-1], j)]
                candidates.append(candidate)
        return candidates

    def beam_search_decoder(self, data, k):
        sequences = [[list(), -float('inf')]]
        # read all files in many threads
        threading = False

        # walk over each step in sequence
        for row in tqdm(data):
            all_candidates = list()
            # expand each current candidate

            for i in range(len(sequences)):
                all_candidates.extend(
                    self._gen_candidates(sequences[i], row)
                )
            # order all candidates by score
            ordered = sorted(all_candidates, key=lambda tup: tup[1], reverse=False)
            # select k best
            sequences = ordered[:k]
        
        for res in sequences:
            yield res[0]

    def run(self, city_locations: List[List[float]] = None, algorithm='beam_search'):
        # if type(city_locations) == None:
        #     self.city_locations = np.array(self.city_locations)
        # else:
        #     self.city_locations = np.array(city_locations)

        print(type(self.city_locations))

        if type(self.city_locations) == list:
            self.city_locations = np.array(self.city_locations)
        print(self.city_locations.shape)
        self.adjacency_matrix = self._build_adjacency_matrix(
            self.city_locations)

        display_path_cost = lambda cost: f"best path cost so far: {cost:.3f}"

        best_path_cost = float('inf')
        best_path = None

        if algorithm == 'beam_search':
            path_generator = self.beam_search_decoder(self.adjacency_matrix, 200)
            for path in (t := tqdm(path_generator, desc=display_path_cost(best_path_cost))):
                assert len(set(path)) == len(path)
                cost = self._calculate_path_cost(path)

                if cost < best_path_cost:
                    best_path_cost = cost
                    best_path = path

                    if city_locations == None:
                        self._send_result(path)

                    t.set_description(display_path_cost(best_path_cost))
        elif algorithm == 'dfs':
            branch = []
            visited = [False for _ in range(self.adjacency_matrix.shape[0])]
            for starting_node in range(len(self.adjacency_matrix)):
                path_generator = self._eval_dfs(
                    starting_node, len(self.adjacency_matrix), branch, visited,
                    list(range(len(self.adjacency_matrix))), 5, best_cost=best_path_cost
                )
                for path in (t := tqdm(path_generator, desc=display_path_cost(best_path_cost))):
                    assert len(set(path)) == len(path)
                    cost = self._calculate_path_cost(path)

                    if cost < best_path_cost:
                        best_path_cost = cost
                        best_path = path

                        if type(city_locations) == None:
                            self._send_result(path)

                        t.set_description(display_path_cost(best_path_cost))
        return best_path


# conda install -c anaconda cython
# git clone https: // github.com / jvkersch / pyconcorde
# pip install - e pyconcorde

if __name__ == "__main__":
    solver = BeamSearchSolver(address="10.90.185.46", port="8000")
    solver.run(algorithm='dfs') # 'beam_search'
    
    # for city_locations in (np.random.rand(i, 2) for i in range(10, 500, 10)):
    #     solver = BeamSearchSolver(address="10.90.185.46", port="8000")
    #     optimum = ConcordeSolver(address="10.90.185.46", port="8000")
    #     # solver = BeamSearchSolver()
    #     solver_res = solver.run(city_locations)
    #     solver_cost = solver._calculate_path_cost(solver_res, algorithm='dfs')
        
    #     opt_res = optimum.run(city_locations) 
    #     opt_cost = solver._calculate_path_cost(opt_res)
    #     print(solver_cost, opt_cost)
    #     assert solver_cost <= opt_cost, f'\n{solver_res}\n{opt_res}\n{len(city_locations)}\n{solver_cost} {opt_cost}'
    
