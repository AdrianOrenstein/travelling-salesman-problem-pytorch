import multiprocessing as mp
from argparse import ArgumentParser
from typing import Dict, List, Set

import matplotlib.pyplot as plt
import numpy as np
import requests
import scipy
import torch
from concorde.tsp import TSPSolver
from tqdm.autonotebook import tqdm

from ml.tspsolver import DefaultTSPSolver
from train_heuristic.generate_data import ConcordeSolver
from functools import partial


class BeamSearchSolver(DefaultTSPSolver):
    def _eval_dfs(
        self,
        u: int,
        ll: int,
        branch: List[int],
        visited: Set[int],
        g: List[int],
        beam_size: int,
        cost_so_far: float = 0,
        best_cost: float = float("inf"),
    ):
        visited[u] = True
        branch.append(u)
        if (
            len(branch) == ll
        ):  # if length of branch equals length of string, print the branch
            yield branch.copy()
        else:
            neighbours = [n for n in g if not visited[n]]
            travel_costs = [
                cost_so_far + self._partial_calc_path_cost(branch[-1], n)
                for n in neighbours
            ]
            neighbours_and_cost = sorted(
                zip(neighbours, travel_costs), key=lambda pair: pair[1], reverse=False
            )

            for n, cost in neighbours_and_cost[:beam_size]:
                yield from self._eval_dfs(
                    n, ll, branch, visited, g, beam_size, cost_so_far + cost
                )
        # backtrack
        visited[u] = False
        branch.remove(u)

    def _gen_candidates(self, sequence_i, row, k):
        candidates = []
        seq, score = sequence_i

        available_nodes = (n for n in range(len(row)) if n not in sequence_i[0])
        for j in available_nodes:
            if seq == []:
                candidate = [seq + [j], 0]
            else:
                candidate = [
                    seq + [j],
                    score + self._partial_calc_path_cost(seq[-1], j),
                ]
            candidates.append(candidate)

        candidates.sort(key=lambda tup: tup[1], reverse=False)
        return candidates[:20]

    def beam_search_decoder(self, data: np.array, k: int, compute_method: str):
        sequences = [[list(), -float("inf")]]
        # read all files in many threads

        print(f"\nk: {k}, {compute_method}")

        if compute_method == "seq":
            # for each depth
            for row in tqdm(data, desc="depth"):
                all_candidates = list()
                # expand each current candidate
                for i in range(len(sequences)):
                    all_candidates.extend(self._gen_candidates(sequences[i], row, k))
        else:
            with mp.Pool() as pool:
                # for each depth
                for row in tqdm(data, desc="depth"):
                    all_candidates = list()

                    if compute_method == "map":
                        func = partial(self._gen_candidates, row=row, k=k)
                        multiple_results = [
                            pool.map_async(
                                func,
                                (sequences[i] for i in range(len(sequences))),
                                # chunksize=len(all_candidates) // 48,
                            )
                        ]

                        for res in (res.get() for res in multiple_results):
                            for chunk in res:
                                all_candidates.extend(chunk)

                    elif compute_method == "apply":
                        multiple_results = [
                            pool.apply_async(
                                self._gen_candidates, [sequences[i], row, k]
                            )
                            for i in range(len(sequences))
                        ]

                        for res in (res.get() for res in multiple_results):
                            all_candidates.extend(res)
                    else:
                        assert False, f'pick: {["seq", "map", "apply"]}'

                    all_candidates.sort(key=lambda tup: tup[1], reverse=False)

                    # select k best
                    sequences = all_candidates[:k]

        for res in sequences:
            yield res[0]

    def run(
        self,
        city_locations: List[List[float]] = None,
        algorithm="beam_search",
        beam_size=250,
        compute_method="apply",
    ):
        if type(city_locations) == type(None):
            self.city_locations = np.array(self.city_locations)
        else:
            self.city_locations = np.array(city_locations)

        print(self.city_locations.shape)

        self.adjacency_matrix = self._build_adjacency_matrix(self.city_locations)

        display_path_cost = lambda cost: f"best path cost so far: {cost:.3f}"

        best_path_cost = float("inf")
        best_path = None

        if algorithm == "beam_search":
            path_generator = self.beam_search_decoder(
                self.adjacency_matrix, beam_size, compute_method
            )
            for path in (
                t := tqdm(path_generator, desc=display_path_cost(best_path_cost))
            ) :
                assert len(set(path)) == len(path)

                cost = self._calculate_path_cost(path)

                if cost < best_path_cost:
                    best_path_cost = cost
                    best_path = path

                    if type(city_locations) == type(None):
                        self._send_result(path)

                    t.set_description(display_path_cost(best_path_cost))
        elif algorithm == "dfs":
            branch = []
            visited = [False for _ in range(self.adjacency_matrix.shape[0])]
            for starting_node in range(len(self.adjacency_matrix)):
                path_generator = self._eval_dfs(
                    starting_node,
                    len(self.adjacency_matrix),
                    branch,
                    visited,
                    list(range(len(self.adjacency_matrix))),
                    beam_size,
                    best_cost=best_path_cost,
                )
                for path in (
                    t := tqdm(path_generator, desc=display_path_cost(best_path_cost))
                ) :
                    assert len(set(path)) == len(path)
                    cost = self._calculate_path_cost(path)

                    if cost < best_path_cost:
                        best_path_cost = cost
                        best_path = path

                        if type(city_locations) == None:
                            self._send_result(path)

                        t.set_description(display_path_cost(best_path_cost))
        return best_path, best_path_cost


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--address", default="10.90.185.46", type=str)
    parser.add_argument("--port", default="8000", type=str)
    parser.add_argument("--gpus", default=1, type=int)
    parser.add_argument("--algorithm", default="beam_search", type=str)
    parser.add_argument(
        "-dev", type=str2bool, nargs="?", const=True, default=False,
    )
    args = parser.parse_args()

    print(args.dev)

    if args.dev == True:
        # for range of nodes, assert that brute force is better than concorde
        for city_locations in (np.random.rand(i, 2) for i in range(20, 100 + 1, 10)):
            solver = BeamSearchSolver()
            optimum = ConcordeSolver()
            # solver = BeamSearchSolver()
            best_path, best_path_cost = solver.run(
                city_locations, algorithm=args.algorithm
            )
            solver_cost = solver._calculate_path_cost(best_path)

            opt_res = optimum.run(city_locations)
            opt_cost = solver._calculate_path_cost(opt_res)
            print(solver_cost, opt_cost)
            assert (
                solver_cost <= opt_cost
            ), f"\n{best_path_cost}\n{opt_res}\n{len(city_locations)}\n{solver_cost} {opt_cost}"
    else:
        solver = BeamSearchSolver(address=args.address, port=args.port)
        best_path, best_cost = solver.run(
            algorithm=args.algorithm, compute_method="apply", beam_size=1000
        )
        print(best_cost, best_path)

