import random
import time
import multiprocessing as mp
from typing import *

import numpy as np
import requests
import torch
from concorde.tsp import TSPSolver
from concorde.tests.data_utils import get_dataset_path
from scipy.spatial import distance
from tqdm.autonotebook import tqdm

from ml.tspsolver import DefaultTSPSolver

import pickle


class ConcordeSolver(DefaultTSPSolver):
    def _generate_sample(self, id: int) -> dict:
        city_locations = np.random.rand(dims, 2)

        solver = TSPSolver.from_data(
            xs=list(city_locations[:, 0]), ys=list(city_locations[:, 1]), norm="EUC_2D")

        path = solver.solve(verbose=False, random_seed=idx)

        data = {
            'locations': np.array(city_locations, dtype=float).tolist(),
            'solution': np.array(path.tour, dtype=int).tolist(),
            'cost': self._calculate_path_cost(np.array(path.tour, dtype=int).tolist())
        }
        return data

    def generate_data(self, data_dir: str, starting_seed: int):
        # city_locations = np.array(self.city_locations)
        np.random.seed(seed=starting_seed)
        dims = 60

        with mp.Pool() as pool:

            results = [pool.apply_async(calculate, t) for t in TASKS]

            for res in tqdm(results):
                data = res.get()

                with open(data_dir + f'/c{dims}d2_{idx}.pkl', 'wb') as handle:
                    pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def run(self, city_locations: List[List[float]], submit=False):
        self.city_locations = np.array(city_locations if type(city_locations) != None else self.city_locations)
        solver = TSPSolver.from_data(
            xs=list(self.city_locations[:, 0]), ys=list(self.city_locations[:, 1]), norm="EUC_2D")

        path = solver.solve(verbose=False, random_seed=1)

        if submit:
            self._send_result(
                path=np.array(path.tour, dtype=int).tolist(),
                user_name='(test) concorde_solver',
                algorithm_name='concorde',
                message='have we reached the upperbound?',
            )

        return np.array(path.tour, dtype=int).tolist()


if __name__ == "__main__":
    # address="10.90.185.46", port="8000"
    solver = ConcordeSolver()
    solver.generate_data(
        data_dir='/home/adrian/projects/travelling-salesman-problem-pytorch/train_heuristic/data/validation',
        starting_seed=93939393,
    )
    solver.generate_data(
        data_dir='/home/adrian/projects/travelling-salesman-problem-pytorch/train_heuristic/data/training',
        starting_seed=12345678,
    )
    # solver.solve(submit=False)
