import multiprocessing as mp
import pickle
import random
import time
from typing import *

import numpy as np
import requests
import torch
from concorde.tests.data_utils import get_dataset_path
from concorde.tsp import TSPSolver
from ml.tspsolver import DefaultTSPSolver
from scipy.spatial import distance
from tqdm.autonotebook import tqdm


class ConcordeSolver(DefaultTSPSolver):
    def _generate_sample(self, idx: int, dims: int, seed: int) -> dict:
        np.random.seed(seed=seed)
        self.city_locations = np.random.rand(dims, 2)

        solver = TSPSolver.from_data(
            xs=list(self.city_locations[:, 0]),
            ys=list(self.city_locations[:, 1]),
            norm="EUC_2D",
        )

        path = solver.solve(verbose=False, random_seed=idx)

        path_as_list = np.array(path.tour, dtype=int).tolist()

        data = {
            "locations": np.array(self.city_locations, dtype=float).tolist(),
            "solution": path_as_list,
            "cost": self._calculate_path_cost(path_as_list),
        }

        return data, idx

    def generate_data(self, data_dir: str, starting_seed: int, amount: int = 10000):
        # city_locations = np.array(self.city_locations)
        np.random.seed(seed=starting_seed)

        with mp.Pool() as pool:

            multiple_results = [
                pool.apply_async(self._generate_sample, [i, dim, starting_seed + i])
                for i, dim in zip(
                    range(1, amount + 1),
                    np.random.randint(low=15, high=1000 + 1, size=amount),
                )
            ]

            for res in (res.get() for res in multiple_results):
                data, idx = res

                with open(
                    data_dir + f"/c{len(data['solution'])}d2_{idx}.pkl", "wb"
                ) as handle:
                    pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def run(self, city_locations: List[List[float]], submit=False):
        self.city_locations = np.array(
            city_locations if type(city_locations) != None else self.city_locations
        )
        solver = TSPSolver.from_data(
            xs=list(self.city_locations[:, 0]),
            ys=list(self.city_locations[:, 1]),
            norm="EUC_2D",
        )

        path = solver.solve(verbose=False, random_seed=1)

        if submit:
            self._send_result(
                path=np.array(path.tour, dtype=int).tolist(),
                user_name="(test) concorde_solver",
                algorithm_name="concorde",
                message="have we reached the upperbound?",
            )

        return np.array(path.tour, dtype=int).tolist()


if __name__ == "__main__":
    # address="10.90.185.46", port="8000"
    solver = ConcordeSolver()
    solver.generate_data(
        data_dir="/home/adrian/projects/travelling-salesman-problem-pytorch/train_heuristic/data/training",
        starting_seed=12345678,
    )
    solver.generate_data(
        data_dir="/home/adrian/projects/travelling-salesman-problem-pytorch/train_heuristic/data/validation",
        starting_seed=93939393,
    )

    # solver.solve(submit=False)
