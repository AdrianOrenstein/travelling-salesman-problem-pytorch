import random
import time
from multiprocessing import Process
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
    def run(self,):
        # city_locations = np.array(self.city_locations)

        for ID in range(1, 1000):
            np.random.seed(seed=ID)
            city_locations = np.random.rand(20, 2)

            solver = TSPSolver.from_data(
                xs=list(city_locations[:, 0]), ys=list(city_locations[:, 1]), norm="EUC_2D")

            path = solver.solve(verbose=False, random_seed=ID)

            data = {
                'locations': np.array(city_locations, dtype=float).tolist(),
                'solution': np.array(path.tour, dtype=int).tolist(),
                'cost': self._calculate_path_cost(np.array(path.tour, dtype=int).tolist())
            }

            with open(f'data/training/c20d2_{ID}.pkl', 'wb') as handle:
                pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

            # with open('data/training/c20d2{ID}.pkl', 'rb') as handle:
            #     b = pickle.load(handle)

        print(data)


if __name__ == "__main__":
    solver = ConcordeSolver(address="10.90.185.46", port="8000")
    solver.run()
