import random
from typing import *

import numpy as np
import requests
import scipy
from scipy import spatial
import torch
from tqdm.autonotebook import tqdm


class DefaultTSPSolver:
    def __init__(
        self,
        address,
        port,
    ):
        self.address = address
        self.port = port

        self._init_env(address, port)

    def _init_env(self, address, port):
        response = requests.get(f"http://{address}:{port}/cities")

        self.city_locations = response.json()["city_locations"]

    def _calculate_path_cost(self, city_indexes: List[Union[float, int]]):
        return sum(
            scipy.spatial.distance.euclidean(
                self.city_locations[a], self.city_locations[b])
            for a, b in zip(
                city_indexes + [city_indexes[-1]],
                city_indexes[1:] + [city_indexes[0]]
            )
        )

    def _build_adjacency_matrix(self, locations: List[List[Union[float, int]]]) -> torch.FloatTensor:
        """

        """
        locations = np.array(locations)

        adj_matrix = spatial.distance.cdist(locations, locations)

        return torch.FloatTensor(adj_matrix)

    def _normalise_city_distances(self, adjacency_matrix: torch.FloatTensor) -> torch.FloatTensor:
        """
        Normalise distances between 0 and 1

        """
        normalised_adj_matrix = adjacency_matrix / adjacency_matrix.sum(1)
        return normalised_adj_matrix

    def _send_result(self, path: List[int]):
        data = {
            "user_name": "adrian",
            "algorithm_name": "beam_search",
            "message": "imagine BFS but it\'s not guaranteed optimimum",
            "city_order": path,
        }

        response = requests.post(
            f"http://{self.address}:{self.port}/submit", json=data
        )

    def run(self):
        path = list(range(len(self.city_locations)))
        best_path_cost = float("inf")

        display_path_cost = lambda cost: f"best path cost so far: {cost:.3f}"

        for _ in (t := tqdm(range(100000), desc=display_path_cost(best_path_cost))):
            random.shuffle(path)

            current_cost = self._calculate_path_cost(path)

            if current_cost < best_path_cost:
                best_path_cost = current_cost
                self._send_result(path)

                t.set_description(display_path_cost(best_path_cost))
                t.refresh()  # to show immediately the update
