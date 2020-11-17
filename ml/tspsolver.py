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
        self, address: str = None, port: str = None,
    ):
        self.address = address
        self.port = port

        if address != None and port != None:
            self._init_env(address, port)

    def _init_env(self, address, port):
        response = requests.get(f"http://{address}:{port}/cities")

        self.city_locations = response.json()["city_locations"]

    def _calculate_path_cost(
        self, city_indexes: List[Union[float, int]],
    ):
        return sum(
            scipy.spatial.distance.euclidean(
                self.city_locations[a], self.city_locations[b]
            )
            for a, b in zip(
                city_indexes + [city_indexes[-1]], city_indexes[1:] + [city_indexes[0]]
            )
        )

    def _partial_calc_path_cost(self, a: int, b: int):
        return scipy.spatial.distance.euclidean(
            self.city_locations[a], self.city_locations[b]
        )

    def _build_adjacency_matrix(
        self, locations: List[List[Union[float, int]]]
    ) -> torch.FloatTensor:
        """

        """
        locations = np.array(locations)

        adj_matrix = spatial.distance.cdist(locations, locations)

        return torch.FloatTensor(adj_matrix)

    def _normalise_city_distances(
        self, adjacency_matrix: torch.FloatTensor
    ) -> torch.FloatTensor:
        """
        Normalise distances between 0 and 1

        """
        normalised_adj_matrix = adjacency_matrix / adjacency_matrix.sum(1)
        return normalised_adj_matrix

    def _send_result(
        self, path: List[int], user_name=None, algorithm_name=None, message=None
    ):
        user_name = user_name if user_name else "adrian"
        algorithm_name = algorithm_name if algorithm_name else "heuristic_bfs"
        message = message if message else ""
        if self.address != None and self.port != None:
            data = {
                "user_name": user_name,
                "algorithm_name": algorithm_name,
                "message": message,
                "city_order": path,
            }

            response = requests.post(
                f"http://{self.address}:{self.port}/submit", json=data
            )

            print("submitted with", response, "data: ", data)

    def run(
        self, city_locations: List[List[float]] = None,
    ):

        self.city_locations = np.array(
            city_locations if type(city_locations) != None else self.city_locations
        )
        path = list(range(len(self.city_locations)))
        best_path_cost = float("inf")

        display_path_cost = lambda cost: f"best path cost so far: {cost:.3f}"

        for _ in (t := tqdm(range(100000), desc=display_path_cost(best_path_cost))) :
            random.shuffle(path)

            current_cost = self._calculate_path_cost(path)

            if current_cost < best_path_cost:
                best_path_cost = current_cost
                self._send_result(path)

                t.set_description(display_path_cost(best_path_cost))
                t.refresh()  # to show immediately the update
