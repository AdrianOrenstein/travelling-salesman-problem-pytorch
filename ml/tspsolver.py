import random
import requests


from scipy.spatial import distance
from typing import *
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
            distance.euclidean(self.city_locations[a], self.city_locations[b])
            for a, b in zip(
                city_indexes + [city_indexes[-1]
                                ], city_indexes[1:] + [city_indexes[0]]
            )
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
                data = {
                    "user_name": "adrian",
                    "algorithm_name": "random",
                    "message": "guaranteed global optima as time approaches ∞",
                    "city_order": path,
                }

                response = requests.post(
                    f"http://{self.address}:{self.port}/submit", json=data
                )

                t.set_description(display_path_cost(best_path_cost))
                t.refresh()  # to show immediately the update
