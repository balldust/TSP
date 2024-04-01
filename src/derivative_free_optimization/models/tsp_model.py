from typing import Any, List

import numpy as np

from src.derivative_free_optimization.models.optimization_model import \
    OptimizationModel


class TSPModel(OptimizationModel):
    def __init__(self, distances: List[List[int]]):
        self.__size = len(distances)
        self.__variable_bounds = np.array(
            [[0, self.__size - 1] for _ in range(self.__size)]
        )
        self.__distances = np.array(distances)

    def evaluate_objective_function(self, manipulated_vars: Any) -> float:
        return float(
            np.sum(self.__distances[manipulated_vars[:-1], manipulated_vars[1:]])
        )

    def evaluate_constraints(self, manipulated_vars: Any) -> np.array:
        return np.array([0])

    def variable_bounds(self) -> np.array:
        return self.__variable_bounds

    def variable_initial_guesses(self) -> np.array:
        return np.array([x for x in range(self.__size)])
