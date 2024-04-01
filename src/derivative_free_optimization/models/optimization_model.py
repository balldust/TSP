from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class OptimizationModel(ABC):
    @abstractmethod
    def evaluate_objective_function(self, manipulated_vars: Any) -> float:
        pass

    @abstractmethod
    def evaluate_constraints(self, manipulated_vars: Any) -> np.array:
        pass

    @abstractmethod
    def variable_bounds(self) -> np.array:
        pass

    @abstractmethod
    def variable_initial_guesses(self) -> np.array:
        pass
