from typing import Any
from unittest import TestCase

import numpy as np

from src.derivative_free_optimization.models.optimization_model import \
    OptimizationModel
from src.derivative_free_optimization.optimizers.simulated_annealing import \
    SimulatedAnnealing


class TestSimulatedAnnealing(TestCase):
    def test_solve(self):
        model = XSquaredModel()
        solver = SimulatedAnnealing(model)

        results = solver.solve()

        self.assertAlmostEqual(0, results.obj_fun, 3)


class XSquaredModel(OptimizationModel):
    def evaluate_objective_function(self, manipulated_vars: Any) -> float:
        return np.power(manipulated_vars, 2)[0]

    def evaluate_constraints(self, manipulated_vars: Any) -> np.array:
        return np.array([])

    def variable_bounds(self) -> np.array:
        return np.array([[-10.0, 10]])

    def variable_initial_guesses(self) -> np.array:
        return np.array([-9])
