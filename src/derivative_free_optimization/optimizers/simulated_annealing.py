import copy

import numpy as np

from src.derivative_free_optimization.models.optimization_model import \
    OptimizationModel
from src.derivative_free_optimization.optimizers.abstract_optimizer import (
    DerivativeFreeOptimizer, OptimizationResults)
from src.derivative_free_optimization.random_value_generators.abstract_generator import \
    RandomValueGenerator
from src.derivative_free_optimization.random_value_generators.float_generators import \
    FloatUniformGenerator


class SimulatedAnnealing(DerivativeFreeOptimizer):
    def __init__(
        self,
        model: OptimizationModel,
        n_iter=1000,
        init_temp=300.0,
        value_generator: RandomValueGenerator = None,
    ):
        super().__init__(model)
        self.__n_iter = n_iter
        self.__init_temperature = init_temp
        if not value_generator:
            self.__value_generator = FloatUniformGenerator()
        else:
            self.__value_generator = value_generator

    def solve(self) -> OptimizationResults:
        cur_candidates = self._model.variable_initial_guesses()
        cur_obj_fun = self.__eval_obj_function(cur_candidates)
        best_candidates = copy.deepcopy(cur_candidates)
        best_obj_fun = cur_obj_fun
        temperature = self.__init_temperature
        for i in range(self.__n_iter):
            new_candidates = self.__get_new_candidates(cur_candidates)
            new_obj_fun = self.__eval_obj_function(new_candidates)
            if new_obj_fun - best_obj_fun < 0:
                best_obj_fun, best_candidates = new_obj_fun, new_candidates
            diff = new_obj_fun - cur_obj_fun
            temperature = temperature / (i + 1)
            metropolis = np.exp(-diff / (temperature + 1e-5))
            if diff < 0 or np.random.rand(1) < metropolis:
                cur_candidates, cur_obj_fun = new_candidates, new_obj_fun
        return OptimizationResults(best_obj_fun, best_candidates)

    def __eval_obj_function(self, candidates: np.array) -> float:
        # The constraints are included in the optimization problem in the form
        # of soft constraints
        return (
            self._model.evaluate_objective_function(candidates)
            + self._model.evaluate_constraints(candidates).sum()
        )

    def __get_new_candidates(self, candidates: np.array) -> np.array:
        return self.__value_generator.generate(candidates)
