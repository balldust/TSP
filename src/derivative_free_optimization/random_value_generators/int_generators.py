import random

import numpy as np

from src.derivative_free_optimization.random_value_generators.abstract_generator import \
    RandomValueGenerator


class UniqueIntegerGenerator(RandomValueGenerator):
    def __init__(self, min_bound: int, max_bound: int):
        self.__min_bound = min_bound
        self.__max_bound = max_bound

    def generate(self, candidates: np.array) -> np.array:
        num_random_integers = len(candidates)

        # Generate unique random integers within the specified range
        return np.random.choice(
            np.arange(self.__min_bound, self.__max_bound + 1),
            size=num_random_integers,
            replace=False,
        )


class ValueSwapper(RandomValueGenerator):
    def __init__(self, min_bound: int, max_bound: int, swap_times=1):
        self.__min_bound = min_bound
        self.__max_bound = max_bound
        self.__swap_times = swap_times

    def generate(self, candidates: np.array) -> np.array:
        for _ in range(self.__swap_times):
            first = random.randint(self.__min_bound, self.__max_bound)
            second = random.randint(self.__min_bound, self.__max_bound)
            candidates[second], candidates[first] = (
                candidates[first],
                candidates[second],
            )
        return candidates
