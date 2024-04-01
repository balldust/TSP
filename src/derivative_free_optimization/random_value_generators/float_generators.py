import numpy as np

from src.derivative_free_optimization.random_value_generators.abstract_generator import \
    RandomValueGenerator


class FloatUniformGenerator(RandomValueGenerator):
    def __init__(self, step_size=1):
        self.__step_size = step_size

    def generate(self, candidates: np.array) -> np.array:
        return candidates + np.multiply(np.random.uniform(-1, 1), self.__step_size)
