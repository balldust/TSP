from abc import ABC, abstractmethod

import numpy as np


class RandomValueGenerator(ABC):
    @abstractmethod
    def generate(self, candidates: np.array) -> np.array:
        pass
