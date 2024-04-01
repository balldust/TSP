from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from src.derivative_free_optimization.models.optimization_model import \
    OptimizationModel


@dataclass
class OptimizationResults:
    obj_fun: float
    best_inputs: Any


class DerivativeFreeOptimizer(ABC):
    @abstractmethod
    def __init__(self, model: OptimizationModel):
        self._model = model

    @abstractmethod
    def solve(self) -> OptimizationResults:
        pass
