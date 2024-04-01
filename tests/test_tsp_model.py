from unittest import TestCase

from src.derivative_free_optimization.models.tsp_model import TSPModel


class TestTSPModel(TestCase):
    def test_evaluate_objective_function(self):
        data = [
            [0, 2, 5, 6],
            [2, 0, 1, 3],
            [5, 1, 0, 4],
            [6, 3, 4, 0],
        ]
        model = TSPModel(data)
        candidates = [x for x in range(len(data))]

        result = model.evaluate_objective_function(candidates)

        self.assertEqual(7.0, result)
