from unittest import TestCase

import numpy as np

from src.derivative_free_optimization.random_value_generators.float_generators import \
    FloatUniformGenerator


class TestFloatUniformGenerator(TestCase):
    def test_generate(self):
        generator = FloatUniformGenerator()
        candidates = np.array([1, 2, 3, 4])

        result = generator.generate(candidates)

        self.assertTrue(np.isclose(candidates, result, atol=1.0).all())
