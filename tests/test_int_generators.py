from unittest import TestCase

import numpy as np

from src.derivative_free_optimization.random_value_generators.int_generators import (
    UniqueIntegerGenerator, ValueSwapper)


class TestUniqueIntegerGenerator(TestCase):
    def test_generate(self):
        generator = UniqueIntegerGenerator(1, 4)
        data = [1, 2, 3, 4]
        candidates = np.array(data)

        result = generator.generate(candidates)

        # Not ideal as we only test that the resulting array has values only
        # from the original dataset, but if the method were to simply return
        # the original array, the test would still pass
        expected_values = set(data)
        result_values = set(result)
        self.assertEqual(expected_values, result_values)
        self.assertEqual(len(candidates), len(result))


class TestValueSwapper(TestCase):
    def test_generate(self):
        swap_times = 1
        generator = ValueSwapper(0, 3, swap_times=swap_times)
        data = [1, 2, 3, 4]
        candidates = np.array(data)

        result = generator.generate(candidates.copy())

        expected_values = set(data)
        result_values = set(result)
        self.assertEqual(expected_values, result_values)
        self.assertEqual(len(candidates), len(result))
        count = 0
        for a, b in zip(candidates, result):
            if a != b:
                count += 1
        # Theoretically, the generator could generate the same value to swap,
        # hence we expect the number of values swapped to be either equal to
        # swap times plus one or zero
        self.assertTrue(count in (0, swap_times + 1))
