# coding: utf-8
import unittest
from experiment import Experimenter


class P4Test(unittest.TestCase):

    @staticmethod
    def _test_xp_method(a, b):
        """Test experiment method that sums a & b"""
        return a + b

    def test_experiment_returns_appropriate_dict(self):
        test_params = {"a": [2, 3], "b": [10, 5]}
        results = Experimenter(P4Test._test_xp_method).experiment(test_params, 3)
        self.assertSequenceEqual(results, self._expected_result())

    def _expected_result(self):
        return [
            {
                "method": "_test_xp_method",
                "values": [12, 12, 12],
                "avg": 12.0,
                "std": 0.0,
                "param_combination": {"a": 2, "b": 10},
                "iterations": 3,
            },
            {
                "method": "_test_xp_method",
                "values": [13, 13, 13],
                "avg": 13.0,
                "std": 0.0,
                "param_combination": {"a": 3, "b": 10},
                "iterations": 3,
            },
            {
                "method": "_test_xp_method",
                "values": [7, 7, 7],
                "avg": 7.0,
                "std": 0.0,
                "param_combination": {"a": 2, "b": 5},
                "iterations": 3,
            },
            {
                "method": "_test_xp_method",
                "values": [8, 8, 8],
                "avg": 8.0,
                "std": 0.0,
                "param_combination": {"a": 3, "b": 5},
                "iterations": 3,
            },
        ]
