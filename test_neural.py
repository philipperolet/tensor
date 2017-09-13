# coding: utf-8

import unittest

from neural import NeuralNet


class NeuralNetTest(unittest.TestCase):

    def setUp(self):
        self.nn = NeuralNet()

    def test_eval_function_works(self):
        # fails if input not correct
        with self.assertRaises(TypeError):
            self.nn.run(3)
        with self.assertRaises(TypeError):
            self.nn.run("x")
        with self.assertRaises(TypeError):
            self.nn.run([4, 4, 3, 5])

        # with weights at 0, two different inputs should yield 0.5
        self.assertEqual(0.5, self.nn.run([1, 1, 1]))
        self.assertEqual(0.5, self.nn.run([2, 2, 0]))

        # with output weights at (0.5, 0.5, 1), should yield 0
        self.nn._output_weights = [0.5, 0.5, 1.0]
        self.assertEqual(0, self.nn.run([1, 1, 1]))
        self.assertEqual(0, self.nn.run([2, 2, 0]))
