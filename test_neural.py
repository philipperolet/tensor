# coding: utf-8

import unittest
import numpy as np

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

        # with weights at 0, two different inputs should yield 0
        self.assertEqual(0, self.nn.run([1, 1, 1]))
        self.assertEqual(0, self.nn.run([2, 2, 0]))

        # with output weights summing to 10, should yield 5
        self.nn._output_weights = [0.5, 0.5, 0.5, 0.5, 3, 3, 2, 0]
        self.assertEqual(5, self.nn.run([1, 1, 1]))
        self.assertEqual(5, self.nn.run([2, 2, 0]))

        # a few tests with real weights
        self.nn._output_weights = [1]*8
        self.nn._weights = [
            [1, -2, 2, 0, 0, 0, 0, 0],
            [0, -1, -2, 0, 0, 0, 0, 0],
            [1, -2, 0, 0, 0, 0, 0, 0]
        ]

        self.assertEqual(4.5, self.nn.run([2**50, 0, 2**50]))  # Large numbers for easy mental check
        self.assertEqual(4.0, self.nn.run([2**51, 2**51, -2**50]))

        # tests with an array of inputs
        np.testing.assert_array_equal(
            np.array([4.0, 4.5]),
            self.nn.run(np.array([[2**51, 2**51, -2**50], [2**50, 0, 2**50]])))

    def test_training(self):
        data = self.generate_data()
        loss = self.nn.train(data)
        print "loss is {}".format(loss)
        print "test set results for xy + xz + yz"
        test_data = self.generate_data(5)
        print [test_data, self.nn.run(test_data[:, :-1])]

    def generate_data(self, size=100000):
        data = np.zeros((size, 4))
        data[:, :-1] = np.random.rand(size, 3)
        # data[:, -1] = np.multiply(data[:, 0], data[:, 1]) + np.multiply(data[:, 0], data[:, 2]) + np.multiply(data[:, 1], data[:, 2])
        data[:, -1] = data[:, 0] + 2 * data[:, 1]
        return data
