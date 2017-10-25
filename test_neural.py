# coding: utf-8

import unittest
import numpy as np
import tensorflow as tf

from neural import NeuralNet, BATCH_SIZE


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
        data[:, -1] = (data[:, 0] + 2 * data[:, 1])/3
        return data

    def test_tf(self):
        # Model parameters
        W_out = tf.Variable(tf.truncated_normal([4, 1], stddev=0.1), dtype=tf.float32)
        W_in = tf.Variable(tf.truncated_normal([3, 4], stddev=0.1), dtype=tf.float32)

        # Model input and output
        x = tf.placeholder(tf.float32, shape=[None, 3])
        neural_model = tf.matmul(tf.sigmoid(tf.matmul(x, W_in)), W_out)
        y = tf.placeholder(tf.float32)

        # loss
        loss = tf.reduce_sum(tf.square(neural_model - y))  # sum of the squares
        # optimizer
        optimizer = tf.train.GradientDescentOptimizer(0.001)
        train = optimizer.minimize(loss)

        # training data
        x_train = np.random.rand(100000, 3)
        y_train = (x_train[:, 0] + 2 * x_train[:, 1])/3
        print x_train[1:10]
        print y_train[1:10]
        # training loop
        init = tf.global_variables_initializer()
        sess = tf.Session()
        sess.run(init)  # reset values to wrong
        for i in range(500):
            print i
            batch_ints = np.random.randint(len(x_train), size=BATCH_SIZE)
            sess.run(train, {x: x_train[batch_ints], y: y_train[batch_ints]})

        # evaluate training accuracy
        curr_W, curr_b, curr_loss = sess.run([W_in, W_out, loss], {x: x_train[batch_ints], y: y_train[batch_ints]})

        print("W: %s b: %s loss: %s" % (curr_W, curr_b, curr_loss))
