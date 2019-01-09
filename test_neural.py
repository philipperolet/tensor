# coding: utf-8

import unittest
import numpy as np
import tensorflow as tf
from sklearn.neural_network import MLPRegressor
from neural import NeuralNet


class NeuralNetTest(unittest.TestCase):

    def test_eval_function_works(self):
        self.nn = NeuralNet(3, 8)
        # fails if input not correct
        with self.assertRaises(TypeError):
            self.nn.run(3)
        with self.assertRaises(TypeError):
            self.nn.run("x")
        with self.assertRaises(TypeError):
            self.nn.run([4, 4, 3, 5])

        # with weights at 0, two different inputs should yield 0
        self.nn._weights = np.zeros([3, 8])
        self.nn._output_weights = np.zeros(8)
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
        self.nn = NeuralNet(3, 4)
        data = self.generate_data()
        loss = self.nn.train(data)
        print "loss is {}".format(loss[-1])
        print "test set results for xy + xz + yz"
        test_data = self.genenrate_data(5)
        print [test_data, self.nn.run(test_data[:, :-1])]

    def generate_data(self, size=100000):
        data = np.zeros((size, 4))
        data[:, :-1] = np.random.randint(-1000, 1000, [size, 3])
        data[:, -1] = np.where(data[:, 0] > 0, 10, 0) - np.where(data[:, 1] > 0, 20, 0)
        return data

    def test_tf(self):
        self.nn = NeuralNet(3, 4)
        # Model parameters
        W_out = tf.Variable(np.reshape(self.nn._output_weights, (4, 1)), dtype=tf.float32)
        W_in = tf.Variable(self.nn._weights, dtype=tf.float32)

        # Model input and output
        x = tf.placeholder(tf.float32, shape=[None, 3])
        neural_model = tf.matmul(tf.sigmoid(tf.matmul(x, W_in)), W_out)
        y = tf.placeholder(tf.float32)

        # loss
        squared_deltas = tf.square(neural_model - y)
        loss = tf.losses.mean_squared_error(y, neural_model)
        # optimizer
        optimizer = tf.train.GradientDescentOptimizer(0.1)
        train = optimizer.minimize(loss)

        # training data
        data = self.generate_data()
        x_train = data[:, :-1]
        y_train = data[:, -1]
        print x_train[1:10]
        print y_train[1:10]
        # training loop
        init = tf.global_variables_initializer()
        sess = tf.Session()
        sess.run(init)
        # print sess.run(neural_model, {x: x_train[1:10]})
        # print self.nn.run(x_train[1:10])
        # sess.run(init)  
        for i in range(500):

            batch_ints = np.random.randint(len(x_train), size=1000)
            if (i % 100 == 0):
                print i
                print sess.run([W_in, W_out])
                print sess.run(neural_model, {x : x_train[1:10]})
#                import pdb; pdb.set_trace()
            sess.run(train, {x: x_train[batch_ints], y: y_train[batch_ints]})

        # evaluate training accuracy
        curr_W, curr_b, curr_loss = sess.run([W_in, W_out, loss], {x: x_train[batch_ints], y: y_train[batch_ints]})

        print("W: %s b: %s loss: %s" % (curr_W, curr_b, curr_loss))

    def test_scikit(self):
        nn = MLPRegressor((4,), 'logistic', 'sgd', warm_start=True)
        for i in range(10):
            data = self.generate_data()
            nn.fit(data[:, :-1], data[:, -1])
            data = self.generate_data()
            print "Loss : {}".format(
                np.sum(np.power(nn.predict(data[:, :-1]) - data[:, -1], 2)) / len(data[:, :-1]))
            print "example : {} -> {}".format(data[:3, :-1], nn.predict(data[:3, :-1]))
