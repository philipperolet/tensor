import numpy as np


class NeuralNet(object):

    def __init__(self, nb_neurons=8, input_size=3):
        self.nb_neurons = nb_neurons
        self.input_size = input_size
        self._weights = np.zeros((input_size, nb_neurons))
        self._output_weights = np.zeros(nb_neurons)

    def run(self, x):
        """Runs the network on an input array"""
        if (not(isinstance(x, list)) or len(x) != self.input_size):
            raise TypeError("Input should be list of size {}".format(self.input_size))
        return np.dot(sigmoid(np.dot(np.array(x), self._weights)), self._output_weights)


def sigmoid(x):
    return 1/(1+np.exp(np.negative(x)))
