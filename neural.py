import numpy as np


class NeuralNet(object):
    """
    """

    def __init__(self, nb_neurons=8, input_size=3):
        self.nb_neurons = nb_neurons
        self.input_size = 3
        self.weights = np.zeros((nb_neurons, input_size))

    def run(self, x):
        if (not(isinstance(x, list)) or len(x) != self.input_size):
            raise TypeError("Input should be list of size {}".format(self.input_size))
        return 0.5

