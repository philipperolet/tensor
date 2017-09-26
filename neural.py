import numpy as np
import math

BATCH_SIZE = 100  # training batch size
ERROR_THRESHOLD = 0.01
GRADIENT_STEP_SIZE = ERROR_THRESHOLD
STEPS_THRESHOLD = 1000


class NeuralNet(object):
    '''Neural network. As a convention, data is an array of inputs with the
    last column being the output'''

    def __init__(self, nb_neurons=8, input_size=3):
        self.nb_neurons = nb_neurons
        self.input_size = input_size
        self._weights = np.zeros((input_size, nb_neurons))
        self._output_weights = np.zeros(nb_neurons)
        self.loss = float('nan')  # current generalization error, undefined at init
        self._previous_loss = float('nan')

    def run(self, x):
        """Runs the network on an input array"""
        if (not(isinstance(x, list)) or len(x) != self.input_size):
            raise TypeError("Input should be list of size {}".format(self.input_size))
        return np.dot(sigmoid(np.dot(np.array(x), self._weights)), self._output_weights)

    def train(self, data):
        while not(self._convergence()):
            self._previous_loss = self.loss
            batch_ints = np.random.randint(len(data), size=BATCH_SIZE)
            self.loss = self._train_batch(data[batch_ints])  # train on batches
        return self.loss

    def _train_batch(self, data):
        # Init
        x = data[:, :-1]

        y = data[:, -1]
        err = y - self.run(x)

        # partial derivative of loss is d/dw (1/2 err^2), that is err * d/dw(err)
        # for the output weight wout_i, d/dwout_i(err) = sigma(wi.x)
        # with wi the vector of weights of neuron i
        wout_err_deriv = sigmoid(np.dot(x, self._weights))

        # for a hidden layer weight j of neuron i, d/dwi_j(err) matrix is as below
        wij_err_deriv = np.multiply(wout_err_deriv,
                                    (1 - wout_err_deriv),
                                    self._output_weights) * np.transpose(x)
    
        # Adjust output weights
        self._output_weights -= np.sum(np.multiply(wout_err_deriv, err)) * GRADIENT_STEP_SIZE

        # Adjust input weights
        self._weights -= np.sum(np.multiply(wij_err_deriv, err)) * GRADIENT_STEP_SIZE
        return np.sum(np.power(y - self.run(x), 2))

    def _convergence(self):
        # Loss / previous loss NaN init make the first ineq eval to False
        return ((abs(self.loss - self._previous_loss) < ERROR_THRESHOLD)
                and self._learning_steps > STEPS_THRESHOLD)

        
def sigmoid(x):
    return 1/(1+np.exp(np.negative(x)))
