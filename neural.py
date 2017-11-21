import numpy as np
import time
import logging

BATCH_SIZE = 1000  # training batch size
ERROR_THRESHOLD = 0.01
GRADIENT_STEP_SIZE = 0.00001
STEPS_THRESHOLD = 1000
logging.getLogger().setLevel(logging.INFO)


class NeuralNet(object):
    '''Neural network with nb_inputs inputs and nb_neurons neurons in a hidden layer,
    with a sigmoid activation function for the hidden layer, and a linear output.'''

    def __init__(self, nb_inputs, nb_neurons):
        self.nb_inputs = nb_inputs
        self.nb_neurons = nb_neurons
        self._weights = (np.random.rand(nb_inputs, nb_neurons)-0.5)*0.2
        self._output_weights = (np.random.rand(nb_neurons)-0.5)*0.2
        self.loss = float('nan')  # current generalization error, undefined at init
        self._previous_loss = float('nan')

    def run(self, x):
        """Runs the network on an array of inputs (as lines)"""
        if not((isinstance(x, list) and len(x) == self.nb_inputs) or
               (isinstance(x, np.ndarray) and x.shape[1] == self.nb_inputs)):
            raise TypeError("Input should be list of size {}".format(self.nb_inputs))
        return np.dot(sigmoid(np.dot(np.array(x), self._weights)), self._output_weights)

    def train(self, data):
        '''data is an array with the first n-1 columns being the inputs
        and the last column being the output'''
        count = 0
        while not(self._convergence()):
            self._previous_loss = self.loss
            batch_ints = np.random.randint(len(data), size=BATCH_SIZE)
            self.loss = self._train_batch(data[batch_ints])  # train on batches
            count += 1
            self.log_every5s("Current iteration : {}, loss : {}".format(count, self.loss))
        return self.loss

    def log_every5s(self, message):
        if (not(hasattr(self, "_last_log")) or (time.time() - self._last_log > 5)):
            logging.info(message)
            self._last_log = time.time()

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
        wij_err_deriv = np.transpose(
            np.multiply(wout_err_deriv, (1 - wout_err_deriv)) * self._output_weights
        ) * err
        wij_err_deriv = np.dot(wij_err_deriv, x)
        
        # Adjust output weights -- sum on first axis
        self._output_weights = self._output_weights + (np.sum(np.multiply(np.transpose(wout_err_deriv), err), 1) * GRADIENT_STEP_SIZE)

        # Adjust input weights
        self._weights = self._weights + (np.transpose(wij_err_deriv) * GRADIENT_STEP_SIZE)
        return np.sum(np.power(y - self.run(x), 2))/len(x)

    def _convergence(self):
        # Loss / previous loss NaN init make the first ineq eval to False
        return False; ((abs(self.loss - self._previous_loss) < ERROR_THRESHOLD/10))


def sigmoid(x):
    return 1/(1+np.exp(np.negative(x)))


def relu(x):
    return np.max(0, x)
