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

    def run(self, x):
        """Runs the network on an array of inputs (as lines)"""
        if not((isinstance(x, list) and len(x) == self.nb_inputs) or
               (isinstance(x, np.ndarray) and x.shape[1] == self.nb_inputs)):
            raise TypeError("Input should be list of size {}".format(self.nb_inputs))
        return np.dot(sigmoid(np.dot(np.array(x), self._weights)), self._output_weights)

    def train(self, data):
        '''data is an array with the first n-1 columns being the inputs
        and the last column being the output'''
        training_losses = []
        while True:
            batch_ints = np.random.randint(len(data), size=BATCH_SIZE)
            current_loss = self._train_batch(data[batch_ints])  # train on batches
            training_losses.append(current_loss)
            self.log_every_seconds(
                "Current iteration : {}, current loss : {}, convergence : {} ".format(
                    len(training_losses), training_losses[-1], self._convergence(training_losses)
                ))
        return training_losses

    def log_every_seconds(self, message, seconds=3):
        if (not(hasattr(self, "_last_log")) or (time.time() - self._last_log > seconds)):
            logging.info(message)
            import pdb; pdb.set_trace()
            self._last_log = time.time()

    def _train_batch(self, data):
        # Init
        x = data[:, :-1]
        y = data[:, -1]
        err = y - self.run(x)

        # gradient values, aka partial derivative of loss,
        # are d/dw (1/2 err^2), that is err * d/dw(err)
        # for the output weight wout_i, d/dwout_i(err) = - err * sigma(wi.x)
        # with wi the vector of weights of neuron i 
        wout_err_deriv = sigmoid(np.dot(x, self._weights))

        # for a hidden layer weight j of neuron i, d/dwi_j(err) matrix is as below
        wij_err_deriv = np.transpose(
            np.multiply(wout_err_deriv, (1 - wout_err_deriv)) * self._output_weights
        ) * err
        wij_err_deriv = np.dot(wij_err_deriv, x)
        
        # Adjust output weights -- sum on first axis, go in opposite
        # gradient direction, that is go in err * sigma(wi.x)
        self._output_weights = self._output_weights + (
            np.sum(np.multiply(np.transpose(wout_err_deriv), err), 1) * GRADIENT_STEP_SIZE
        )

        # Adjust input weights
        self._weights = self._weights + (np.transpose(wij_err_deriv) * GRADIENT_STEP_SIZE)
        return np.sum(np.power(y - self.run(x), 2))/len(x)

    def _convergence(self, losses):
        '''Using training losses, if average over last 10% tail of list
        is ~= average over [20%-10%] bracket, then convergence is True'''
        avg_size = int(len(losses)/10)
        prev_avg = np.average(losses[-2*avg_size:-avg_size])
        cur_avg = np.average(losses[-avg_size:])
        return abs(2 * (prev_avg - cur_avg)/(prev_avg + cur_avg)) < 0.01, prev_avg, cur_avg, avg_size


def sigmoid(x):
    return 1/(1+np.exp(np.negative(x)))


def relu(x):
    return np.max(0, x)
