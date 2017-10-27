import numpy as np


BATCH_SIZE = 1000  # training batch size
ERROR_THRESHOLD = 0.01
GRADIENT_STEP_SIZE = 0.00001
nSTEPS_THRESHOLD = 1000


class NeuralNet(object):
    '''Neural network. As a convention, data is an array of inputs with the
    last column being the output'''

    def __init__(self, nb_neurons=4, input_size=3):
        self.nb_neurons = nb_neurons
        self.input_size = input_size
        self._weights = (np.random.rand(input_size, nb_neurons)-0.5)*0.2
        self._output_weights = (np.random.rand(nb_neurons)-0.5)*0.2
        self.loss = float('nan')  # current generalization error, undefined at init
        self._previous_loss = float('nan')

    def run(self, x):
        """Runs the network on an input array, lines are """
        if not((isinstance(x, list) and len(x) == self.input_size) or
               (isinstance(x, np.ndarray) and x.shape[1] == self.input_size)):
            raise TypeError("Input should be list of size {}".format(self.input_size))
        return np.dot(sigmoid(np.dot(np.array(x), self._weights)), self._output_weights)

    def train(self, data):
        count = 0
        while not(self._convergence()):
            if (count % 15000 == 0):
                print "Trained batch {} - loss is {}.".format(count, self.loss)
                print "Example on {} - y is {}, nn is {}".format([50, 100, -100], -10, self.run([50, 100, -100]))
                print self._weights
                print self._output_weights
#                import pdb; pdb.set_trace()

            self._previous_loss = self.loss
            batch_ints = np.random.randint(len(data), size=BATCH_SIZE)
            self.loss = self._train_batch(data[batch_ints])  # train on batches
            count += 1

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
