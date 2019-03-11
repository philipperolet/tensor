import torch
import time
import torch.nn as mods


class Trainer(object):

    def __init__(self, model, data, parameters, loss=mods.MSELoss()):
        self.model = model
        self.data = data
        self.params = parameters
        self.loss = loss
        self.optimizer = torch.optim.SGD(model.parameters(), lr=self.params['eta'])

    def train(self):
        training_start = time.perf_counter()
        for step in range(self.params['steps']):
            self._train_step(step, training_start)
#            self._display_stats(step, training_start)
        return self._compute_test_error()

    def _train_step(self, step, training_start):
        chunk_nb = int(self.data['training_input'].size(0)/self.params['minibatch_size'])
        data_batches = torch.chunk(self.data['training_input'], chunk_nb)
        target_batches = torch.chunk(self.data['training_target'], chunk_nb)

        for data_batch, target_batch in zip(data_batches, target_batches):
            self._minibatch_step(data_batch, target_batch)

    def _print_step_data(self, step, training_start):
        with torch.no_grad():
            print("Step: {}, test error: {:.1f}%, time elapsed: {}s".format(
                step,
                self._compute_test_error() * 100,
                int(time.perf_counter() - training_start)
            ))

    def _minibatch_step(self, data_batch, target_batch):
        tr_loss = self.loss(self.model(data_batch), target_batch.squeeze())
        self.optimizer.zero_grad()
        tr_loss.backward()
        self.optimizer.step()

    def _display_stats(self, step, training_start):
        with torch.no_grad():
            tr_error, test_error = self._compute_errors()
            print("Step {} : Loss {}, error {} %, test error: {} %, elapsed time: {}s".format(
                step,
                self.loss(self.model(self.data['training_input']), self.data['training_target'].squeeze()),
                100.0 * tr_error,
                100.0 * test_error,
                int(time.perf_counter() - training_start)
            ))

    def _compute_errors(self):
        '''Computes training error and test error'''
        return (self._compute_nb_errors(self.data['training_input'], self.data['training_target']),
                self._compute_test_error())

    def _compute_test_error(self):
        return self._compute_nb_errors(self.data['test_input'], self.data['test_target'])

    def _compute_nb_errors(self, input_, target):
        # if target is 1-hot vectors, need to get argmax to reduce to class numbers
        final_target = target.squeeze() if target.size(1) == 1 else torch.argmax(target, dim=1)
        return torch.nonzero(
            final_target
            - torch.argmax(self.model(input_), dim=1)
        ).size(0) / input_.size(0)
