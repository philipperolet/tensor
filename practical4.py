# coding: utf-8
import time
import torch
import argparse
import torch.nn as mods
import torch.nn.functional as F
import dlc_practical_prologue as prologue


class CustomNet(torch.nn.Module):

    def __init__(self):
        super(CustomNet, self).__init__()
        self.conv1 = mods.Conv2d(1, 32, kernel_size=5)
        self.conv2 = mods.Conv2d(32, 64, kernel_size=5)
        self.fc1 = mods.Linear(256, 200)
        self.fc2 = mods.Linear(200, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), kernel_size=3, stride=3))
        x = F.relu(F.max_pool2d(self.conv2(x), kernel_size=2, stride=2))
        x = x.view(-1, 256)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class CustomNetTrainer(object):

    def __init__(self, model, data, parameters):
        self.model = model
        self.data = data
        self.params = parameters
        self.loss = mods.MSELoss()
        self.optimizer = torch.optim.SGD(model.parameters(), lr=self.params['eta'])

    def train(self):
        training_start = time.perf_counter()
        for step in range(self.params['steps']):
            self._train_step(step, training_start)

    def _train_step(self, step, training_start):
        chunk_nb = int(self.data['training_input'].size(0)/self.params['minibatch_size'])
        data_batches = torch.chunk(self.data['training_input'], chunk_nb)
        target_batches = torch.chunk(self.data['training_target'], chunk_nb)

        for data_batch, target_batch in zip(data_batches, target_batches):
            self._minibatch_step(data_batch, target_batch)

        self._display_stats(step, training_start)

    def _display_stats(self, step, training_start):
        with torch.no_grad():
            tr_error, test_error = self._compute_errors()
            print("Step {} : Loss {}, error {} %, test error: {} %, elapsed time: {}s".format(
                step,
                self.loss(self.model(self.data['training_input']), self.data['training_target']),
                100.0 * tr_error / self.data['training_input'].size(0),
                100.0 * test_error / self.data['test_input'].size(0),
                int(time.perf_counter() - training_start)
            ))

    def _minibatch_step(self, data_batch, target_batch):
        tr_loss = self.loss(self.model(data_batch), target_batch)
        self.optimizer.zero_grad()
        self.model.zero_grad()
        tr_loss.backward()
        self.optimizer.step()

    def _compute_errors(self):
        '''Computes training error and test error'''
        tr_error = torch.nonzero(
            torch.argmax(self.data['training_target'], dim=1)
            - torch.argmax(self.model(self.data['training_input']), dim=1)
        ).size(0)

        test_error = torch.nonzero(
            torch.argmax(self.data['test_target'], dim=1)
            - torch.argmax(self.model(self.data['test_input']), dim=1)
        ).size(0)

        return tr_error, test_error


parser = argparse.ArgumentParser()
parser.add_argument("-ds", "--datasize", default='normal')
parser.add_argument("-bs", "--batchsize", default=100)
args = parser.parse_args()

# Initialize data
train_data, train_target, test_data, test_target = prologue.load_data(
    normalize=True,
    one_hot_labels=True,
    flatten=False,
    data_size=args.datasize,
)
# zeta = 0.9
# train_target *= zeta

data = dict(
    training_input=train_data,
    training_target=train_target,
    test_input=test_data,
    test_target=test_target,
)

parameters = dict(
    steps=25,
    eta=0.1,
    minibatch_size=args.batchsize,
    data_size=args.datasize
)

CustomNetTrainer(CustomNet(), data, parameters).train()
