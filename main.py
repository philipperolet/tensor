# coding: utf-8
import json
import time
import torch
import torch.nn as mods
import torch.nn.functional as F
from trainer import Trainer as CustomNetTrainer
from experiment import Experimenter
from dlc_practical_prologue import load_data, args
from pprint import pprint
import matplotlib.pyplot as plt


class CustomNet(torch.nn.Module):

    def __init__(self, hidden_units=200):
        super(CustomNet, self).__init__()
        self.conv1 = mods.Conv2d(3, 32, kernel_size=5)
        self.conv2 = mods.Conv2d(32, 64, kernel_size=5)
        self.fc1 = mods.Linear(256, hidden_units)
        self.fc2 = mods.Linear(hidden_units, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), kernel_size=3, stride=3))
        x = F.relu(F.max_pool2d(self.conv2(x), kernel_size=2, stride=2))
        x = x.view(-1, 256)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class CustomNetConv3(torch.nn.Module):

    def __init__(self, hidden_units=200):
        super(CustomNetConv3, self).__init__()
        self.conv1 = mods.Conv2d(1, 16, kernel_size=3)
        self.conv2 = mods.Conv2d(16, 32, kernel_size=4)
        self.conv3 = mods.Conv2d(32, 64, kernel_size=2)
        self.fc1 = mods.Linear(256, hidden_units)
        self.fc2 = mods.Linear(hidden_units, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = F.relu(F.max_pool2d(self.conv3(x), 2))
        x = x.view(-1, 256)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


# Initialize data
train_data, train_target, test_data, test_target = load_data(
    normalize=True,
    one_hot_labels=False,
    flatten=False,
    cifar=args.cifar,
    data_size='full' if args.full else 'normal',
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
    steps=50,
    optimizer_class=torch.optim.Adam,
    optimizer_params=dict(),
    minibatch_size=args.batchsize,
)

default_trainer = CustomNetTrainer(CustomNet(), data, parameters)


def compute_hidden_layer_test_error(hidden_layer_size):
    return CustomNetTrainer(CustomNet(hidden_layer_size), data, parameters, loss=mods.CrossEntropyLoss()).train()


def conv3_xp(convnet):
    parameters['step'] = 100
    return CustomNetTrainer(convnet(), data, parameters).train()


def conv3_experiment():
    # 3 conv layer experiment (ex. 4)
    results = Experimenter(conv3_xp).experiment(
        {'convnet': [CustomNetConv3, CustomNet]},
        iterations=5
    )
    with open("conv3_xp_{}_{}.json".format(args.suffix, int(time.time())), 'w') as res_file:
        json.dump(results, res_file)


def hidden_layer_experiment(hidden_layers=[10, 50], iters=1):
    # Hidden layer experiment (ex. 3)
    return Experimenter(compute_hidden_layer_test_error, pprint).experiment(
        {'hidden_layer_size': hidden_layers},
        iterations=iters,
        json_dump=True,
    )


def sgd_experiment():
    """
    Trying various SGD parameters to see how they influence learning speed
    (by using the proxy of accuracy at a fixed speed)
    """
    def compute_optimizer_test_error(optim):
        parameters = dict(
            steps=25,
            optimizer_class=optim['class'],
            optimizer_params=optim['params'],
            minibatch_size=args.batchsize,
        )
        return CustomNetTrainer(CustomNet(200), data, parameters).train()

    results = Experimenter(compute_optimizer_test_error, pprint).experiment(
        {'optim': [
            {'class': torch.optim.SGD, 'params': {'lr': 0.1, 'momentum': 0}},
            {'class': torch.optim.SGD, 'params': {'lr': 0.2, 'momentum': 0}},
            {'class': torch.optim.SGD, 'params': {'lr': 0.05, 'momentum': 0}},
            {'class': torch.optim.SGD, 'params': {'lr': 0.1, 'momentum': 1}},
            {'class': torch.optim.SGD, 'params': {'lr': 0.2, 'momentum': 1}},
            {'class': torch.optim.SGD, 'params': {'lr': 0.05, 'momentum': 1}},
            {'class': torch.optim.SGD, 'params': {'lr': 0.1, 'momentum': 0.5}},
            {'class': torch.optim.SGD, 'params': {'lr': 0.2, 'momentum': 0.5}},
            {'class': torch.optim.SGD, 'params': {'lr': 0.05, 'momentum': 0.5}},
            {'class': torch.optim.Adam, 'params': {}}
            ]
         },
        iterations=5,
        json_dump=True,
        )


def loss_experiment():
    """
    Tries multiple losses to see which performs best
    """
    def compute_loss_test_error(loss):
        return CustomNetTrainer(CustomNet(200), data, parameters, loss).train()

    pprint(Experimenter(compute_loss_test_error, pprint).experiment(
        {'loss': [
            mods.CrossEntropyLoss(),
            ]},
        iterations=3,
        json_dump=True,
        )
    )


if __name__ == '__main__':
    pprint(hidden_layer_experiment())
