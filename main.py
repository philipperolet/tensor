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


def mse_loss(x):
    """Computes "MSE values" suitable for use with pytorch NLLLoss"""
    assert len(x.size()) == 2
    return x * (x.transpose(0, 1) - torch.eye(x.size(1), x.size(0), device=x.device))


class CustomNet(torch.nn.Module):
    """Customizable CNN

    Args:
    is_cifar -- should be set to True if using CIFAR rather than MNIST
    hidden_units -- size of the last hidden layer
    exp_factor -- to experiment on "pre-CE" normalization
    """

    loss_methods = {
        "exp002": lambda x: F.log_softmax(x.pow(0.02)),
        "crossE": lambda x: F.log_softmax(x),
        "true": lambda x: torch.sign(x - x.max()),
        "almost": lambda x: x - x.max(),
        "MSE": mse_loss,
        "dummy": lambda x: x,
        "bad": lambda x: torch.empty(x.size()).normal_(),
    }

    activation_functions = {
        "ReLU": F.relu,
        "tanh": F.tanh,
        "negsqr": lambda x: torch.mul(x.sign(), x.abs().sqrt()),
        "logweird": lambda x: x.div(2) * x.log(x.pow(2)),
        "LeakyReLU": F.leaky_relu,
        "ReLulu": lambda x: x.div(4) + F.hardtanh(x),
    }

    def __init__(self, is_cifar=args.cifar, hidden_units=200, loss_method="crossE", activation_function="ReLU"):
        super(CustomNet, self).__init__()
        # If cifar dataset, 3 channels, if mnist only 1
        self.conv1 = mods.Conv2d(3 if is_cifar else 1, 32, kernel_size=5)
        self.conv2 = mods.Conv2d(32, 64, kernel_size=5)
        self.fc1 = mods.Linear(256, hidden_units)
        self.fc2 = mods.Linear(hidden_units, 10)
        self.loss_method = CustomNet.loss_methods[loss_method]
        self.activation_function = CustomNet.activation_functions[activation_function]

    def forward(self, x):
        x = self.activation_function(F.max_pool2d(self.conv1(x), kernel_size=3, stride=3))
        x = self.activation_function(F.max_pool2d(self.conv2(x), kernel_size=2, stride=2))
        x = x.view(-1, 256)
        x = self.activation_function(self.fc1(x))
        x = self.fc2(x)
        x = self.loss_method(x)
        return x


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


parameters = dict(
    steps=20,
    optimizer_class=torch.optim.Adam,
    optimizer_params=dict(),
    minibatch_size=args.batchsize,
)


datasets = dict(
    cifar=dict(normal=None, full=None),
    mnist=dict(normal=None, full=None),
)


def get_data(name='cifar' if args.cifar else 'mnist',
             size='full' if args.full else 'normal'):
    """Retrieves desired dataset

    Args:
    name -- name of dataset, defaults to mnist or commandline
    size -- 'nomral' for restricted version of dataset to 1000 examples, 'full' otherwise
    """
    if datasets[name][size]:
        return datasets[name][size]

    # Initialize data
    train_data, train_target, test_data, test_target = load_data(
        normalize=True,
        one_hot_labels=False,
        flatten=False,
        cifar=True if name == 'cifar' else False,
        data_size=size,
    )

    # Adapt it for GPU computation if appropriate
    datasets[name][size] = dict(
        training_input=train_data if not(torch.cuda.is_available()) else train_data.cuda(),
        training_target=train_target if not(torch.cuda.is_available()) else train_target.cuda(),
        test_input=test_data if not(torch.cuda.is_available()) else test_data.cuda(),
        test_target=test_target if not(torch.cuda.is_available()) else test_target.cuda(),
    )

    return datasets[name][size]


class Xprunner(object):
    """Wrapper class to run Experimenter more efficiently """

    def __init__(self):
        pass

    def conv3_experiment(self):
        """3 conv layer experiment (ex. 4)"""
        data = get_data()

        def conv3_xp(convnet):
            parameters['step'] = 100
            return CustomNetTrainer(convnet(), data, parameters).train()

        results = Experimenter(conv3_xp).experiment(
            {'convnet': [CustomNetConv3, CustomNet]},
            iterations=5
        )
        with open("conv3_xp_{}_{}.json".format(args.suffix, int(time.time())), 'w') as res_file:
            json.dump(results, res_file)

    def hidden_layer_experiment(self, hidden_layers=[10, 50], iters=1, cifar=False):
        """Hidden layer experiment (ex. 3)"""
        data = get_data()

        def compute_hidden_layer_test_error(hidden_layer_size):
            return CustomNetTrainer(
                CustomNet(hidden_units=hidden_layer_size),
                data,
                parameters,
                loss=mods.CrossEntropyLoss()
            ).train()

        return Experimenter(compute_hidden_layer_test_error, pprint).experiment(
            {'hidden_layer_size': hidden_layers},
            iterations=iters,
            json_dump=True,
        )

    def sgd_experiment(self):
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
            return CustomNetTrainer(CustomNet(), get_data(), parameters).train()

        return Experimenter(compute_optimizer_test_error, pprint).experiment(
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

    def loss_experiment(self):
        """
        Tries multiple losses to see which performs best
        """
        def compute_loss_test_error(loss_method, dataset, dataset_size):
            return CustomNetTrainer(
                CustomNet(is_cifar=(dataset == 'cifar'), loss_method=loss_method),
                get_data(dataset, dataset_size),
                parameters,
                mods.NLLLoss()
            ).train(20)

        return Experimenter(compute_loss_test_error, pprint).experiment(
            {
                'loss_method': CustomNet.loss_methods.keys(),
                'dataset': ['mnist', 'cifar'],
                'dataset_size': ['normal', 'full'],
            },
            iterations=20,
            json_dump=True,
            )

    def activation_experiment(self):
        """
        Tries multiple losses to see which performs best
        """
        def compute_loss_test_error(activation_function, dataset, dataset_size):
            return CustomNetTrainer(
                CustomNet(is_cifar=(dataset == 'cifar'), activation_function=activation_function),
                get_data(dataset, dataset_size),
                parameters,
                mods.NLLLoss()
            ).train(20)

        return Experimenter(compute_loss_test_error, pprint).experiment(
            {
                'activation_function': CustomNet.activation_functions.keys(),
                'dataset': ['mnist', 'cifar'],
                'dataset_size': ['normal', 'full'],
            },
            iterations=20,
            json_dump=True,
            )

    def default_xp(self):
        global data
        data = get_data()
        pprint(self.hidden_layer_experiment())
