# coding: utf-8
import json
import time
import importlib
import torch
import torch.nn as mods

from customnet import CustomNet, CustomNetConv3
from trainer import Trainer as CustomNetTrainer
from experiment import Experimenter
from dlc_practical_prologue import load_data, args
from pprint import pprint


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
                CustomNet(is_cifar=(dataset == 'cifar'), loss_function=loss_method),
                get_data(dataset, dataset_size),
                parameters,
                mods.NLLLoss()
            ).train(20)

        return Experimenter(compute_loss_test_error, pprint).experiment(
            {
                'loss_method': CustomNet.loss_functions.keys(),
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


def run_experiment(xp_name):
    """Runs an experiment described by parameters in the provided python module
    named {xp_name}.py in experiments/ folder

    The result file will be named according to the input file, replacing '.py' by
    '.{date-time}.result.json'
    """
    xp_data = importlib.import_module('experiments.' + xp_name)

    def default_xp_function(**variable_params):
        return CustomNetTrainer(
            CustomNet(is_cifar=(variable_params['dataset'] == 'cifar'), **variable_params),
            get_data(variable_params['dataset'], variable_params['dataset_size']),
            xp_data.fixed_parameters,
            mods.NLLLoss()
        ).train(xp_data.learning_curve_points)

    return Experimenter(default_xp_function, pprint).experiment(
        xp_data.variable_parameters,
        iterations=xp_data.iterations,
        name=xp_name,
        json_dump=True)
