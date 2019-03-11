import torch
import math

from torch import nn
from collections import OrderedDict
from experiment import Experimenter
from trainer import Trainer


def generate_disc_set(nb):
    tr_input = torch.empty(nb, 2).uniform_(-1, 1)
    tr_target = torch.lt(tr_input.norm(dim=1), math.sqrt(2/math.pi))
    return tr_input, tr_target.type(torch.long).view(-1, 1)


def generate_learning_data():
    train_data, train_target = generate_disc_set(1000)
    train_data.sub_(train_data.mean(dim=0)).div_(train_data.std(dim=0))

    test_data, test_target = generate_disc_set(1000)
    test_data.sub_(test_data.mean(dim=0)).div_(test_data.std(dim=0))

    return dict(
        training_input=train_data,
        training_target=train_target,
        test_input=test_data,
        test_target=test_target,
    )


def create_shallow_model():
    return nn.Sequential(
        nn.Linear(2, 128),
        nn.ReLU(),
        nn.Linear(128, 2),
    )


def create_deep_model():
    modules_dict = OrderedDict()
    previous_size = 2
    for layer_size in [4, 8, 16, 32, 64, 2]:
        modules_dict["fc_{}".format(layer_size)] = nn.Linear(previous_size, layer_size)
        modules_dict["relu_{}".format(layer_size)] = nn.ReLU()
        previous_size = layer_size
    del modules_dict["relu_2"]
    return nn.Sequential(modules_dict)


if __name__ == '__main__':
    data = generate_learning_data()
    parameters = dict(
        steps=250,
        eta=0.1,
        minibatch_size=100
    )

    def initialization_experiment(model_function, init_std=None):
        model = model_function()
        if init_std is not None:
            with torch.no_grad():
                for param in model.parameters():
                    param.normal_(0, init_std)

        return Trainer(model, data, parameters, nn.CrossEntropyLoss()).train()

    def display(result):
        print(
            "Function {param_combination[model_function]}, initialization_param {param_combination[init_std]} : average {avg:.2%} ons {iterations} xps (std {std:.2%})".format(**result)
        )

    results = Experimenter(initialization_experiment, display).experiment(
        {'model_function': [create_shallow_model, create_deep_model],
         'init_std': [None, 0.001, 0.01, 0.1, 1, 10]},
        iterations=10)
