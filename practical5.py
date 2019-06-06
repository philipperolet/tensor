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


def generate_learning_data(normalize=True, training_normalization=True):
    train_data, train_target = generate_disc_set(1000)
    mean, std = train_data.mean(dim=0), train_data.std(dim=0)
    test_data, test_target = generate_disc_set(1000)
    if normalize:
        train_data.sub_(mean).div_(std)
        if not(training_normalization):
            mean, std = test_data.mean(dim=0), test_data.std(dim=0)
        test_data.sub_(mean).div_(std)

    return dict(
        training_input=train_data,
        training_target=train_target,
        test_input=test_data,
        test_target=test_target,
    )


def create_shallow_model(last_relu=False):
    if last_relu:
        return nn.Sequential(
            nn.Linear(2, 128),
            nn.ReLU(),
            nn.Linear(128, 2),
            nn.ReLU(),
        )
    return nn.Sequential(
        nn.Linear(2, 128),
        nn.ReLU(),
        nn.Linear(128, 2),
    )


def create_deep_model(last_relu=False):
    modules_dict = OrderedDict()
    previous_size = 2
    for layer_size in [4, 8, 16, 32, 64, 2]:
        modules_dict["fc_{}".format(layer_size)] = nn.Linear(previous_size, layer_size)
        modules_dict["relu_{}".format(layer_size)] = nn.ReLU()
        previous_size = layer_size
    if not(last_relu):
        del modules_dict["relu_2"]
    return nn.Sequential(modules_dict)


if __name__ == '__main__':
    parameters = dict(
        steps=250,
        optimizer_class=torch.optim.SGD,
        optimizer_params=dict(lr=0.1),
        minibatch_size=100
    )

    def initialization_experiment(model_function,
                                  last_relu,
                                  normalize,
                                  training_normalization,
                                  init_std=None):
        data = generate_learning_data(normalize, training_normalization)
        model = model_function(last_relu)
        if init_std is not None:
            with torch.no_grad():
                for param in model.parameters():
                    param.normal_(0, init_std)

        return Trainer(model, data, parameters, nn.CrossEntropyLoss()).train()

    def display(result):
        print((
            "Function {param_combination[model_function].__name__}, "
            "normalization {param_combination[normalize]}, "
            "training_normalization {param_combination[training_normalization]}, "
            "last_relu {param_combination[last_relu]} : "
            "average {avg:.2%} on {iterations} xps (std {std:.2%})"
        ).format(**result))

    results = Experimenter(initialization_experiment, display).experiment(
        {'model_function': [create_shallow_model, create_deep_model],
         'last_relu': [True, False],
         'normalize': [True, False],
         'training_normalization': [True, False]},
        iterations=10)
