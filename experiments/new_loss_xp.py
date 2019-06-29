import torch


fixed_parameters = dict(
    steps=20,
    optimizer_class=torch.optim.Adam,
    optimizer_params=dict(),
    minibatch_size=100,
)

iterations = 40

learning_curve_points = 20

variable_parameters = {
    'loss_function': ["pow3", "exp", "crossE", "crossEclose", "crossEexact", "pow2"],
    'dataset': ['mnist', 'cifar'],
    'dataset_size': ['normal', 'full'],
}
