import torch


fixed_parameters = dict(
    steps=20,
    optimizer_class=torch.optim.Adam,
    optimizer_params=dict(),
    minibatch_size=100,
)

iterations = 20

learning_curve_points = 20

variable_parameters = {
    'conv1_channels': [16, 32, 64],
    'conv1_kernel_size': [2, 5, 8],
    'conv2_channels': [32, 64, 128],
    'conv2_kernel_size': [4, 5, 6],
    'dataset': ['mnist', 'cifar'],
    'dataset_size': ['normal', 'full'],
}
