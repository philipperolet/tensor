import torch
import torch.nn as mods
import torch.nn.functional as F


class CustomNet(torch.nn.Module):
    """Customizable CNN

    Args:
    is_cifar -- should be set to True if using CIFAR rather than MNIST
    hidden_units -- size of the last hidden layer
    exp_factor -- to experiment on "pre-CE" normalization
    """
    IMAGE_SIZE = 32  # Size of image edge in pixels

    loss_functions = {
        "pow3rd": lambda x: F.log_softmax(x.pow(0.33)),
        "pow3": lambda x: F.log_softmax(x * x * x),
        "exp": lambda x: F.log_softmax(x.exp()),
        "crossE": lambda x: F.log_softmax(x),
        "crossEexact": lambda x: F.log_softmax(x.pow(1.0)),
        "crossEclose": lambda x: F.log_softmax(x.sign() * x.abs().pow(1.02)),
        "pow2": lambda x: F.log_softmax(x * x),
        "true": lambda x: torch.sign(x - x.max()),
        "almost": lambda x: x - x.max(),
        "bad": lambda x: x,
    }

    activation_functions = {
        "ReLU": F.relu,
        "tanh": F.tanh,
        "negsqr": lambda x: torch.mul(x.sign(), x.abs().sqrt()),
        "logweird": lambda x: x.div(2) * (x.pow(2)).log(),
        "LeakyReLU": F.leaky_relu,
        "ReLulu": lambda x: x.div(4) + F.hardtanh(x),
    }

    def __init__(self,
                 is_cifar,
                 hidden_units=200,
                 loss_function="crossE",
                 activation_function="ReLU",
                 conv1_channels=32,
                 conv1_kernel_size=5,
                 pool1_kernel_size=3,
                 conv2_channels=64,
                 conv2_kernel_size=5,
                 pool2_kernel_size=2,
                 **kwargs):

        super(CustomNet, self).__init__()

        # If cifar dataset, 3 channels, if mnist only 1
        input_channels = 3 if is_cifar else 1

        # Setup first convolution layer
        self.conv1 = mods.Conv2d(input_channels, conv1_channels, kernel_size=conv1_kernel_size)
        self.pool1 = mods.MaxPool2d(pool1_kernel_size)
        # assert (self.IMAGE_SIZE - conv1_kernel_size) % pool1_kernel_size == 0
        activation1_edge_size = (self.IMAGE_SIZE - conv1_kernel_size) / pool1_kernel_size

        # Setup second convolution layer
        self.conv2 = mods.Conv2d(conv1_channels, conv2_channels, kernel_size=conv2_kernel_size)
        self.pool2 = mods.MaxPool2d(pool2_kernel_size)
        # assert (activation1_edge_size - conv2_kernel_size) % pool2_kernel_size == 0
        activation2_edge_size = (activation1_edge_size - conv2_kernel_size) / pool2_kernel_size

        # Setup linear layers
        self.fc1 = mods.Linear(int(activation2_edge_size**2 * conv2_channels), hidden_units)
        self.fc2 = mods.Linear(hidden_units, 10)
        self.loss_function = CustomNet.loss_functions[loss_function]
        self.activation_function = CustomNet.activation_functions[activation_function]

    def forward(self, x):
        x = self.activation_function(self.pool1(self.conv1(x)))
        x = self.activation_function(self.pool2(self.conv2(x)))
        x = x.view(-1, self.fc1.in_features)
        x = self.activation_function(self.fc1(x))
        x = self.fc2(x)
        x = self.loss_function(x)
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
