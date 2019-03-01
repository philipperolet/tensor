import time
import torch
from torch.autograd import Variable
from torch import nn
from torch.nn import functional as F

import dlc_practical_prologue as prologue

train_input, train_target, test_input, test_target = \
    prologue.load_data(one_hot_labels = True, normalize = True, flatten = False)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5)
        self.fc1 = nn.Linear(256, 200)
        self.fc2 = nn.Linear(200, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), kernel_size=3, stride=3))
        x = F.relu(F.max_pool2d(self.conv2(x), kernel_size=2, stride=2))
        x = F.relu(self.fc1(x.view(-1, 256)))
        x = self.fc2(x)
        return x


def _compute_errors(model, input, target, minibatch_size):
    '''Computes training error and test error'''
    input_batch = input
    target_batch = target
#    print(torch.argmax(model(input_batch), dim=1))
#    print(torch.argmax(target_batch, dim=1))
#    print(target_batch)
    nb_errors = torch.nonzero(
        torch.argmax(target_batch, dim=1)
        - torch.argmax(model(input_batch), dim=1)
    ).size(0)
#    print(nb_errors)
    return float(nb_errors)/target_batch.size(0)

train_input, train_target = Variable(train_input), Variable(train_target)

model, criterion = Net(), nn.MSELoss()
eta, mini_batch_size = 1e-1, prologue.args.batchsize

training_start = time.perf_counter()
for e in range(0, 25):
    sum_loss = 0
    # We do this with mini-batches
    for b in range(0, train_input.size(0), mini_batch_size):
        output = model(train_input.narrow(0, b, mini_batch_size))
        loss = criterion(output, train_target.narrow(0, b, mini_batch_size))
        sum_loss = sum_loss + loss.item()
        model.zero_grad()
        loss.backward()
        for p in model.parameters():
            p.data.sub_(eta * p.grad.data)
    print(e, sum_loss)

for _ in range(1):
    print("Test error {}, time {}s".format(
        _compute_errors(model, test_input, test_target, mini_batch_size) * 100.0,
        int(time.perf_counter() - training_start)
    ))
