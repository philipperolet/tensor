# coding: utf-8
import time
import torch
import torch.nn as F
import dlc_practical_prologue as prologue

# Initialize data
torch.manual_seed(0)

train_data, train_target, test_data, test_target = prologue.load_data(
    normalize=True,
    one_hot_labels=True)

print("Gradient Required? {}".format(train_data.requires_grad))

eta_const = 0.1
epochs = 1000
eta = eta_const / train_data.size(0)
zeta = 0.9
train_target *= zeta

# Define model & loss
model = F.Sequential(
    F.Linear(784, 50),
    F.Tanh(),
    F.Linear(50, 10),
    F.Tanh(),
)
loss = F.MSELoss(reduction='sum')

# Train
training_start = time.perf_counter()
optimizer = torch.optim.SGD(model.parameters(), lr=eta)

for step in range(epochs):
    optimizer.zero_grad()
    output = model(train_data)
    tr_loss = loss(output, train_target)
    tr_loss.backward()
    optimizer.step()

    with torch.no_grad():
        tr_error = torch.nonzero(
            torch.argmax(train_target, dim=1) - torch.argmax(output, dim=1)
        ).size(0)

        test_error = torch.nonzero(
            torch.argmax(test_target, dim=1) - torch.argmax(model(test_data), dim=1)
        ).size(0)

        print("Step {} : Loss {}, error {} %, test error: {} %, elapsed time: {}s".format(
            step,
            tr_loss,
            100.0 * tr_error / train_data.size(0),
            100.0 * test_error / test_data.size(0),
            int(time.perf_counter() - training_start)
        ))
