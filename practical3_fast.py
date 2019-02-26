# coding: utf-8
import time
import torch
import dlc_practical_prologue as prologue


def sigma(x):
    return torch.tanh(x)


def dsigma(x):
    return 1 / torch.pow(torch.cosh(x), 2)


def loss(v, t):
    return ((v-t) * (v-t)).sum(dim=1)


def dloss(v, t):
    return 2 * (t - v)


def forward_pass(w1, b1, w2, b2, x):
    s1 = x @ w1.t() + b1
    x1 = sigma(s1)
    s2 = x1 @ w2.t() + b2
    x2 = sigma(s2)
    return x, s1, x1, s2, x2


def backward_pass(w1, b1, w2, b2,
                  t,
                  x, s1, x1, s2, x2):
    
    # Activations derivatives
    dl_dx2 = dloss(x2, t)
    dl_ds2 = dl_dx2 * dsigma(s2)
    dl_dx1 = dl_ds2 @ w2
    dl_ds1 = dl_dx1 * dsigma(s1)

    # Weights derivatives
    dl_db1 = torch.sum(dl_ds1, 0)
    dl_db2 = torch.sum(dl_ds2, 0)
    dl_dw1 = torch.sum(
        torch.matmul(dl_ds1.view(dl_ds1.size(0), -1, 1), x.view(x.size(0), 1, -1)),
        0
    )
    dl_dw2 = torch.sum(
        torch.matmul(dl_ds2.view(dl_ds2.size(0), -1, 1), x1.view(x1.size(0), 1, -1)),
        0
    )

    return dl_dw1, dl_db1, dl_dw2, dl_db2


def train_network(zeta, epsilon, eta_const):
    # Load data form prologue
    train_data, train_target, test_data, test_target = prologue.load_data(
        normalize=True,
        one_hot_labels=True)
    train_target *= zeta
    eta = eta_const / train_data.size(0)

    # init weights
    w1 = torch.empty([50, 784]).normal_(0, epsilon)
    b1 = torch.empty([50]).normal_(0, epsilon)
    w2 = torch.empty([10, 50]).normal_(0, epsilon)
    b2 = torch.empty([10]).normal_(0, epsilon)

    training_start = time.perf_counter()
    for step in range(300):

        # init accumulators
        dl_dw2 = torch.empty([10, 50]).fill_(0)
        dl_db2 = torch.empty(10).fill_(0)
        dl_dw1 = torch.empty([50, 784]).fill_(0)
        dl_db1 = torch.empty(50).fill_(0)

        # run passes on every example
        x, s1, x1, s2, x2 = forward_pass(w1, b1, w2, b2, train_data)
        dl_dw1, dl_db1, dl_dw2, dl_db2 = backward_pass(
            w1, b1, w2, b2,
            train_target,
            x, s1, x1, s2, x2)

        # update weights
        w1 += eta * dl_dw1
        w2 += eta * dl_dw2
        b1 += eta * dl_db1
        b2 += eta * dl_db2

        # compute training loss, training error
        _, _, _, _, v = forward_pass(w1, b1, w2, b2, train_data)
        tr_loss = torch.sum(loss(v, train_target))
        tr_error = torch.nonzero(torch.argmax(train_target, dim=1) - torch.argmax(v, dim=1)).size(0)

        # compute test error
        _, _, _, _, v = forward_pass(w1, b1, w2, b2, test_data)
        test_error = torch.nonzero(torch.argmax(test_target, dim=1) - torch.argmax(v, dim=1)).size(0)

        # print step data
        print("Step {} : Loss {}, error {} %, test error: {} %, elapsed time: {}s".format(
            step,
            tr_loss,
            100.0 * tr_error / train_data.size(0),
            100.0 * test_error / test_data.size(0),
            int(time.perf_counter() - training_start)
        ))


if __name__ == '__main__':
    train_network(0.9, 10**-6, 0.1)
