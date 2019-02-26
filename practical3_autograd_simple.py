# coding: utf-8
import time
import torch
import dlc_practical_prologue as prologue


def sigma(x):
    return torch.tanh(x)


def dsigma(x):
    return 1 / torch.pow(torch.cosh(x), 2)


def loss(v, t):
    return torch.dot(v-t, v-t)


def dloss(v, t):
    return 2 * (t - v)


def forward_pass(w1, b1, w2, b2, x):
    s1 = x @ w1.t() + b1
    x1 = sigma(s1)
    s2 = x1 @ w2.t() + b2
    x2 = sigma(s2)
    return x, s1, x1, s2, x2

def train_network(zeta, epsilon, eta_const):
    torch.manual_seed(0)
    # Load data form prologue
    train_data, train_target, test_data, test_target = prologue.load_data(
        normalize=True,
        one_hot_labels=True)
    train_target *= zeta
    eta = eta_const / train_data.size(0)

    # init weights
    w1 = torch.empty([50, 784]).normal_(0, epsilon).requires_grad_()
    b1 = torch.empty([50]).normal_(0, epsilon).requires_grad_()
    w2 = torch.empty([10, 50]).normal_(0, epsilon).requires_grad_()
    b2 = torch.empty([10]).normal_(0, epsilon).requires_grad_()

    training_start = time.perf_counter()
    for step in range(300):
        w1.grad, w2.grad, b1.grad, b2.grad = None, None, None, None
        # run passes on every example
        for x, t in zip(train_data, train_target):
            x, s1, x1, s2, x2 = forward_pass(w1, b1, w2, b2, x)
            loss(x2, t).backward()
        
        # update weights
        with torch.no_grad():
            w1 -= eta * w1.grad
            w2 -= eta * w2.grad
            b1 -= eta * b1.grad
            b2 -= eta * b2.grad

        # compute training loss, training error
        tr_loss = 0
        tr_error = 0
        for x, t in zip(train_data, train_target):
            _, _, _, _, v = forward_pass(w1, b1, w2, b2, x)
            tr_loss += loss(v, t)
            if (torch.argmax(t) != torch.argmax(v)):
                tr_error += 1

        # compute test error
        test_loss = 0
        test_error = 0
        for x, t in zip(test_data, test_target):
            _, _, _, _, v = forward_pass(w1, b1, w2, b2, x)
            test_loss += loss(v, t)
            if (torch.argmax(t) != torch.argmax(v)):
                test_error += 1

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
