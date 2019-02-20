#!/usr/bin/env python
# coding: utf-8

import torch
import time


def exo1():
    dalle_step = 5
    dalle_nb = 3
    matrix_size = dalle_step * dalle_nb + 3

    x = torch.full([matrix_size, matrix_size], 1)

    twos_line = torch.empty(0).set_(x.storage(),
                                    storage_offset=matrix_size,
                                    size=(dalle_nb+1, matrix_size),
                                    stride=(dalle_step * matrix_size, 1))

    twos_line.fill_(2)

    twos_column = torch.empty(0).set_(x.storage(),
                                      storage_offset=1,
                                      size=(matrix_size, dalle_nb+1),
                                      stride=(matrix_size, dalle_step))

    twos_column.fill_(2)
    threes_matrices = torch.empty(0).set_(x.storage(),
                                          storage_offset=(int((dalle_step + 1)/2)) * (1 + matrix_size),
                                          size=(dalle_nb, dalle_nb, 2, 2),
                                          stride=(matrix_size*dalle_step, dalle_step, matrix_size, 1))

    threes_matrices.fill_(3)

    print(x)


def exo2():
    m = torch.empty(20, 20)
    m.normal_()
    d = torch.diag(torch.FloatTensor(range(1, 21)))
    print(m)
    print(d)
    eigs = torch.eig(torch.inverse(m) @ d @ m)
    print(eigs)


def exo3():
    size = 5000
    m1 = torch.empty([size, size]).normal_()
    m2 = torch.empty([size, size]).normal_()
    nb_ops_mul = size**3
    t1 = time.perf_counter()
    res = torch.mm(m1, m2)
    t2 = time.perf_counter()
    print("GigaFlops: {}".format(nb_ops_mul/((t2-t1) * 10**9)))
    return res


def mul_row(t):
    assert len(t.size()) == 2
    r = torch.empty(t.size())
    for i in range(t.size(0)):
        for j in range(t.size(1)):
            r[i][j] = t[i][j] * (i+1)
    return r


def mul_row_fast(t):
    assert len(t.size()) == 2
    fact = torch.arange(1.0, t.size(0)+1).view(t.size(0), 1)
    return fact * t


test_t = torch.empty(1000, 400).normal_()

t1 = time.perf_counter()
m1 = mul_row(test_t)
t2 = time.perf_counter()
m2 = mul_row_fast(test_t)
t3 = time.perf_counter()
print("Correctness: {}".format(torch.norm(m2-m1)))
print("T1 : {}, T2: {}, ratio: {}".format(t2 - t1, t3-t2, (t2-t1)/(t3-t2)))
