import unittest
import torch
import math

def sigma(x):
    return torch.tanh(x)

class Practical3Test(unittest.TestCase):

    def test_sigma(self):
        self.assertTrue(torch.all(torch.eq(
            sigma(torch.Tensor([0, 0])),
            torch.Tensor([2]).fill_(0)
        )))
        self.assertTrue(torch.all(torch.eq(
            sigma(torch.Tensor([0, 1])),
            torch.Tensor([0, math.tanh(1)])
        )))

    def test_point_multiplication(self):
        self.assertTrue(torch.all(torch.eq(
            torch.Tensor([2, 3, 3]) * torch.Tensor([2, 1, 3]),
            torch.Tensor([4, 3, 9])
        )))

    def test_point_inverse(self):
        self.assertTrue(torch.all(torch.eq(
            1 / torch.Tensor([[2.0, 4],  [8.0, 10.0]]),
            torch.Tensor([[0.5, 0.25], [0.125, 0.1]])
        )))

    def test_sum_with_weight_vector(self):
        '''if x is "normal as dimension", aka dimension d (3 here) are columns
        and w is multilayered (each line is a d-size weight vector)
        then the correct way to multiply is x.wT
        '''
        x = torch.tensor([1, 2, 3])
        w = torch.tensor([[1, 1, 1], [3, -3, 0]])
        b = torch.tensor([1, 3])
        b += x @ w.t()
        self.assertTrue(torch.all(torch.eq(
            b,
            torch.tensor([7, 0]),
        )))

    def test_expansion_with_add(self):
        """When doing A [n, k] + B [k] tensor operation, B is expanded to match [n, k] and added"""
        a = torch.tensor([[1, 2, 3], [4, 5, 6]])
        b = torch.tensor([1, 0, -1])
        self.assertTrue(torch.all(torch.eq(
            a + b,
            torch.tensor([[2, 2, 2], [5, 5, 5]]),
        )))
        
