import numpy as np

def sigmoid(x, alpha=.01):
    return 1/(1 + np.exp(-alpha*x))


def d_sigmoid(x, alpha=.01):
    sigm = sigmoid(x, alpha)
    return (1 - sigm)*sigm


def ReLU(x):
    return max([0, x])
