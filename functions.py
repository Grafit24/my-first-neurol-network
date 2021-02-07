import numpy as np

class Sigmoid:
    @staticmethod
    def f(x):
        y = 1./(1. + np.exp(-x))
        return y
    
    @staticmethod
    def df(x):
        sigm = Sigmoid.f(x)
        y = (1. - sigm)*sigm
        return y


class Tanh:
    @staticmethod
    def f(x):
        y = np.tanh(x)
        return y
    
    @staticmethod
    def df(x):
        tanh = Tanh.f(x)
        y = 1 - tanh**2
        return y


class ReLU:
    @staticmethod
    def f(x):
        y = np.zeros_like(x)
        condition = x >= 0
        # x >= 0
        y[condition] = x[condition]
        # x < 0
        y[np.logical_not(condition)] = 0
        return y
    
    @staticmethod
    def df(x):
        y = np.zeros_like(x)
        condition = x >= 0
        # x >= 0
        y[condition] = 1
        # x < 0
        y[np.logical_not(condition)] = 0
        return y
