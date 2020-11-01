from typing import List
import numpy as np
import pandas as pd

train_data = pd.read_csv('data/train.csv')
X_train = np.array(train_data.iloc[:, 1:])
y_train = np.array(train_data.iloc[:, 0])

# input_neurons = 784
# layer_1_neurons = 1
# output_neurons = 10

class NN:
    def __init__(self, n_input_layer: int, n_output_layer: int, n_hidden_layer: List[int]):
        self.structure = [n_input_layer, *n_hidden_layer, n_output_layer]
        self.W = None


    def feedforward(self, x: "vector", func=sigmoid)-> "vector":
        # a0 = x
        a = x
        for l in range(len(self.W)):
            a = func(np.dot(self.W[l], a))
        return a


    def SGM(self, learning_rate: float, X: "Matrix", y: "vector", steps: int)-> None:
        rand = np.random.random
        # init random w
        self.W = [rand(size=(self.structure[i+1], self.structure[i])) 
                  for i in range(len(self.structure)-1)
                  ]

        for i in range(steps):
            out = self.feedforward(X_train[0])
            

    

NN(784, 10, (8, 8)).SGM(1, 1, 1, 1)
