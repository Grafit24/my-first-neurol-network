import os
from typing import List, Tuple
import pickle

import numpy as np
import numpy.random as rand

from loader import load_data

train_data, valid_data = load_data("train.csv", valid=True)
# test_data = load_data("test.csv")

class Network(object):
    def __init__(self, sizes: List[int], random_state=None):
        self.nlayers = len(sizes) 
        self.sizes = sizes

        self.weights = [rand.randn(sizes[i+1], sizes[i]) 
                        for i in range(self.nlayers-1)]
        self.biases = [rand.randn(sizes[i+1]) 
                       for i in range(self.nlayers-1)]
    
    def SGD(self, training_data, eta, epochs, mini_bunch_size, test_data=None):
        """Используем stochastic gradient descent для тренировки нейросети.
        Prameters
        ---------
        training data : List[Tuple[(X, y)]]
            это лист вида описанного в load_data
        eta : int
            learning rate
        epochs : int
        mini_bunch_size : int
        test_data : List[Tuple[(X, y)]]
            Если None ,тогда не пишет точность сети 
            после каждой эпохи. В обратном случае 
            соответсвенно пишет.
        """
        for epoch in range(epochs):
            rand.shuffle(training_data)
            
            n_train_d = len(training_data)
            mini_bunches = [training_data[k:(mini_bunch_size+k)] 
                            for k in range(0, n_train_d, mini_bunch_size)]

            # Обновляем веса по mini-bunch
            for mini_bunch in mini_bunches:
                self.update_mini_bunch(mini_bunch, eta)
            
            # Пишем точность сети.
            text = "Epoch {0} complete ".format(epoch+1)
            if test_data != None:
                evaluate_result = self.evaluate(test_data)
                n_test_data = len(test_data)
                text += "with accuracity {0}/{1} = {2}".format(
                    evaluate_result, n_test_data, 
                    evaluate_result/n_test_data
                    )
            print(text)
        
    def update_mini_bunch(self, mini_bunch: list, eta: int):
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        for x, y in mini_bunch:
            delta_nabla_w, delta_nabla_b = self.backprop(x, y)
            nabla_w = [nw+dnw for nw, dnw in 
                        zip(nabla_w, delta_nabla_w)]
            nabla_b = [nb+dnb for nb, dnb in 
                        zip(nabla_b, delta_nabla_b)]
            
        mb_size = len(mini_bunch)
        self.weights = [w-(eta/mb_size)*nw 
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/mb_size)*nb 
                        for b, nb in zip(self.biases, nabla_b)]
        
    def feedforward(self, a):
        for w, b in zip(self.weights, self.biases):
            a = sigmoid(np.dot(w, a)+b)
        return a
        
    def backprop(self, x, y):
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        nabla_b = [np.zeros(b.shape) for b in self.biases]

        activation = x
        activations = [x]
        zs = []
        # feedforward
        for w, b in zip(self.weights, self.biases):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
            
        # output layer
        cd = self.cost_derivative(activations[-1], y)
        delta = cd*sigmoid_prime(zs[-1])
        nabla_w[-1] = np.dot(delta.reshape(-1, 1), activations[-2].reshape(1, -1))
        nabla_b[-1] = delta

        # hidden layeres
        for l in range(2, self.nlayers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta)*sp
            nabla_w[-l] = np.dot(delta.reshape(-1, 1), activations[-l-1].reshape(1, -1))
            nabla_b[-l] = delta

        return (nabla_w, nabla_b)
        
    def evaluate(self, test_data):
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y.argmax()) for (x, y) in test_results)


    def cost_derivative(self, output, y):
        return output-y


def sigmoid(x):
    return 1./(1. + np.exp(-x))


def sigmoid_prime(x):
    sigm = sigmoid(x)
    return (1. - sigm)*sigm


net = Network([784, 16, 10])
net.SGD(train_data, 3., 30, 30, test_data=valid_data)



# predict = pd.DataFrame(nn.predict(np.array(test_data)))
# predict.set_index(pd.Series(test_data.index, name='ImageId')+1, inplace=True)
# predict.columns = ['Label']
# predict['Label'] = predict['Label'].astype(int)
# predict.to_csv('multiPerceptron_test.csv')
                        
