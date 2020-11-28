
from typing import List
import numpy as np
import numpy.random as rand
import pandas as pd
from sklearn.model_selection import train_test_split

train_data = pd.read_csv('data/train.csv')
test_data = pd.read_csv('data/test.csv')


class Network(object):
    def __init__(self, sizes: List[int]):
        """Get sizes of neural network"""
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.weights = [rand.random() for i in sizes]
        pass

    
    def SGD(self):
        pass
    
    
    def feetforward(self):
        pass


    def backprop(self):
        pass


def sigmoid(x):
    return 1./(1. + np.exp(-x))


def sigmoid_prime(x):
    sigm = sigmoid(x)
    return (1. - sigm)*sigm


test_data = pd.read_csv('data/test.csv')
nn = Network(28*28, 10, (8, 8))
# nn.backprop(X_train, y_train, X_train.shape[0])
print(nn.train(train_data, 30))





# predict = pd.DataFrame(nn.predict(np.array(test_data)))
# predict.set_index(pd.Series(test_data.index, name='ImageId')+1, inplace=True)
# predict.columns = ['Label']
# predict['Label'] = predict['Label'].astype(int)
# predict.to_csv('multiPerceptron_test.csv')
                        
