
from typing import List
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from functions import d_sigmoid, sigmoid

train_data = pd.read_csv('data/train.csv')
test_data = pd.read_csv('data/test.csv')


class Network(object):
    def __init__(self, n_input_layer: int, n_output_layer: int, n_hidden_layer: List[int]):
        self.structure = [n_input_layer, *n_hidden_layer, n_output_layer]
        self.W = None


    def feedforward(self, x: "vector", func=sigmoid)-> "vector":
        # a0 = x
        a = x
        a_layers = [a]
        for l in range(len(self.W)):
            a = func(np.dot(self.W[l], a))
            a_layers.append(a.copy())
        return a_layers


    def backprop(self, X, y, steps, learning_rate=.00001):
        for d in range(steps):
            a_layers = self.feedforward(X[d])
            out = a_layers[-1]
            # reverse for the cuherant
            W = self.W.copy()[::-1]

            # Definite devirate C by a(out),
            # where out is index of last layer.
            y_empty = [0]*W[0].shape[0]
            y_empty[y[d]] += 1
            y_true = np.array(y_empty)
            dcost = 2*(out-y_true)

            # Output layer correction
            w_output = W[0].copy()
            # if a(l) is output layer ,then a_preout is a(l-1).
            a_preout = a_layers[-2]
            # product of dz/dw and da/dz
            a_preout_T = a_preout.reshape(a_preout.shape[0], 1)
            product_dev = a_preout_T*d_sigmoid(np.dot(w_output, a_preout))
            dC_dwout = product_dev*dcost
            w_output -= learning_rate*dC_dwout.T
            W[0] = w_output

            # Hidden layer correction
            n_wlayers = len(self.W)
            output_layer = W[0]


            Z = dict()
            def calculate_d_sigmoid_z(w, a, l):
                """Calcualte z = der_sigmoid(dot(w, a)) 
                ,where 
                der_sigmoid - derivative of sigmoid function
                w - weight of neuron, 
                a - related neurons with this neuron.
                if z wasn't calculate before. 
                """
                # nonlocal Z

                # if not (l in Z.keys()):
                #     Z[l] = d_sigmoid(np.dot(w, a))
                z = d_sigmoid(np.dot(w, a))

                return z
            

            def get_dC_dai(l):
                l -= 1

                if l == 0:
                    w = W[l]
                    # output layer is last.
                    dsigm_z = calculate_d_sigmoid_z(output_layer, a_preout, l)
                    return np.dot(w.T, dsigm_z*dcost)
                else:
                    w = W[l]
                    a = a_hidden_layers[l+1]
                    dsigm_z = calculate_d_sigmoid_z(w, a, l)
                    # np.dot need for the summarize elements.
                    return np.dot(w.T, dsigm_z*get_dC_dai(l))


            a_hidden_layers = a_layers[::-1]
            for l in range(1, n_wlayers):
                a_0 = a_hidden_layers[l+1]
                dsigm_z_0 = calculate_d_sigmoid_z(W[l], a_0, l)
                dC_dai = get_dC_dai(l)
                # set W
                dC_dwij = a_0.reshape(a_0.shape[0], 1)*(dsigm_z_0*dC_dai)
                W[l] -= learning_rate*dC_dwij.T

            self.W = W[::-1]
        

    def predict(self, X):
        predicted =  np.zeros(X.shape[0])
        for i, x in enumerate(X):
            predicted[i] = int(self.feedforward(x)[-1].argmax())
        return predicted
    

    def train(self, train_data, epochs):
        # init W
        np.random.seed(100)
        rand = np.random.random
        self.W = [rand(size=(self.structure[i+1], self.structure[i])) 
                  for i in range(len(self.structure)-1)
                  ]

        for epoch in range(epochs):
            train_data = np.array(train_data)
            np.random.shuffle(train_data)
            y, X = (train_data[:, 0], train_data[:, 1:])
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2)
        
            self.backprop(X_train, y_train, X_train.shape[0])
            print(f'Epoch {epoch} complete!')

            # make test
            true_answer = 0
            for x, y in zip(X_test, y_test):
                true_answer += self.feedforward(x)[-1].argmax() == y
            print(f"Scoring of {epoch} is {true_answer/y_test.shape[0]}")
        
        return "Complete"


test_data = pd.read_csv('data/test.csv')
nn = Network(28*28, 10, (8, 8))
# nn.backprop(X_train, y_train, X_train.shape[0])
print(nn.train(train_data, 30))





# predict = pd.DataFrame(nn.predict(np.array(test_data)))
# predict.set_index(pd.Series(test_data.index, name='ImageId')+1, inplace=True)
# predict.columns = ['Label']
# predict['Label'] = predict['Label'].astype(int)
# predict.to_csv('multiPerceptron_test.csv')
                        
