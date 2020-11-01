# %%
import numpy as np
import pandas as pd
from functions import sigmoid, d_sigmoid

train_data = pd.read_csv('data/train.csv')
X_train = np.array(train_data.iloc[:, 1:])
y_train = np.array(train_data.iloc[:, 0])


class P1L:
    """Perceptron with one layer - output layer. Uses sigmoid function"""
    def __init__(self, n_input, n_output):
        self.structure = (n_input, n_output)
        self.alpha = .001
    
    def feedforward(self, x: "vector")-> "vector":
        return sigmoid(np.dot(self.W, x), self.alpha)

    def backpropogation(self, X, y, steps, learning_rate=.1):
        # init W_0
        n_input = self.structure[0]
        n_output = self.structure[1]
        self.W = np.random.random(size=(n_output, n_input))

        for d in range(steps):
            x = X[d]
            out = self.feedforward(x)
            # yt definition
            y_empty = [0]*n_output
            y_empty[y[d]] += 1
            y_true = np.array(y_empty)
            # calculate derivative of $$C=(yt-y)^2$$
            dC_da = 2*(out-y_true)
            # calculate derivative of dsigma(z)
            dsigma = np.zeros(n_output)
            for j in range(n_output):
                z = np.dot(self.W[j, :], x)
                dsigma[j] = d_sigmoid(z, self.alpha)
            
            # calculate dC/dwij
            E = dsigma*dC_da
            dgradient = E.reshape(n_output, 1)*x

            # update weights
            self.W -= learning_rate*dgradient
  
        return self.W
    
    def train_test(self, X, y):
        n_train = int(X.shape[0]*.8-1)
        # train
        X_train = np.array(X.iloc[:n_train, :])
        y_train = np.array(y.iloc[:n_train])
        # test
        X_test = np.array(X.iloc[n_train:, :])
        y_test = np.array(y.iloc[n_train:])

        n_input = self.structure[0]
        n_output = self.structure[1]
        self.backpropogation(X_train, y_train, X_train.shape[0])

        # make test
        true_answer = 0
        for x, y in zip(X_test, y_test):
            true_answer += self.feedforward(x).argmax() == y
        
        return true_answer/y_test.shape[0]
    

    def predict(self, X):
        predicted =  np.zeros(X.shape[0])
        for i, x in enumerate(X):
            predicted[i] = int(self.feedforward(x).argmax())
        return predicted


test_data = pd.read_csv('data/test.csv')
model = P1L(28*28, 10)
model.backpropogation(X_train, y_train, X_train.shape[0])
predict = pd.DataFrame(model.predict(np.array(test_data)))
predict.set_index(pd.Series(test_data.index, name='ImageId')+1, inplace=True)
predict.columns = ['Label']
predict['Label'] = predict['Label'].astype(int)
predict.to_csv('p1l_v1.csv')
# %%
