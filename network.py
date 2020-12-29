"""Многослойный перцептрон

обучающющийся через стохастический крадиентный спуск. Для вычисления 
градиента используются алгоритм обратного распространения ошибки.

Note. Весь код ниже написан с целью понять ,что творится под "капотом" 
нейросети. Писался код с опорой на книгу "Neural Networks and Deep Learning" 
Michael Nielsen'а.
"""
from typing import List, Tuple

import numpy as np
import numpy.random as rand

from loader import load_train_data, load_test_data, export_data


class Network(object):
    """Многослойный перцептрон с сигмоидной функцией активации.
    
    Attributes
    ----------
    sizes : List[int]
        Размер сети каждое число соответсвует кол-ву нейронов 
        в соответсвующим слое. Первый элемент списка соответсвует 
        кол-ву нейронов в входном слое ,а последний в выходном.

    random_state : int
        Число для установления начальных условий генератора случайных чисел.
        Если None ,то никак не влияет на генерацию.
        Note. Используется для тестирования разных параметров ,чтобы исключить 
        рандом.
    
    Methods
    -------
    SGD(training_data, eta, epochs, mini_batch_size, test_data=None)
        Обучает нейросеть по алгоритму stochastic gradient descent.

    update_mini_batch(mini_batch, eta)
        Обновляет веса и смещения по примерам из mini_batch.

    backprop(x, y)
        Алгоритм обратного распространения ошибки.

    feedforward(a)
        Прямой проход через нейросеть. 

    evaluate(test_data)
        Считает точность сети на тестовых данных(test_data).

    сost_deveriate(output, y)
        Возвращает вектор частный производных dC/da_output.

    predict(X)
        Определяет наиболее вероятный класс вектора x из множества X.
    """
    def __init__(self, sizes: List[int], random_state=None):
        self.nlayers = len(sizes) 
        self.sizes = sizes
        self.random_state = random_state

        rand.seed(random_state)
        self.weights = [rand.randn(sizes[i+1], sizes[i]) 
                        for i in range(self.nlayers-1)]
        rand.seed(random_state)
        self.biases = [rand.randn(sizes[i+1]) 
                       for i in range(self.nlayers-1)]
    
    def SGD(self, training_data, eta, epochs, mini_batch_size, test_data=None):
        """Обучает нейросеть по алгоритму stochastic gradient descent.

        В кратце суть алгоритма. Ищем наискорейший спуск к точке минимума cost function 
        обновляя для этого веса и смещения используя обратный градиент. 
        Стохастический же алгоритм различен тем ,что в нём мы обновляем веса не по каждому примеру,
        а по усреднённой сумме подмножества всех примеров(по mini-batch'у).
        Note. Это позволяет получить выйгрыш в скорости вычислений. 
        
        Prameters
        ---------
        training data : List[Tuple[(x, y)]]
            это список соостоящий из кортежей первый элемент которых 
            вектор x размера входного слоя и вектор y размера выходного слоя.

        eta : int
            learning rate

        epochs : int

        mini_batch_size : int

        test_data : List[Tuple[(X, y)]]
            Если None ,тогда не пишет точность сети 
            после каждой эпохи. В обратном случае 
            соответсвенно пишет. Должен быть тот же вид,
            что и у training_data.
        """
        for epoch in range(epochs):
            np.random.seed(self.random_state)
            rand.shuffle(training_data)
            
            n_train_d = len(training_data)
            mini_batches = [training_data[k:(mini_batch_size+k)] 
                            for k in range(0, n_train_d, mini_batch_size)]

            # Обновляем веса по mini-bunch
            for mini_batch in mini_batches:
                data = mini_batch
                x = []
                y = []
                for i in data:
                    x.append(i[0])
                    y.append(i[1])

                x = np.array(x).transpose()
                y = np.array(y).transpose()
                self.update_mb(x, y, eta)
            
            # Пишем в консоль точность сети
            text = "Epoch {0} complete ".format(epoch+1)
            if test_data != None:
                evaluate_result = self.evaluate(test_data)
                n_test_data = len(test_data)
                text += "with accuracity {0}/{1} = {2}".format(
                    evaluate_result, n_test_data, 
                    evaluate_result/n_test_data
                    )
            print(text)

    def update_mb(self, x, y, eta):
        """Обновляет веса и смещения по примерам из mini_batch.
        
        Считаем градиенты для всех примеров mini-batch после 
        вычисляем их ср.ариф. умноженное на learning_rate 
        и отнимает от старых значений(тк для спуска нужен обратный градиент).

        Parameters
        ----------
        mini_batch : List[Tuple[x, y]]
            Список вида описанного в функции SGD/training_data.

        eta : float
            learning_rate.
        """
        nabla_w, nabla_b = self.backprop(x, y)

        mb_size = x.shape[1]
        # вернуть self !
        self.weights = [w-(eta/mb_size)*nw 
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/mb_size)*nb 
                        for b, nb in zip(self.biases, nabla_b)]
    
    def backprop(self, x, y):
        """Алгоритм обратного распространения ошибки. Возвращает dC/dw и dC/db."""
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        nabla_b = [np.zeros(b.shape) for b in self.biases]

        activation = x
        activations = [x]
        zs = []
        # feedforward
        for w, b in zip(self.weights, self.biases):
            b = b.reshape(-1, 1)
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
            
        # output layer
        cd = self.cost_derivative(activations[-1], y)
        delta = cd*sigmoid_prime(zs[-1])
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        nabla_b[-1] = delta.sum(axis=1)

        # hidden layeres
        for l in range(2, self.nlayers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta)*sp
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
            nabla_b[-l] = delta.sum(axis=1)

        return (nabla_w, nabla_b)

    def feedforward(self, a: np.ndarray)->np.ndarray:
        """Прямой проход через нейросеть."""
        for w, b in zip(self.weights, self.biases):
            a = sigmoid(np.dot(w, a)+b)
        return a
    
    def matrixbase_feedforward(self, a: np.ndarray)->np.ndarray:
        """Прямой проход через нейросеть."""
        for w, b in zip(self.weights, self.biases):
            a = sigmoid(np.dot(w, a)+b.reshape(-1, 1))
        return a
        
    def evaluate(self, test_data):
        """Считает точность сети на тестовых данных(test_data)"""
        x = [xi for xi, _ in test_data]
        x = np.array(x).transpose()
        y = [yi for _, yi in test_data]
        y = np.array(y).argmax(axis=1)

        test_results = np.argmax(self.matrixbase_feedforward(x), axis=0)

        # Надо будет изменить функцию ,чтобы она возрачала булева да нет.
        return (test_results==y).sum()
    
    def predict(self, X):
        """Определяет наиболее вероятный класс вектора x 
        из множества X, состоящего из векторов длинны входного слоя.
        """
        results = []
        for x in X:
            y_predicted = np.argmax(self.feedforward(x))
            results.append(y_predicted)
        return np.array(results)

    def cost_derivative(self, output, y):
        """Возвращает вектор частный производных dC/da_output"""
        return output-y


def sigmoid(x):
    """Сигмоидная функция"""
    return 1./(1. + np.exp(-x))


def sigmoid_prime(x):
    """Производная сигмойдной функции"""
    sigm = sigmoid(x)
    return (1. - sigm)*sigm

import time
train, valid = load_train_data('train.csv', valid_size=.2, random_state=42)

netw = Network([784, 16, 16, 10], random_state=42)
start_time = time.time()
netw.SGD(train, 3, 20, 30, test_data=valid)
print("--- %s seconds ---" % (time.time() - start_time))