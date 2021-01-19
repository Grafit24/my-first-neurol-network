"""Многослойный перцептрон

обучающющийся через стохастический крадиентный спуск. Для вычисления 
градиента используются алгоритм обратного распространения ошибки.

Note. Весь код ниже написан с целью понять ,что творится под "капотом" 
нейросети. Писался код с опорой на книгу "Neural Networks and Deep Learning" 
Michael Nielsen'а.

Список улучшений которые я довесил в сравнение с v1:
- новая функция потерь cross-entropy
- L2 регулиризация
- изменил инизилизацию весов
- no-improvement-in-n-epochs
- learning shedule(динамическое изменение learning rate'а)
"""
from typing import List, Tuple, Dict

import numpy as np
import numpy.random as rand


class Network(object):
    """Многослойный перцептрон с сигмоидной функцией активации.
    Функция потерь - cross entropy. С использованием L2 регулиризации.
    
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
    set_monitoring(evaluation_accuracy=False, evaluation_cost=False, 
                   training_accuracy=False, training_cost=False, 
                   learning_rate=False, off=False)
        Настраивает отображение данных при обучении сети.

    SGD(self, training_data, eta, epochs, mini_batch_size, 
        evaluation_data=None, lmbda=0.0, n_epoch=None, factor=None)
        Обучает нейросеть по алгоритму stochastic gradient descent.

    update_mb(mini_batch, eta)
        Обновляет веса и смещения по примерам из mini_batch.

    backprop(x, y)
        Алгоритм обратного распространения ошибки.

    feedforward(a)
        Вычиляет вектор-результат сети для вектора(единичного примера).

    matrixbase_feedforward(a)
        Вычиляется вектор-результат сети для матрицы(нескольких примеров).

    evaluate(evaluation_data)
        Считает точность сети на тестовых данных(evaluation_data).

    predict(X)
        Определяет наиболее вероятный класс вектора x из множества X.
    
    сost_deveriate(output, y)
        Возвращает вектор частный производных dC/da_output 
        для функции потерь - cross entropy.
        
    cost_function(x)
        Считает функцию потерь - cross entropy.
    
    no_improvement_in_n(current_accuracy, n=1)
        Если сеть не улучшает результаты accuracy 
        в течение n epoch возвращает True.
    """
    def __init__(self, sizes: List[int], random_state=None):
        self.nlayers = len(sizes) 
        self.sizes = sizes
        self.random_state = random_state

        self.set_monitoring(evaluation_accuracy=True)

        rand.seed(random_state)
        self.weights = [rand.randn(sizes[i+1], sizes[i])/np.sqrt(sizes[i])
                        for i in range(self.nlayers-1)]
        rand.seed(random_state)
        self.biases = [rand.randn(sizes[i+1]) 
                       for i in range(self.nlayers-1)]
    
    def set_monitoring(self, evaluation_accuracy=False, evaluation_cost=False, 
            training_accuracy=False, training_cost=False, 
            learning_rate=False, off=False):
        """Настраивает отображение данных при обучение сети.
        
        Parameters
        ----------
        evaluation_accuracy=False : bool
        evaluation_cost=False : bool
            пишет в консоль точность для evaluation_data.

        training_accuracy=False : bool
        training_cost=False : bool
            пишет в консоль значение потерь для обучающих данных.
        
        learning_rate=False : bool
            пишет в консоль изменения learning rate'а.
            Если factor и n_epochs не None!

        off=False : bool
            если True перестаёт писать ,что либо в консоль.
        """
        self.monitor = {"evaluation_accuracy":evaluation_accuracy,
                        "evaluation_cost": evaluation_cost,
                        "training_accuracy":training_accuracy,
                        "training_cost":training_cost,
                        "learning_rate":learning_rate,
                        "off":off,
                        }

    
    def SGD(self, training_data, eta, epochs, mini_batch_size, 
            evaluation_data=None, lmbda=0.0, n_epoch=None, factor=None,
            ):
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

        eta : float
            learning rate

        epochs : int

        mini_batch_size : int

        evaluation_data=None : List[Tuple[(X, y)]]
            Если None ,тогда не пишет точность сети 
            после каждой эпохи. В обратном случае 
            соответсвенно пишет. Должен быть тот же вид,
            что и у training_data.

        lmbda=0.0 : float
            константа регулиризатора.
        
        n_epoch=None : int
            реализуют стратегию no-improvement-in-n epochs. 
            Если нет улучшений в течение n эпох ,то:
            - Если factor = None ,то заканчивает обучение сети.
            - Если factor != None (смотреть factor)
        
        factor=None : Tuple[int]
            Не действует без n_epochs! 
            (factor, factor_stop) Если нет улучщений learning_rate/factor,
            пока eta не станет равно eta/(factor^factor_stop).
        """
        self.lmbda = lmbda
        self.n_samples = len(training_data)

        self.best_accuracy = 0
        self.epoch_ago = 0
        
        if factor is not None: 
            factor, factor_stop = factor
        else:
            factor_stop = None
        factor_rate = 0

        results = []
        accuracy_test, cost_test = [], []
        accuracy_train, cost_train = [], []
        for epoch in range(epochs):
            np.random.seed(self.random_state)
            rand.shuffle(training_data)
            
            n_train_d = len(training_data)
            mini_batches = [training_data[k:(mini_batch_size+k)] 
                            for k in range(0, n_train_d, mini_batch_size)]

            # Обновляем веса по mini-bunch
            for mini_batch in mini_batches:
                data = mini_batch
                x, y = [], []
                for i in data:
                    x.append(i[0])
                    y.append(i[1])

                x = np.array(x).transpose()
                y = np.array(y).transpose()
                self.update_mb(x, y, eta)
            
            # Данные по точности и cost сети,
            # а также вывод в консоль.
            text = "Epoch {0} training complete\n".format(epoch+1)
            if evaluation_data is not None:
                n_test_data = len(evaluation_data)
                acc, cost = self.evaluate(evaluation_data)
                evaluate_result = np.sum(acc)
                accuracy_test.append(evaluate_result/n_test_data)
                cost_test.append(cost)
                if self.monitor["evaluation_accuracy"] or self.monitor["evaluation_cost"]:
                    text += " evaluation data\n"
                if self.monitor["evaluation_accuracy"]:
                    text += 4*" " + "| Accuracy: %f \n" % \
                        (accuracy_test[-1]*100,)
                if self.monitor["evaluation_cost"]:
                    text += 4*" " + "| Cost:     %f\n" % \
                        cost_test[-1]
            if self.monitor["training_accuracy"] or self.monitor["training_cost"]:
                acc_t, cost_t = self.evaluate(training_data)
                accuracy_train.append(np.sum(acc_t)/self.n_samples)
                cost_train.append(cost_t)

                text += " training data\n"
                if self.monitor["training_accuracy"]:
                    text += 4*" " + "| Accuracy: %f\n" % \
                        (accuracy_train[-1]*100,)
                if self.monitor["training_cost"]:
                    text += 4*" " + "| Cost:     %f\n" % cost_train[-1] 

            # Остановка обучения досрочно.
            if (n_epoch is not None) and (evaluation_data is not None):
                no_improv = self.no_improvement_in_n(accuracy_test[-1], n=n_epoch)
                if no_improv:
                    if factor_stop is not None:
                        # Learning shedule by factor
                        eta /= factor
                        factor_rate += 1
                        self.epoch_ago = 0

                        if self.monitor["learning_rate"]:
                            text += " learning rate: %f \n" % eta

                        if factor_rate >= factor_stop: break
                    else:
                        break
                
            if not self.monitor["off"]:
                print(text)
                
        self.best_accuracy = max(accuracy_test)

        if evaluation_data is not None:
            results.append(accuracy_test)
            results.append(cost_test)
        if self.monitor["training_cost"] or self.monitor["training_accuracy"]:
            results.append(accuracy_train)
            results.append(cost_train)

        return tuple(results)

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
        self.weights = [(1-eta*(self.lmbda/self.n_samples))*w-(eta/mb_size)*nw 
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
        delta = self.cost_derivative(activations[-1], y)
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
        
    def evaluate(self, data):
        """Считает точность и функцию стоимости
         сети на тестовых данных(data).
         """
        x = [xi for xi, _ in data]
        x = np.array(x).transpose()
        y = [yi for _, yi in data]
        y_matrix = np.array(y).transpose()
        y_vector = y_matrix.argmax(axis=0)

        # Выход сети.
        network_results_matrix = self.matrixbase_feedforward(x)
        # Наиболее вероятный результат выхода сети.
        network_results_vector = np.argmax(network_results_matrix, axis=0)

        results = network_results_vector==y_vector
        cf_results = self.cost_function(network_results_matrix, y_matrix)

        return results, cf_results
    
    def predict(self, X):
        """Определяет наиболее вероятный класс вектора x 
        из множества X, состоящего из векторов длинны входного слоя.
        """
        results = []
        for x in X:
            y_predicted = np.argmax(self.feedforward(x))
            results.append(y_predicted)
        return np.array(results)
    
    def cost_function(self, output, y):
        """Возвращает скаляр функции потерь - cross entropy."""
        fn = np.sum(np.nan_to_num(-y*np.log(output)-(1-y)*np.log(1-output)), axis=0)
        fn_mean = np.mean(fn)
        l2 = 0.5*(self.lmbda/output.shape[1])*sum(
            np.linalg.norm(w)**2 for w in self.weights)
        return fn_mean+l2

    def cost_derivative(self, output, y):
        """Возвращает вектор частный производных dC/da_output 
        для функции потерь cross entropy.
        """
        return output-y
    
    def no_improvement_in_n(self, current_accuracy, n=1):
        """Если сеть не улучшает результаты accuracy 
        в течение n epoch возвращает True.
        """
        if current_accuracy > self.best_accuracy:
            self.best_accuracy = current_accuracy
            self.epoch_ago = 0
        else:
            self.epoch_ago += 1

        return self.epoch_ago >= n

def sigmoid(x):
    """Сигмоидная функция"""
    return 1./(1. + np.exp(-x))


def sigmoid_prime(x):
    """Производная сигмойдной функции"""
    sigm = sigmoid(x)
    return (1. - sigm)*sigm