"""Модуль для загрузки и экспорта данных.

Загружает данные тренировачные и тестовые (.csv) в нужном для сети формате.
И экспортирует предсказания по шаблону data/submission_sample.csv.
Note. Название модуля и export_data немного вводит в диссонанс ,но я не предумал пока 
другого названия.
"""
import gzip
import pickle
from os import path

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

data_folder = path.join(path.dirname(__file__), 'data')

def load_train_data(file_name: str, nrows=None):
    """Загружает данные для обучения.
    
    Parameters
    ----------
    file_name : str
        Название файла(не его путь!).

    nrows : int
        Кол-во строк.
    
    Returns
    -------
    Tuple or List[ndarray]
        Возвращает список(train_data) элементы которого 
        это кортежы (x, y) ,где x и y вектора. При этом вектор y
        содержит нули и единицу на месте класса. 
    """
    df = pd.read_csv(path.join(data_folder, file_name), nrows=nrows)
    X = np.array(df.iloc[:, 1:])
    Y = np.array(df.iloc[:, 0])
    X = X/255

    data = [(x, y_to_vector(y)) for x, y in zip(X, Y)]
    
    return data


def load_test_data(file_name: str, nrows=None):
    """Загружает тестовые данные с kaggle."""
    df = pd.read_csv(path.join(data_folder, file_name), nrows=nrows)
    df = np.array(df)/255
    data = [x for x in df]

    return data


def load_evaluation_data():
    """"Загружает valid и test даынные из файла mnist.pkl.gz"""
    with gzip.open(path.join(data_folder, "mnist.pkl.gz"), "rb") as f:
        _, valid, test = pickle.load(f, encoding="latin-1"
        )
    valid = [(x, y_to_vector(y)) for x, y in zip(*valid)]
    test = [(x, y_to_vector(y)) for x, y in zip(*test)]

    return valid, test


def load_mnist_data():
    """Загружает данные: train(из train.csv), valid, test."""
    train = load_train_data("train.csv")
    valid, test = load_evaluation_data()
    return train, valid, test

def y_to_vector(y):
    """Заменяет скаляр вектором в котором 
    на месте индекса=скаляра стоит 1 остальные 
    элементы равны 0.

    Пример: y_vec[y]=1 -> y_vec=[0, ..., 0, 1, 0, ..., 0]
    """
    y_vector = np.zeros(10)
    y_vector[y] = 1
    return y_vector


def export_data(file_name, data):
    """Преобразует ndarray в csv соответсвующий 
    шаблону data/sample_submission.csv.
    Note. Шаблон взят с kaggle в соревнование 
    по распознованию рукописных цифр.
    
    Parameters
    ----------
    file_name : str
        Название файла в который экспортируются данные.

    data : Any(ndarray, df)
        Матрица (n, 2) ,где два столбца индекс примера 
        и предсказанная цифра для примера
    """
    data = pd.DataFrame(data)
    data.set_index(pd.Series(data.index, name='ImageId')+1, inplace=True)
    data.columns = ['Label']
    data['Label'] = data['Label'].astype(int)
    data.to_csv(file_name)