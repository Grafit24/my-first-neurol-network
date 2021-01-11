"""Модуль для загрузки и экспорта данных.

Загружает данные тренировачные и тестовые (.csv) в нужном для сети формате.
И экспортирует предсказания по шаблону data/submission_sample.csv.
Note. Название модуля и export_data немного вводит в диссонанс ,но я не предумал пока 
другого названия.
"""
from os import path

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

data_folder = path.join(path.dirname(__file__), 'data')

def load_train_data(file_name: str, valid_size=None, 
                    random_state=None, nrows=None):
    """Загружает данные для обучения.
    
    Parameters
    ----------
    file_name : str
        Название файла(не его путь!).

    valid_size : str
        Доля данных валидации от всех данных файла.

    random_state : int
        Число для установления начальных условий генератора случайных чисел.
        Если None ,то никак не влияет на генерацию.

    nrows : int
        Кол-во строк.
    
    Returns
    -------
    Tuple or List[ndarray]
        В случае ,если valid_size!=None. 
        Элементы которого train_data и valid_data.
        Или в другом случае список(train_data) элементы которого 
        это кортежы (x, y) ,где x и y вектора. При этом вектор y
        содержит нули и единицу на месте класса. 
    """
    df = pd.read_csv(path.join(data_folder, file_name), nrows=nrows)
    X = np.array(df.iloc[:, 1:])
    Y = np.array(df.iloc[:, 0])
    X = X/255

    if valid_size != None:
        X_train, X_valid, Y_train, Y_valid = train_test_split(X, Y, 
                                                              test_size=valid_size,
                                                              random_state=random_state
                                                              )
        data_train = [(x, y_to_vector(y)) for x, y in zip(X_train, Y_train)]
        data_valid = [(x, y_to_vector(y)) for x, y in zip(X_valid, Y_valid)]
        data = (data_train, data_valid)
    else:
        data = [(x, y_to_vector(y)) for x, y in zip(X, Y)]
    
    return data

def load_test_data(file_name: str, nrows=None):
    """Загружает данные для теста. 
    Параметры соответсвуют load_train_data.
    """
    df = pd.read_csv(path.join(data_folder, file_name), nrows=nrows)
    df = np.array(df)/255
    data = [x for x in df]

    return data


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