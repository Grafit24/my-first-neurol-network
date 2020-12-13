from os import path
import pickle

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

data_folder = path.join(path.dirname(__file__), 'data')

def load_data(file_name, valid=False, nrows=None):
    df = pd.read_csv(path.join(data_folder, file_name), nrows=nrows)
    X = np.array(df.iloc[:, 1:])
    Y = np.array(df.iloc[:, 0])

    if valid:
        X_train, X_valid, Y_train, Y_valid = train_test_split(X, Y, test_size=.2)
        data_train = [(x, y) for x, y in zip(X_train, Y_train)]
        data_valid = [(x, y) for x, y in zip(X_valid, Y_valid)]
        data = (data_train, data_valid)
    else:
        data = [(x, y) for x, y in zip(X, Y)]
    return data


with open('test_data.pickle', 'wb') as f:
    pickle.dump(load_data('train.csv', nrows=1), f, protocol=2)
