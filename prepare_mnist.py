"""Модуль для удаления дубликатов.

P.S. Я скачал полный датасет мниста для валидации и теста 
,ведь на кэгле есть ограничения на кол-во попыток ,а резать 
обучающии данные не очень хочется.
"""
import os
import gzip
import pickle
import pandas as pd
import numpy as np

print("Loading data...")

data_folder = os.path.join(os.path.dirname(__file__), 'data')

# Загружаем данные
with gzip.open(os.path.join(data_folder, "mnist.pkl.gz"), "rb") as f:
        train, valid, test = pickle.load(f, encoding="latin-1")
        train = [pd.DataFrame(d) for d in train[::-1]]
        valid = [pd.DataFrame(d) for d in valid[::-1]]
        test = [pd.DataFrame(d) for d in test[::-1]]

orig_train = pd.read_csv(os.path.join(data_folder, "train.csv"))
orig_train.iloc[:, 1:] /= 256
print("Loading data complete")

# Переводим данные в df
train = pd.concat(train, axis=1)
valid = pd.concat(valid, axis=1)
test = pd.concat(test, axis=1)

mnist = pd.concat([train, valid, test])
orig_train.columns = mnist.columns
print("Size of maintain dataset:", mnist.shape[0])

full = pd.concat([mnist, orig_train])
print("Number of duplicates:", full.duplicated().sum())

print("Deleting duplicates...")
new_full = full.drop_duplicates(keep=False)
new_full = new_full.reset_index(drop=True)

indexer = np.array(new_full.index)
np.random.shuffle(indexer)

# Разделяем датасет попалам
size = new_full.shape[0]
valid_index = indexer[:int(size/2)]
test_index = indexer[int(size/2):]

valid = new_full.iloc[valid_index, :]
test = new_full.iloc[test_index, :]
print("Deleting done!")

# Экспортируем
print("Exporting new valid and test data...")
valid.reset_index(drop=True).to_csv("data/mnist_valid.csv")
test.reset_index(drop=True).to_csv("data/mnist_test.csv")
print("Export done!")

