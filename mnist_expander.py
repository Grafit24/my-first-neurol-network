"""Модуль для расширения dataset'а за счёт поворота изображения."""
import os 
import numpy as np
import pandas as pd
from numpy.lib.function_base import angle
from scipy.ndimage.interpolation import rotate

def mnist_expand(dataset, digreas):
    if type(dataset) is pd.DataFrame:
        dataset_np = dataset.to_numpy()
    rotated_dataset = np.zeros_like(dataset_np)
    for i in range(dataset_np.shape[0]):
        vector = dataset_np[i]
        n_pixels = int(np.sqrt(vector.shape[0]))
        image = vector.reshape(n_pixels, -1)
        rot_image = rotate(image, digreas, reshape=False)
        vector_of_rot_image = rot_image.reshape(n_pixels**2,)
        rotated_dataset[i, :] = vector_of_rot_image

    return rotated_dataset

if __name__ == "__main__":
    ANGLE = 15
    dirname = os.path.dirname(__file__)
    data_folder = os.path.join(dirname, "data")

    dataset = pd.read_csv(os.path.join(data_folder, "train.csv"))
    print("Start size of dataset:", dataset.shape[0])
    if "train_expanded.csv" not in os.listdir(data_folder):
        data = dataset.iloc[:, 1:]
        labels = dataset["label"].copy()
        new_data = mnist_expand(data, ANGLE)
        new_dataset = pd.DataFrame(
            np.c_[labels.to_numpy(), new_data],
            columns=dataset.columns
            )
        dataset_expanded = pd.concat([dataset, new_dataset], ignore_index=True)
        
        print("Expanded dataset size:", dataset_expanded.shape[0])
        dataset_expanded.to_csv("data/train_expanded.csv", index=False)
    else:
        print("Expanded dataset already exisist.")
