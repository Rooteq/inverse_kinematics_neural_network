import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import keras
import os.path

def train_dataset():
    data = pd.read_csv("dataset.csv")

    print(data)

    data_size = len(data)

    model = keras.models.Sequential([
        keras.layers.Dense(units=100, activation='relu'),
        keras.layers.Dense(units=2)
    ])

    loss = keras.losses.mean_squared_error

train_dataset()
