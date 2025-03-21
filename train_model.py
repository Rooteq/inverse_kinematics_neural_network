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
    input_position = []
    input_joint = []
    output_joint = []
    for i in range(data_size):
        input_position.append( tuple( [float(i) for i in data['input position'][i][1:-1].split(',')] ) )
        input_joint.append( tuple( [float(i) for i in data['input joint'][i][1:-1].split(',')] ) )
        output_joint.append( tuple( [float(i) for i in data['output joint'][i][1:-1].split(',')] ) )

    x_train = []
    y_train = []
    for i in range(data_size):
        x_sample = list(input_position[i])
        x_sample.extend(input_joint[i])
        x_train.append(x_sample)
        y_sample = list(output_joint[i])
        y_train.append(y_sample)

    x_train = np.array(x_train)
    y_train = np.array(y_train)

    model = keras.models.Sequential([
        keras.layers.Dense(units=100, input_shape=(4,), activation='relu'),
        keras.layers.Dense(units=100, activation='relu'),
        keras.layers.Dense(units=100, activation='relu'),
        keras.layers.Dense(units=2)
    ])

    loss = keras.losses.mean_squared_error
    optimizer = keras.optimizers.Adam(learning_rate=1e-2)

    early_stop = keras.callbacks.EarlyStopping(monitor='loss', patience=100, restore_best_weights=True)

    model.compile(optimizer=optimizer, loss=loss, metrics=[keras.metrics.RootMeanSquaredError()])
    model.fit(tf.expand_dims(x_train, -1), y_train, epochs=200, verbose=1, callbacks=[early_stop])

    model.predict( np.array([[-0.32,  0.74, -3.13,  0.86]]) )
    if os.path.isfile('arm_model.h5') is False:
        model.save('arm_model.h5')

train_dataset()