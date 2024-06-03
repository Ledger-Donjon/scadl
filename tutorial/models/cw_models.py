import tensorflow as tf
from keras import preprocessing
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Dense, Flatten
from tensorflow.keras.layers import Dropout
from keras.optimizers import RMSprop
import keras
from tensorflow.keras.losses import CategoricalCrossentropy
from keras.layers import Flatten, Dense, Input, Conv1D, AveragePooling1D
from tensorflow.keras.layers import BatchNormalization

"""No protected CW AES"""

def model_mlp():
    model = Sequential()
    # model.add(tf.keras.layers.Flatten())
    model.add(Dense(500, activation=tf.nn.relu))
    model.add(Dense(500, activation=tf.nn.relu))
    model.add(Dense(500, activation=tf.nn.relu))
    model.add(Dense(500, activation=tf.nn.relu))
    model.add(Dense(256, activation=tf.nn.softmax))
    model.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]     # sparse_categorical_crossentropy
    )
    return model

def model_cnn(len_samples, guess_range):
    model = Sequential()
    model.add(Conv1D(filters=20, kernel_size=5, input_shape=(len_samples, 1)))
    model.add(MaxPooling1D(pool_size=5))
    model.add(Flatten())
    model.add(Dense(200, activation="relu"))
    model.add(Dense(guess_range, activation="softmax"))
    model.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
    )
    return model


"""Multi-label architectures"""

def mlp_multi_label(node=50, layer_nb=4):
    model = Sequential()
    model.add(Dense(node, activation="relu"))
    for i in range(layer_nb - 2):
        model.add(Dense(node, activation="relu"))
        # Dropout(0.1)
        # BatchNormalization()
    model.add(Dense(512, activation="sigmoid"))
    optimizer = "adam"
    model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=["accuracy"])
    return model


def cnn_multi_label(len_samples, guess_range):
    model = Sequential()
    model.add(Conv1D(filters=20, kernel_size=5, input_shape=(len_samples, 1)))
    model.add(MaxPooling1D(pool_size=5))
    model.add(Flatten())
    model.add(Dense(200, activation="relu"))
    model.add(Dense(guess_range, activation="sigmoid"))
    model.compile(
        optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"]
    )
    return model



# def model_cnn(len_samples, guess_range):
#     model = Sequential()
#     model.add(Conv1D(filters=10, kernel_size=5, input_shape=(len_samples, 1)))
#     model.add(MaxPooling1D(pool_size=5))
#     model.add(Flatten())
#     model.add(Dense(200, activation="relu"))
#     model.add(Dense(guess_range, activation="softmax"))
#     model.compile(
#         optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"]
#     )
#     return model



"""Non-profiling attacks"""

def mlp_non_profiling():
    model = Sequential()
    # model.add(tf.keras.layers.Flatten())
    model.add(Dense(20, activation=tf.nn.relu))
    model.add(Dense(10, activation=tf.nn.relu))
    # model.add(Dense(500, activation=tf.nn.relu))
    # model.add(Dense(500, activation=tf.nn.relu))
    # model.add(Dense(500, activation=tf.nn.relu))
    # model.add(Dense(9, activation=tf.nn.softmax))
    model.add(Dense(2, activation=tf.nn.softmax))
    model.compile(optimizer="adam", loss="mean_squared_error", metrics=["accuracy"])
    return model



def mlp_best(node=200, inner_layers=4):  # node=500
    model = Sequential()
    model.add(Dense(node, activation="relu"))  # 28   #node
    for i in range(inner_layers):
        model.add(Dense(node, activation="relu"))
        # Dropout(0.1)  # Dropout(0.01)
        # BatchNormalization()
    model.add(Dense(2, activation="softmax"))
    optimizer = "adam"  # keras.optimizers.Adam(learning_rate=0.01) #RMSprop(lr=0.00001)# 'adam'#RMSprop(lr=0.00001)
    model.compile(
        loss="mean_squared_error", optimizer=optimizer, metrics=["accuracy"]
    )  # categorical_crossentropy  mean_squared_error
    return model


def cnn_best(len_samples, guess_range):
    model = Sequential()
    model.add(Conv1D(filters=20, kernel_size=5, input_shape=(len_samples, 1)))
    model.add(MaxPooling1D(pool_size=5))
    model.add(Flatten())
    model.add(Dense(200, activation="relu"))
    model.add(Dense(guess_range, activation="softmax"))
    model.compile(
        optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"]
    )
    return model


