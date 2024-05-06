import tensorflow as tf
from keras import preprocessing
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Dense, Flatten



def model_mlp():
    model = Sequential()
    # model.add(tf.keras.layers.Flatten())
    model.add(Dense(500, activation=tf.nn.relu))
    model.add(Dense(500, activation=tf.nn.relu))
    model.add(Dense(500, activation=tf.nn.relu))
    model.add(Dense(500, activation=tf.nn.relu))
    model.add(Dense(256, activation=tf.nn.softmax))
    model.compile(optimizer='adam',
                  loss = 'sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model



def model_cnn(len_samples, guess_range):
    # img_input = Input(shape=(len_samples, 1))
    model = Sequential()
    model.add(Conv1D(filters=10, kernel_size=5, input_shape=(len_samples, 1)))
    # model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(784,1)))
    model.add(MaxPooling1D(pool_size=5))
    model.add(Flatten())
    model.add(Dense(1000, activation='relu'))
    model.add(Dense(guess_range, activation='softmax'))
    model.compile(optimizer='adam',
                  loss = 'sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model



