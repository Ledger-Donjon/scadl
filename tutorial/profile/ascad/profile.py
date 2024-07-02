import sys
import h5py
import numpy as np
from keras.models import Sequential, Model
from keras.layers import Conv1D, Dense, Flatten
from keras.layers import Dropout
from keras.optimizers import RMSprop
import keras
from keras.layers import Input, AveragePooling1D
from keras.layers import BatchNormalization, Dropout, GaussianNoise
from scadl.profile import Profile
from scadl.tools import sbox, normalization, remove_avg
from scadl.augmentation import Mixup, RandomCrop
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.optimizers import Adam
import tensorflow as tf


def leakage_model(data):
    """leakage model for sbox[2]"""
    return sbox[data["plaintext"][2] ^ data["key"][2]]


def aug_mixup(x, y):
    """Data augmenatation function based on mixup"""
    mix = Mixup()
    x, y = mix.generate(x_train=x, y_train=y, ratio=2, alpha=1)
    return x, y


def aug_crop(x, y):
    """Data augmenatation function based on RandomCrop"""
    mix = RandomCrop()
    x, y = mix.generate(x_train=x, y_train=y, ratio=1, window=5)
    return x, y


def mlp_short(len_samples):
    """It returns an MLP model"""
    model = Sequential()
    model.add(Dense(20, input_dim=len_samples, activation="relu"))
    # BatchNormalization()
    model.add(Dense(50, activation="relu"))
    model.add(Dense(256, activation="softmax"))
    optimizer = "adam"  # RMSprop(learning_rate=0.00001) #tf.keras.optimizers.Adam() # RMSprop(lr=0.00001)
    model.compile(
        loss="categorical_crossentropy",
        optimizer=optimizer,
        metrics=["accuracy"],
    )
    return model


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Need to specify the location of training data")
        exit()
    DIR = sys.argv[1]

    """loading traces and metadata for training"""
    file = h5py.File(f"{DIR}/ASCAD.h5", "r")
    leakages = file["Profiling_traces"]["traces"][:]
    metadata = file["Profiling_traces"]["metadata"][:]

    """Selecting poi where SNR gives the max value"""
    poi = leakages # np.concatenate((leakages[:, 515:520], leakages[:, 148:158]), axis=1)
    # poi = leakages - np.mean(leakages, axis=0)

    """Processing the traces"""
    x_train = normalization(remove_avg(poi), feature_range=(-1, 1))
    # x_train = handy_normalization(remove_avg(poi))
    GUESS_RANGE = 256

    """Loading the DL model mlp"""
    model_dl = mlp_short(x_train.shape[1])
    # model_dl = model_cnn(x_train.shape[1], GUESS_RANGE)

    """Profiling"""
    profile_engine = Profile(model_dl, leakage_model=leakage_model)
    profile_engine.data_augmentation(aug_mixup)
    profile_engine.train(
        x_train=x_train,
        metadata=metadata,
        epochs=50,
        batch_size=128,
        validation_split=0.1,
        data_augmentation=False,
    )
    profile_engine.save_model("model.keras")
