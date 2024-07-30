import sys
import h5py
import numpy as np
from keras.models import Sequential
from keras.layers import Conv1D, Dense, Flatten
from keras.layers import Dropout
from keras.optimizers import RMSprop, Adam
import keras
from keras.layers import MaxPooling1D
from scadl.profile import Profile
from scadl.tools import sbox, normalization, remove_avg
from scadl.augmentation import Mixup, RandomCrop
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf


def leakage_model(data):
    """leakage model for sbox[2]"""
    return sbox[data["plaintext"][2] ^ data["key"][2]]


def aug_mixup(x, y):
    """Data augmenatation function based on mixup"""
    mix = Mixup()
    x, y = mix.generate(x_train=x, y_train=y, ratio=1, alpha=1)
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


def model_cnn(sample_len, range_outer_layer):
    """It takes sample_len and guess_range and passes a CNN model"""
    model = Sequential()
    model.add(
        Conv1D(
            filters=8,
            kernel_size=32,
            padding="same",
            input_shape=(sample_len, 1),
            activation="relu",
        )
    )
    model.add(MaxPooling1D(pool_size=3))
    model.add(
        Conv1D(
            filters=8,
            kernel_size=16,
            padding="same",
            input_shape=(sample_len, 1),
            activation="tanh",
        )
    )
    model.add(MaxPooling1D(pool_size=3))
    model.add(
        Conv1D(
            filters=8,
            kernel_size=8,
            padding="same",
            input_shape=(sample_len, 1),
            activation="tanh",
        )
    )
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    Dropout(0.1)
    model.add(Dense(50, activation="relu"))
    model.add(Dense(range_outer_layer, activation="softmax"))
    model.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
    )
    return model


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Need to specify the location of training data and model")
        exit()
    DIR = sys.argv[1]

    """loading traces and metadata for training"""
    file = h5py.File(f"{DIR}/ASCAD.h5", "r")
    leakages = file["Profiling_traces"]["traces"][:]
    metadata = file["Profiling_traces"]["metadata"][:]

    """Selecting poi where SNR gives the max value"""
    poi = (
        leakages  # np.concatenate((leakages[:, 515:520], leakages[:, 148:158]), axis=1)
    )
    """Processing the traces"""
    x_train = normalization(remove_avg(poi), feature_range=(-1, 1))
    GUESS_RANGE = 256
    """Loading the DL model mlp"""
    if sys.argv[2] == "mlp":
        model_dl = mlp_short(x_train.shape[1])
    else:
        model_dl = model_cnn(x_train.shape[1], GUESS_RANGE)
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
