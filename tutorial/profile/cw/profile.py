import sys
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, Dense, Flatten
from scadl.profile import Profile
from scadl.tools import sbox, normalization
from scadl.augmentation import Mixup


def model_mlp(sample_len, range_outer_layer):
    """It returns an MLP model"""
    model = Sequential()
    model.add(Dense(500, input_dim=sample_len, activation=tf.nn.relu))
    model.add(Dense(500, activation=tf.nn.relu))
    model.add(Dense(500, activation=tf.nn.relu))
    model.add(Dense(500, activation=tf.nn.relu))
    model.add(Dense(range_outer_layer, activation=tf.nn.softmax))
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"],  # sparse_categorical_crossentropy
    )
    return model


def model_cnn(sample_len, range_outer_layer):
    """It takes sample_len and guess_range and passes a CNN model"""
    model = Sequential()
    model.add(Conv1D(filters=20, kernel_size=5, input_shape=(sample_len, 1), activation='tanh'))
    model.add(MaxPooling1D(pool_size=6))
    model.add(Flatten())
    model.add(Dense(200, activation="relu"))
    model.add(Dense(range_outer_layer, activation="softmax"))
    model.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
    )
    return model


def leakage_model(data):
    """leakage model for sbox[0]"""
    return sbox[data["plaintext"][0] ^ data["key"][0]]


def data_aug(x_training, y_training):
    """It's used for data augmentation and it takes x, y as leakages and labels"""
    mix = Mixup()
    x, y = mix.generate(x_train=x_training, y_train=y_training, ratio=0.6)
    return x, y


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Need to specify the location of training data and model")
        exit()
    DIR = sys.argv[1]
    leakages = np.load(DIR + "/train/traces.npy")
    metadata = np.load(DIR + "/train/combined_train.npy")
    """Selecting poi where SNR gives the max value"""
    x_train = normalization(
        leakages[:, 1315:1325], feature_range=(0, 1)
    )  # Normalization is used for improving the learning

    len_samples = x_train.shape[1]
    """Loading the DL model mlp"""
    len_samples = x_train.shape[1]
    GUESS_RANGE = 256
    if sys.argv[2] == "mlp":
        model_dl = model_mlp(sample_len=len_samples, range_outer_layer=GUESS_RANGE)
    else:
        model_dl = model_cnn(len_samples, GUESS_RANGE)    
    """Profiling"""
    profile_engine = Profile(model_dl, leakage_model=leakage_model)
    profile_engine.data_augmentation(data_aug)
    profile_engine.train(
        x_train=x_train,
        metadata=metadata,
        epochs=50,
        batch_size=100,
        data_augmentation=False,
    )
    profile_engine.save_model("model.keras")
