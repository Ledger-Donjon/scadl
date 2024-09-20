import sys

import keras
import numpy as np
from keras.layers import Conv1D, Dense, Flatten, MaxPooling1D
from keras.models import Sequential
from sklearn.preprocessing import MultiLabelBinarizer

from scadl.multi_label_profile import MultiLabelProfile
from scadl.tools import gen_labels, sbox


def mlp_multi_label(nb_neurons: int = 50, nb_layers: int = 4) -> keras.Model:
    """It takes nb_neurons as the number of neurons per layer and number of layres.
    It returns an MLP model"""
    model = Sequential()
    model.add(Dense(nb_neurons, activation="relu"))
    for _ in range(nb_layers - 2):
        model.add(Dense(nb_neurons, activation="relu"))
        # Dropout(0.1)
        # BatchNormalization()
    model.add(Dense(512, activation="sigmoid"))
    optimizer = "adam"
    model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=["accuracy"])
    return model


def cnn_multi_label(len_samples: int, guess_range: int) -> keras.Model:
    """It takes len of samples and guess range.
    It returns a CNN model"""
    model = Sequential()
    model.add(Conv1D(filters=20, kernel_size=5, input_shape=(len_samples, 1)))
    model.add(MaxPooling1D(pool_size=5))
    model.add(Flatten())
    model.add(Dense(200, activation="relu"))
    model.add(Dense(guess_range, activation="sigmoid"))
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


def leakage_model(data: np.ndarray, key_byte: np.ndarray) -> int:
    """It takes data and key_byte and returns a leakage function"""
    return sbox[data["plaintext"][key_byte] ^ data["key"][key_byte]]


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Need to specify the location of training data")
        exit()

    DIR = sys.argv[1]
    leakages = np.load(DIR + "/train/traces.npy")
    metadata = np.load(DIR + "/train/combined_train.npy")
    size_profiling = len(metadata)
    """poi for sbox[p0^k0] and sbox[p1^k1]"""
    poi = np.concatenate((leakages[:, 1315:1325], leakages[:, 1490:1505]), axis=1)
    """"generate labels"""
    y_0 = gen_labels(
        leakage_model=leakage_model, metadata=metadata, key_byte=0
    ).reshape((size_profiling, 1))
    y_1 = gen_labels(
        leakage_model=leakage_model, metadata=metadata, key_byte=1
    ).reshape((size_profiling, 1))
    """shifting second label by 256"""
    combined_labels = np.concatenate(
        (y_0, y_1 + 256), axis=1
    )  # second labels are shifted by 256
    label = MultiLabelBinarizer()
    labels_fit = label.fit_transform(combined_labels)
    """load model"""
    GUESS_RANGE = 512
    if sys.argv[2] == "mlp":
        model_dl = mlp_multi_label()
    else:
        model_dl = cnn_multi_label(poi.shape[1], GUESS_RANGE)

    """call multi-label profiling engine"""
    profile = MultiLabelProfile(model_dl)
    profile.train(x_train=poi, y_train=labels_fit, epochs=100)
    profile.save_model("model.keras")
