import sys

import h5py
import keras
import matplotlib.pyplot as plt
import numpy as np
from keras.layers import Dense
from keras.models import Sequential
from tqdm import tqdm

from scadl.non_profile import NonProfile
from scadl.tools import normalization, remove_avg, sbox

TARGET_BYTE = 2


def leakage_model(data: np.ndarray, guess: int) -> int:
    """It returns the leakage function"""
    # return 1 & ((sbox[data["plaintext"][TARGET_BYTE] ^ guess]) >> 7) #msb
    return 1 & ((sbox[data["plaintext"][TARGET_BYTE] ^ guess]))  # lsb
    # return hw(sbox[data['plaintext'][TARGET_BYTE] ^ guess]) #hw


def mlp_short(len_samples: int) -> keras.Model:
    """It returns an MLP model"""
    model = Sequential()
    model.add(Dense(20, input_dim=len_samples, activation="relu"))
    model.add(Dense(10, activation="relu"))
    model.add(Dense(2, activation="softmax"))
    optimizer = "adam"
    model.compile(
        loss="mean_squared_error",
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
    SIZE_TEST = 20000
    file = h5py.File(f"{DIR}/ASCAD.h5", "r")
    leakages = np.array(file["Profiling_traces"]["traces"][:], dtype=np.int8)[
        0:SIZE_TEST
    ]
    metadata = file["Profiling_traces"]["metadata"][:][0:SIZE_TEST]
    correct_key = metadata["key"][0][TARGET_BYTE]

    """Subtracting average from traces + normalization"""
    x_train = normalization(remove_avg(leakages), feature_range=(-1, 1))
    """Non-profiling DL"""
    EPOCHS = 15
    guess_range = range(0, 256)
    acc = np.zeros((len(guess_range), EPOCHS))
    profile_engine = NonProfile(leakage_model=leakage_model)
    for index, guess in enumerate(tqdm(guess_range)):
        acc[index] = profile_engine.train(
            model=mlp_short(x_train.shape[1]),
            x_train=x_train,
            metadata=metadata,
            hist_acc="accuracy",
            guess=guess,
            num_classes=2,
            epochs=EPOCHS,
            batch_size=1000,
        )
    guessed_key = np.argmax(np.max(acc, axis=1))
    print(f"guessed key = {guessed_key}")
    plt.plot(acc.T, "grey")
    plt.plot(acc[correct_key], "black")
    plt.xlabel("Number of epochs")
    plt.ylabel("Accuracy ")
    plt.show()
