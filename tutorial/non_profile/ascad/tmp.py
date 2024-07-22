import sys
import h5py
import numpy as np
import matplotlib.pyplot as plt
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
from scadl.non_profile import NonProfile


TARGET_BYTE = 2


def leakage_model(data, guess):
    """It returns the leakage function"""
    # return 1 & ((sbox[data["plaintext"][TARGET_BYTE] ^ guess]) >> 7) #msb
    return 1 & ((sbox[data["plaintext"][TARGET_BYTE] ^ guess]))  # lsb
    # return hw(sbox[data['plaintext'][TARGET_BYTE] ^ guess]) #hw


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
    """Selecting the model"""
    model_dl = mlp_short(x_train.shape[1])

    """Non-profiling DL"""
    range_key = range(0, 256)
    EPOCHS = 20
    acc = np.zeros((len(range_key), EPOCHS))
    for index, guess in enumerate(range_key):
        profile_engine = NonProfile(leakage_model=leakage_model)
        model_dl = mlp_short(x_train.shape[1])
        acc[index] = profile_engine.train(
            model=model_dl,
            x_train=x_train,
            metadata=metadata,
            hist_acc="accuracy",
            key_range=range(guess, guess + 1),
            num_classes=2,
            epochs=EPOCHS,
            batch_size=1000,
        )

    """Selecting the key with the highest accuracy key"""
    guessed_key = np.argmax(np.max(acc, axis=1))
    print(f"guessed key = {guessed_key}")
    plt.plot(acc.T, "grey", linewidth=2)
    plt.plot(acc[correct_key], "black", linewidth=2)
    plt.xlabel("Number of epochs", fontsize=40)
    plt.ylabel("Accuracy ", fontsize=40)
    plt.xticks(fontsize=25)
    plt.yticks(fontsize=25)
    plt.show()
