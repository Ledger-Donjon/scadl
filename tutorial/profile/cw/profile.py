import sys
from pathlib import Path

import keras
import numpy as np
from keras.layers import Conv1D, Dense, Flatten, Input, MaxPooling1D
from keras.models import Sequential

from scadl.augmentation import Mixup
from scadl.profile import Profile
from scadl.tools import normalization, sbox


def model_mlp(sample_len: int, range_outer_layer: int) -> keras.Model:
    model = Sequential()
    model.add(Input(shape=(sample_len,)))
    model.add(Dense(500, activation="relu"))
    model.add(Dense(500, activation="relu"))
    model.add(Dense(500, activation="relu"))
    model.add(Dense(500, activation="relu"))
    model.add(Dense(range_outer_layer, activation="softmax"))
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def model_cnn(sample_len: int, range_outer_layer: int) -> keras.Model:
    model = Sequential()
    model.add(Input(shape=(sample_len, 1)))
    model.add(Conv1D(filters=20, kernel_size=5, activation="tanh"))
    model.add(MaxPooling1D(pool_size=6))
    model.add(Flatten())
    model.add(Dense(200, activation="relu"))
    model.add(Dense(range_outer_layer, activation="softmax"))
    model.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
    )
    return model


def leakage_model(data: np.ndarray) -> int:
    """leakage model for sbox[0]"""
    return sbox[data["plaintext"][0] ^ data["key"][0]]


def data_aug(
    x_training: np.ndarray, y_training: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """It's used for data augmentation and it takes x, y as leakages and labels"""
    mix = Mixup()
    x, y = mix.generate(x_train=x_training, y_train=y_training, ratio=0.6)
    return x, y


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Need to specify the location of training data and model")
        exit()

    dataset_dir = Path(sys.argv[1])
    leakages = np.load(dataset_dir / "train/traces.npy")
    metadata = np.load(dataset_dir / "train/combined_train.npy")

    # Select POIs where SNR gives the max value
    # Normalization improves the learning
    x_train = normalization(leakages[:, 1315:1325], feature_range=(0, 1))

    len_samples = x_train.shape[1]

    # Build the model
    len_samples = x_train.shape[1]
    GUESS_RANGE = 256
    if sys.argv[2] == "mlp":
        model = model_mlp(sample_len=len_samples, range_outer_layer=GUESS_RANGE)
    elif sys.argv[2] == "cnn":
        model = model_cnn(len_samples, GUESS_RANGE)
    else:
        print("Invalid model type")
        exit()

    model.summary()

    # Train the model
    profile_engine = Profile(model, leakage_model=leakage_model)
    profile_engine.data_augmentation(data_aug)
    profile_engine.train(
        x_train=x_train,
        metadata=metadata,
        guess_range=256,
        epochs=50,
        batch_size=100,
        data_augmentation=False,
    )
    profile_engine.save_model("model.keras")
