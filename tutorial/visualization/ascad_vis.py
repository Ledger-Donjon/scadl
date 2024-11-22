import sys
from pathlib import Path

import h5py
import innvestigate
import keras
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras.layers import Dense, Input
from keras.models import Sequential

from scadl.augmentation import Mixup, RandomCrop
from scadl.profile import Profile
from scadl.tools import normalization, remove_avg, sbox

tf.compat.v1.disable_eager_execution()


def leakage_model(data):
    """leakage model for sbox[2]"""
    return sbox[data["plaintext"][2] ^ data["key"][2]]


def aug_mixup(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Data augmentation based on mixup"""
    mix = Mixup()
    x, y = mix.generate(x_train=x, y_train=y, ratio=1, alpha=1)
    return x, y


def aug_crop(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Data augmentation based on random crop"""
    mix = RandomCrop()
    x, y = mix.generate(x_train=x, y_train=y, ratio=1, window=5)
    return x, y


def mlp_short(len_samples: int) -> keras.Model:
    """
    param len_samples: size of a single trace
    """
    model = Sequential()
    model.add(Input(shape=(len_samples,)))
    model.add(Dense(20, activation="relu"))
    # BatchNormalization()
    model.add(Dense(50, activation="relu"))
    model.add(Dense(256, activation="softmax"))
    model.compile(
        loss="categorical_crossentropy",
        optimizer="adam",
        metrics=["accuracy"],
    )
    return model


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Need to specify the location of training data and model")
        exit()
    dataset_dir = Path(sys.argv[1])

    # Load traces and metadata for training
    file = h5py.File(dataset_dir / "ASCAD.h5", "r")
    leakages = file["Profiling_traces"]["traces"][:]
    metadata = file["Profiling_traces"]["metadata"][:]

    # Select POIs where SNR is high
    poi = leakages

    # Preprocess traces
    x_train = normalization(remove_avg(poi), feature_range=(-1, 1))
    GUESS_RANGE = 256

    # Build the model
    model = mlp_short(x_train.shape[1])
    # Train the model
    profile_engine = Profile(model, leakage_model=leakage_model)
    profile_engine.data_augmentation(aug_mixup)
    profile_engine.train(
        x_train=x_train,
        metadata=metadata,
        guess_range=256,
        epochs=25,
        batch_size=128,
        validation_split=0.1,
        data_augmentation=False,
    )
    model = profile_engine.model
    # Call test traces
    test_traces = file["Attack_traces"]["traces"][:]
    test_metadata = file["Attack_traces"]["metadata"][:]
    test_traces = normalization(remove_avg(test_traces), feature_range=(-1, 1))
    model_wo_sm = innvestigate.model_wo_softmax(model)
    gradient_analyzer = innvestigate.analyzer.Gradient(model_wo_sm)
    vis_trace = np.zeros(700)
    for index, trace_sample in enumerate(test_traces):
        trace = trace_sample.reshape(1, 700)
        prob = model.predict(trace)
        vis_trace += gradient_analyzer.analyze(trace)[0]
    plt.plot(abs(vis_trace / len(test_traces)))
    plt.show()
