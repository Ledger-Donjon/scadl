from scadl.profile import profileEngine
from scadl import sbox, normalization
import sys
import h5py
from scadl.tools import remove_avg
import numpy as np

sys.path.append("../../models")
from ascad_models import *


"""Leakage function"""


def leakage_model(metadata):
    """leakage model for sbox[2]"""
    return sbox[metadata["plaintext"][2] ^ metadata["key"][2]]


if __name__ == "__main__":
    """loading traces and metadata for training"""
    directory = "D:/ascad/ASCAD.h5"
    file = h5py.File(directory, "r")
    leakages = file["Profiling_traces"]["traces"][:]
    metadata = file["Profiling_traces"]["metadata"][:]

    """Selecting poi where SNR gives the max value"""
    poi = np.concatenate((leakages[:, 515:520], leakages[:, 148:158]), axis=1)

    """Processing the traces"""
    x_train = normalization(
        remove_avg(poi)
    )  # normalization(remove_avg(leakages)) #normalization(leakages)  # Normalization is used for improving the learning
    len_samples = x_train.shape[1]
    guess_range = 256

    """Loading the DL model mlp"""
    model = mlp_short(len_samples)
    # model = model_cnn(len_samples, guess_range)

    """Profiling"""
    profile_engine = profileEngine(model, leakage_model=leakage_model)
    profile_engine.train(
        x_train=x_train,
        metadata=metadata,
        epochs=100,
        batch_size=128,
        validation_split=0.02,
    )

    """Save model"""
    profile_engine.save_model("model_mlp.keras")
