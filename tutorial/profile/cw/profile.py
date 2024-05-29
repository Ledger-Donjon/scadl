from scadl.profile import profileEngine
from scadl import sbox, normalization
import sys

sys.path.append("../../models")
from cw_models import *


"""Leakage function"""


def leakage_model(metadata):
    """leakage model for sbox[0]"""
    return sbox[metadata["plaintext"][0] ^ metadata["key"][0]]


if __name__ == "__main__":
    """loading traces and metadata for training"""
    directory = "D:/stm32f3_aes_unprotected/train/"
    leakages = np.load(directory + "traces.npy")
    metadata = np.load(directory + "combined_train.npy")

    """Selecting poi where SNR gives the max value"""
    x_train = normalization(
        leakages[:, 1315:1325]
    )  # Normalization is used for improving the learning

    """Loading the DL model"""
    model = model_mlp()  # model_cnn(20, 256)

    """Profiling"""
    profile_engine = profileEngine(model, leakage_model=leakage_model)
    profile_engine.train(x_train=x_train, metadata=metadata, epochs=100, batch_size=100)

    """Save model"""
    profile_engine.save_model("model_mlp.keras")
