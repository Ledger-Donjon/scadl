from scadl.non_profile import profileEngine
from scadl import sbox, normalization, remove_avg
import sys

sys.path.append("../../models")
from cw_models import cnn_best, mlp_non_profiling
import matplotlib.pyplot as plt
import numpy as np


"""leakage model"""


def leakage_model(metadata, guess):
    # return 1 & ((sbox[metadata["plaintext"][0] ^ guess]) >> 7) #msb
    return 1 & ((sbox[metadata["plaintext"][0] ^ guess]))  # lsb
    # return hw(sbox[metadata['plaintext'][0] ^ guess]) #hw


if __name__ == "__main__":
    """loading traces"""
    directory = "D:/stm32f3_aes_unprotected/test/"
    leakages = np.load(directory + "traces.npy")
    metadata = np.load(directory + "combined_test.npy")
    correct_key = metadata["key"][0][0]

    """Subtracting average from traces + normalization"""
    avg = remove_avg(leakages[:, 1315:1325])
    x_train = normalization(avg)

    """Selecting the model"""
    key_range = 2
    len_samples = x_train.shape[1]
    model = mlp_non_profiling()
    # model = cnn_best(len_samples, key_range)

    """Non-profiling DL"""
    profile_engine = profileEngine(model, leakage_model=leakage_model)
    acc = profile_engine.train(
        x_train=x_train,
        metadata=metadata,
        hist_acc="val_accuracy",
        key_range=range(0, 256),
        epochs=50,
        batch_size=1000,
    )

    """Selecting the key with the highest accuracy key"""
    guessed_key = np.argmax(np.max(acc, axis=1))
    print(f"guessed key = {guessed_key}")

    """Plotting """
    plt.plot(acc.T, "grey")
    plt.plot(acc[correct_key], "black")
    plt.show()
