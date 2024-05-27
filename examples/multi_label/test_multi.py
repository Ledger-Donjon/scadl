from scadl.multi_label_profile import matchEngine
from scadl.tools import sbox
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model


"""Leakage model"""
def leakage_model(metadata, guess):
    return sbox[guess ^ metadata["plaintext"][target_byte]]


if __name__ == "__main__":

    """loading traces and metadata for testing"""
    directory = "D:/stm32f3_aes_unprotected/"
    size_test = 50
    leakages = np.load(directory + "test/traces.npy")[0:size_test]
    metadata = np.load(directory + "test/combined_test.npy")[0:size_test]
    """selecting which key byte needs to be attacked"""
    target_byte = 0  # or 1
    """Probability range is selected based on the key byte """
    prob_range = (target_byte * 256, 256 + target_byte * 256)
    correct_key = metadata["key"][0][target_byte]

    """poi have the same indexes like the profiling phase"""
    poi = np.concatenate((leakages[:, 1315:1325], leakages[:, 1490:1505]), axis=1)

    """Loading the model"""
    model = load_model("multi_mlp.keras")

    """Matching process"""
    test_engine = matchEngine(model=model, leakage_model=leakage_model)
    rank, number_traces = test_engine.match(
        x_test=poi,
        metadata=metadata,
        guess_range=256,
        correct_key=correct_key,
        step=1,
        prob_range=prob_range,
    )

    """Plotting the key rank"""
    plt.plot(number_traces, rank)
    plt.ylabel(f"Rank of K[{target_byte}]")
    plt.xlabel("Number of traces")
    plt.show()
