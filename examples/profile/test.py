from scadl.profile import matchEngine
from scadl.tools import sbox, normalization
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model


def leakage_model(metadata, guess):
    return sbox[guess ^ metadata["plaintext"][0]]


if __name__ == "__main__":
    """loading traces and metadata for testing"""
    directory = "D:/stm32f3_aes_unprotected/"
    size_test = 50
    leakages = np.load(directory + "test/traces.npy")[0:size_test]
    metadata = np.load(directory + "test/combined_test.npy")[0:size_test]

    """correct key value to test it's rank"""
    correct_key = metadata["key"][0][0]

    """Selecting poi where SNR gives the max value and it should have 
    the same index like what is used in the profiling phase """
    poi = normalization(
        leakages[:, 1315:1325]
    )  # Normalization is used for improving the learning

    """Loading the DL model"""
    model = load_model("model_mlp.keras")

    """Testing the correct key rank"""
    test_engine = matchEngine(model=model, leakage_model=leakage_model)
    rank, number_traces = test_engine.match(
        x_test=poi, metadata=metadata, guess_range=256, correct_key=correct_key, step=1
    )

    """Plotting the result"""
    plt.plot(number_traces, rank)
    plt.xlabel("Number of traces")
    plt.ylabel("K[0] rank")
    plt.show()
