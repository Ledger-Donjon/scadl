import sys
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
from scadl.multi_label_profile import MatchMultiLabel
from scadl.tools import sbox


TARGET_BYTE = 1  # or 0


def leakage_model(data, guess):
    """Leakage function"""
    return sbox[guess ^ data["plaintext"][TARGET_BYTE]]


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Need to specify the location of testing data")
    DIR = sys.argv[1]

    SIZE = 50
    leakages = np.load(DIR + "/test/traces.npy")[0:SIZE]
    metadata = np.load(DIR + "/test/combined_test.npy")[0:SIZE]
    """selecting which key byte needs to be attacked"""

    prob_range = (TARGET_BYTE * 256, 256 + TARGET_BYTE * 256)
    correct_key = metadata["key"][0][TARGET_BYTE]

    """poi have the same indexes like the profiling phase"""
    poi = np.concatenate((leakages[:, 1315:1325], leakages[:, 1490:1505]), axis=1)

    """Loading the model"""
    model = load_model("model.keras")

    """Matching process"""
    test_engine = MatchMultiLabel(model=model, leakage_model=leakage_model)
    rank, number_traces = test_engine.match(
        x_test=poi,
        metadata=metadata,
        guess_range=256,
        correct_key=correct_key,
        step=1,
        prob_range=prob_range,
    )

    """Plotting the key rank"""
    plt.plot(number_traces, rank, "black", linewidth=5)
    plt.xlabel("Number of traces", fontsize=40)
    plt.ylabel("Average rank of K[1] ", fontsize=40)
    plt.xticks(fontsize=25)
    plt.yticks(fontsize=25)
    plt.show()
