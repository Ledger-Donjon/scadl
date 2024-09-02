import sys
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
from scadl.profile import Match
from scadl.tools import sbox, normalization


def leakage_model(data, guess):
    """leakage function takes the data and guess. It returns the leakage model"""
    return sbox[guess ^ data["plaintext"][0]]


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Need to specify the location of testing data")
    DIR = sys.argv[1]
    SIZE_TEST = 50
    leakages = np.load(DIR + "/test/traces.npy")[0:SIZE_TEST]
    metadata = np.load(DIR + "/test/combined_test.npy")[0:SIZE_TEST]
    correct_key = metadata["key"][0][0]
    poi = normalization(
        leakages[:, 1315:1325]
    )  # Normalization is used for improving the learning
    model = load_model("model.keras")
    test_engine = Match(model=model, leakage_model=leakage_model)
    rank, number_traces = test_engine.match(
        x_test=poi, metadata=metadata, guess_range=256, correct_key=correct_key, step=1
    )
    """Plotting the result"""
    FONT_SIZE = 2
    LINE_WIDTH = 2
    plt.plot(number_traces, rank, "black", linewidth=LINE_WIDTH)
    plt.xlabel("Number of traces", fontsize=FONT_SIZE)
    plt.ylabel("Average rank of K[0] ", fontsize=FONT_SIZE)
    plt.xticks(fontsize=25)
    plt.yticks(fontsize=25)
    plt.show()
