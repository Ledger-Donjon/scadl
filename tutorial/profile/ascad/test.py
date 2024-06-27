import sys
import numpy as np
import matplotlib.pyplot as plt
import h5py
from keras.models import load_model
from scadl.profile import Match
from scadl.tools import sbox, normalization, remove_avg


def leakage_model(data, guess):
    """It returns the leakage model"""
    return sbox[guess ^ data["plaintext"][2]]


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Need to specify the location of training data")
        exit()
    DIR = sys.argv[1]

    """loading traces and metadata for training"""
    file = h5py.File(f"{DIR}/ASCAD.h5", "r")
    leakages = file["Attack_traces"]["traces"][:]
    metadata = file["Attack_traces"]["metadata"][:]

    """correct key value to test it's rank"""
    correct_key = metadata["key"][0][2]

    """Selecting poi where SNR gives the max value and it should have 
    the same index like what is used in the profiling phase """
    poi = np.concatenate((leakages[:, 515:520], leakages[:, 148:158]), axis=1)
    poi = normalization(remove_avg(poi))

    # Normalization is used for improving the learning

    """Loading the DL model"""
    model = load_model("model.keras")
    SIZE = 1000
    TRIALS = 20
    test_engine = Match(model=model, leakage_model=leakage_model)

    for i in range(TRIALS):
        index = np.random.randint(len(leakages) - SIZE)
        sample_poi = poi[index : index + SIZE]
        sample_metadata = metadata[index : index + SIZE]
        """Testing the correct key rank"""
        rank, number_traces = test_engine.match(
            x_test=sample_poi,
            metadata=sample_metadata,
            guess_range=256,
            correct_key=correct_key,
            step=10,
        )
        if i == 0:
            avg_rank = rank
        else:
            avg_rank += rank
    avg_rank = avg_rank / TRIALS

    """Plotting the result"""
    plt.plot(number_traces, avg_rank, "black", linewidth=5)
    plt.xlabel("Number of traces", fontsize=40)
    plt.ylabel("Average rank of K[2] ", fontsize=40)
    plt.xticks(fontsize=25)
    plt.yticks(fontsize=25)
    plt.show()
