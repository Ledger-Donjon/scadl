import sys
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
from keras.models import load_model

from scadl.profile import Match
from scadl.tools import normalization, remove_avg, sbox


def leakage_model(data: np.ndarray, guess: int) -> int:
    return sbox[guess ^ data["plaintext"][2]]


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Need to specify the location of training data")
        exit()
    dataset_dir = Path(sys.argv[1])

    # Load traces and metadata for attack
    file = h5py.File(dataset_dir / "ASCAD.h5", "r")
    leakages = file["Attack_traces"]["traces"][:]
    metadata = file["Attack_traces"]["metadata"][:]

    # correct key value to estimate the rank against
    correct_key = metadata["key"][0][2]

    # Select POIs where SNR gives the max value. It should have the same index
    # like what is used in the profiling phase.
    poi = leakages

    # Same preprocessing as for the training
    poi = normalization(remove_avg(poi), feature_range=(-1, 1))

    # Load the model
    model = load_model("model.keras")
    SIZE = 1000
    TRIALS = 20
    test_engine = Match(model=model, leakage_model=leakage_model)

    for i in range(TRIALS):
        index = np.random.randint(len(leakages) - SIZE)
        sample_poi = poi[index : index + SIZE]
        sample_metadata = metadata[index : index + SIZE]
        # Test the correct key rank
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

    # Plot the result
    FONT_SIZE = 2
    LINE_WIDTH = 2
    plt.plot(number_traces, avg_rank, "black", linewidth=LINE_WIDTH)
    plt.xlabel("Number of traces", fontsize=FONT_SIZE)
    plt.ylabel("Average rank of K[2] ", fontsize=FONT_SIZE)
    plt.xticks(fontsize=25)
    plt.yticks(fontsize=25)
    plt.show()
