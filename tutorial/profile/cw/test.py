import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from keras.models import load_model

from scadl.profile import Match
from scadl.tools import normalization, sbox


def leakage_model(data: np.ndarray, guess: int) -> int:
    return sbox[guess ^ int(data["plaintext"][0])]


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Need to specify the location of testing data")
        exit()

    dataset_dir = Path(sys.argv[1])
    SIZE_TEST = 50
    leakages = np.load(dataset_dir / "test/traces.npy")[0:SIZE_TEST]
    metadata = np.load(dataset_dir / "test/combined_test.npy")[0:SIZE_TEST]

    correct_key = metadata["key"][0][0]
    # Select the same POIs and apply the same preprocessing as in the training
    poi = normalization(leakages[:, 1315:1325])

    # Load the model and evaluate the rank
    model = load_model("model.keras")
    test_engine = Match(model=model, leakage_model=leakage_model)
    rank, number_traces = test_engine.match(
        x_test=poi, metadata=metadata, guess_range=256, correct_key=correct_key, step=1
    )

    # Plot the result
    plt.plot(number_traces, rank, "black")
    plt.xlabel("Number of traces")
    plt.ylabel("Average rank of K[0]")
    plt.show()
