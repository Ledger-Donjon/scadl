if __name__ == "__main__":
    import sys
    from pathlib import Path

    import matplotlib.pyplot as plt
    import numpy as np
    from keras.models import load_model

    from scadl.multi_task import compute_guessing_entropy
    from scadl.tools import sbox, standardize

    NB_BYTES = 16

    if len(sys.argv) != 2:
        print("Need to specify the location of the dataset")
        exit()

    dataset_dir = Path(sys.argv[1])

    # Load traces and metadata for the attack
    dataset_dir = Path(sys.argv[1])
    traces = np.load(dataset_dir / "test/traces.npy")
    metadata = np.load(dataset_dir / "test/combined_test.npy")

    correct_key = metadata["key"][0]

    traces = standardize(traces)

    model = load_model("model.keras")

    predictions = model.predict(traces)

    for i in range(NB_BYTES):
        guessing_entropy, number_traces = compute_guessing_entropy(
            predictions[i],
            lambda data, guess: sbox[guess ^ int(data["plaintext"][i])],
            metadata,
            256,
            correct_key[i],
            1,
            3,
        )
        plt.plot(number_traces, guessing_entropy)
    plt.show()
