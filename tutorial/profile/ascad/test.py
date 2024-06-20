from scadl.profile import matchEngine
from scadl.tools import sbox, normalization, remove_avg
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
import h5py


def leakage_model(metadata, guess):
    return sbox[guess ^ metadata["plaintext"][2]]


def rot(traces):
    len_trace = traces.shape[1]
    tmp_traces = traces
    for i, trace in enumerate(traces):
        index = np.random.randint(len_trace)
        tmp_traces[i] = np.concatenate((trace[index:], trace[0:index]))
    return tmp_traces


if __name__ == "__main__":
    """loading traces and metadata for testing"""
    directory = "D:/ascad/ASCAD.h5"
    file = h5py.File(directory, "r")
    leakages = file["Attack_traces"]["traces"][:]
    metadata = file["Attack_traces"]["metadata"][:]

    """correct key value to test it's rank"""
    correct_key = metadata["key"][0][2]

    """Selecting poi where SNR gives the max value and it should have 
    the same index like what is used in the profiling phase """
    poi = np.concatenate((leakages[:, 515:520], leakages[:, 148:158]), axis=1)
    poi = normalization(
        remove_avg(poi)
    )  
    
    # Normalization is used for improving the learning

    """Loading the DL model"""
    model = load_model("model_mlp.keras")
    len_test = 1000
    no_trials = 50
    test_engine = matchEngine(model=model, leakage_model=leakage_model)

    for i in range(no_trials):
        index = np.random.randint(len(leakages) - len_test)
        sample_poi = poi[index : index + len_test]
        sample_metadata = metadata[index : index + len_test]
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
    avg_rank = avg_rank / no_trials

    """Plotting the result"""
    plt.plot(number_traces, avg_rank)
    plt.xlabel("Number of traces")
    plt.ylabel("K[2] rank")
    plt.show()
