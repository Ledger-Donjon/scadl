from scadl import profileEngine, sbox, matchEngine, normalization
import numpy as np
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

def leakage_model(metadata, guess):
    return sbox[guess ^ metadata['plaintext'][0]]


if __name__ == "__main__":
    directory = "D:/KARIM/projects/intenship/traces/"
    size_test = 50
    leakages = np.load(directory + 'test/traces.npy')[0:size_test]
    metadata = np.load(directory + 'test/combined.npy')[0:size_test]
    x_test = normalization(leakages[:, 1940: 1960])
    model = load_model("model_mlp.model")
    test_engine = matchEngine(model, leakage_model=leakage_model)
    rank, x_rank = test_engine.match(x_test=x_test, metadata=metadata, guess_range=256,
                       correct_key=170, step=1)
    plt.plot(x_rank, rank)
    plt.show()


