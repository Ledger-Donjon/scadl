from scadl.multi_label_profile import matchEngine
from scadl.tools import sbox
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model


def leakage_model(metadata, guess):
    return sbox[guess ^ metadata['plaintext'][0]]


if __name__ == "__main__":
    directory = "D:/KARIM/projects/intenship/traces/"
    size_test = 50
    leakages = np.load(directory + 'test/traces.npy')[0:size_test]
    metadata = np.load(directory + 'test/combined.npy')[0:size_test]
    poi=np.concatenate((leakages[:,4096:4100],leakages[:,17180:17188]),axis=1)
    model = load_model("multi_mlp.model")
    test_engine = matchEngine(model=model, leakage_model=leakage_model)
    rank, x = test_engine.match(x_test=poi, metadata=metadata, guess_range=256, correct_key=170, step=1, prob_rang=(0, 256))
    plt.plot(x, rank)
    plt.show()



