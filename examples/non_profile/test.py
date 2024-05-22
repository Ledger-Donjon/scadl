from scadl.non_profile import profileEngine
from scadl import sbox, normalization
import sys

sys.path.append('../models')
from used_models import mlp_non_profiling, mlp_best
import matplotlib.pyplot as plt
import numpy as np

def hw(a):
    return bin(a).count('1')

def leakage_model(metadata, guess):
    return (1 & ((sbox[metadata['plaintext'][1] ^ guess]) >> 7))
    # return (1 & ((sbox[metadata['plaintext'][1] ^ guess])))
    # return hw(sbox[metadata['plaintext'][0] ^ guess])


def remove_avg(traces):
    avg = np.average(traces, axis=0)
    return traces - avg



if __name__ == "__main__":
    directory = "D:/test/"
    leakages = np.load(directory + 'poi_key_1.npy')[0:10000]
    metadata = np.load(directory + 'combined.npy')[0:10000]
    avg = remove_avg(leakages)
    x_train = normalization(avg)
    model = mlp_best() 
    profile_engine = profileEngine(model, leakage_model=leakage_model)
    acc = profile_engine.train(x_train=x_train, metadata=metadata, key_range=range(256), epochs=10, batch_size=1000)
    plt.plot(acc.T, 'grey')
    plt.plot(acc[126], 'black')
    plt.show()
