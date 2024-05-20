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
    # return (1 & ((sbox[metadata[0] ^ guess]) >> 7))
    return (1 & ((sbox[metadata['plaintext'][0] ^ guess]) >> 7))
    # return 1 & ((sbox[metadata['plaintext'][0] ^ guess]))
    # return hw(sbox[metadata['plaintext'][0] ^ guess])


def remove_avg(traces):
    avg = np.average(traces, axis=0)
    return traces - avg



if __name__ == "__main__":
    directory = "D:/KARIM/projects/intenship/traces/"
    leakages = np.load(directory + 'test/traces.npy')[0:10000]
    metadata = np.load(directory + '/test/combined.npy')[0:10000]
    # directory = "D:/KARIM/projects/embedded_dev/r_dev/data/cw/"
    # leakages = np.load(directory + 'leakages.npy')
    # metadata = np.load(directory + 'plaintexts.npy')
    avg = remove_avg(leakages[:, 2040: 2060])
    x_train = normalization(avg)
    model = mlp_best() # mlp_non_profiling() 
    profile_engine = profileEngine(model, leakage_model=leakage_model)
    acc = profile_engine.train(x_train=x_train, metadata=metadata, key_range=range(165, 175), epochs=20, batch_size=1000)
    plt.plot(acc.T, 'grey')
    # plt.plot(acc[170], 'black')
    plt.show()
    



