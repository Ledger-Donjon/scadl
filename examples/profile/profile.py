from scadl import profileEngine, sbox, normalization
import sys
sys.path.append('../models')
from used_models import *

def leakage_model(metadata):
    return sbox[metadata['plaintext'][0] ^ metadata['key'][0]]

if __name__ == "__main__":
    directory = "D:/KARIM/projects/intenship/traces/"
    leakages = np.load(directory + 'profile/traces.npy')[0:50000]
    metadata = np.load(directory + '/profile/combined.npy')[0:50000]
    x_train = normalization(leakages[:, 1940: 1960])
    model = model_mlp() #model_cnn(20, 256)
    profile_engine = profileEngine(model, leakage_model=leakage_model)
    profile_engine.train(x_train=x_train, metadata=metadata, epochs=5, batch_size=100)
    profile_engine.save_model("model_mlp.model")








