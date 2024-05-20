from scadl import multiLabelEngine, matchEngine, sbox
# from models import mlp_multi_label
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
import sys
sys.path.append("../../models")
from used_models import mlp_multi_label

def leakage_model(metadata, key_byte):
    return sbox[metadata['plaintext'][key_byte] ^ metadata['key'][key_byte]]

def gen_labels(metadata, key_byte):
    return np.array([leakage_model(i, key_byte=key_byte) for i in metadata])

if __name__ == "__main__":
    directory = "D:/KARIM/projects/intenship/traces/"
    size_profiling = 50000
    leakages = np.load(directory + 'profile/traces.npy')[0:50000]
    metadata = np.load(directory + '/profile/combined.npy')[0:50000]
    """poi for sbox[p0^k0] and sbox[p1^k1]""" 
    poi=np.concatenate((leakages[:,4096:4100],leakages[:,17180:17188]),axis=1) 
    """"generate labels"""
    y_0 = gen_labels(metadata=metadata, key_byte=0).reshape((size_profiling, 1))
    y_1 = gen_labels(metadata=metadata, key_byte=1).reshape((size_profiling, 1))
    """shifting second label by 256"""
    combined_labels = np.concatenate((y_0, y_1 + 256), axis=1)
    label = MultiLabelBinarizer()
    labels_fit = label.fit_transform(combined_labels)
    """load model"""
    model = mlp_multi_label()
    """call multi-label profiling engine"""
    profile = multiLabelEngine(model)
    profile.train(x_train=poi,y_train=labels_fit)
    profile.save_model("multi_mlp.model")

    







