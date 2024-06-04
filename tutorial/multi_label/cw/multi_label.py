from scadl.multi_label_profile import multiLabelEngine
from scadl.tools import sbox, gen_labels

# from models import mlp_multi_label
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
import sys

sys.path.append("../../models")
from cw_models import mlp_multi_label, cnn_multi_label


def leakage_model(metadata, key_byte):
    return sbox[metadata["plaintext"][key_byte] ^ metadata["key"][key_byte]]


if __name__ == "__main__":

    """loading traces and metadata for training"""
    directory = "D:/stm32f3_aes_unprotected/train/"
    leakages = np.load(directory + "traces.npy")
    metadata = np.load(directory + "combined_train.npy")
    size_profiling = len(metadata)
    """poi for sbox[p0^k0] and sbox[p1^k1] -> k[0] and k[1]"""
    poi = np.concatenate((leakages[:, 1315:1325], leakages[:, 1490:1505]), axis=1)
    """"generate labels"""
    y_0 = gen_labels(
        leakage_model=leakage_model, metadata=metadata, key_byte=0
    ).reshape((size_profiling, 1))
    y_1 = gen_labels(
        leakage_model=leakage_model, metadata=metadata, key_byte=1
    ).reshape((size_profiling, 1))
    """shifting second label by 256"""
    combined_labels = np.concatenate(
        (y_0, y_1 + 256), axis=1
    )  # second labels are shifted by 256
    label = MultiLabelBinarizer()
    labels_fit = label.fit_transform(combined_labels)
    """load model"""
    len_samples = poi.shape[1]
    guess_range = 512

    # model = mlp_multi_label()
    model = cnn_multi_label(len_samples, 512)

    """call multi-label profiling engine"""
    profile = multiLabelEngine(model)
    profile.train(x_train=poi, y_train=labels_fit, epochs=100)
    profile.save_model("multi_cnn.keras")
