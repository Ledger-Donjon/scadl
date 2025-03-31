import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.layers import Dense, Input
from keras.models import Sequential
from scadl.augmentation import Mixup
from scadl.profile import Profile
from scadl.tools import normalization
import tensorflow as tf
import innvestigate
tf.compat.v1.disable_eager_execution()


def aug_mixup(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Data augmentation based on mixup"""
    mix = Mixup()
    x, y = mix.generate(x_train=x, y_train=y, ratio=1, alpha=1)
    return x, y

def model_mlp(sample_len: int, range_outer_layer: int) -> keras.Model:
    """param sample_len: number of samples
       param range_outer_layer: Number of guess
    """
    model = Sequential()
    model.add(Input(shape=(sample_len,)))
    model.add(Dense(50, activation="relu"))
    model.add(Dense(50, activation="relu"))
    model.add(Dense(range_outer_layer, activation="softmax"))
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model

def leakage_model(data: np.ndarray) -> int:
    """leakage model"""
    return data


def simulator (len_traces: int, len_samples: int, value: int, randomize=False) -> np.ndarray:
    """A leakage value simulator"""
    leakages = np.random.uniform(1, 1.1, size=(len_traces, len_samples))
    if randomize:
        for index in range(len_traces):
            random_offset = np.random.randint(len_samples)
            leakages[index, random_offset] = value
    return leakages



def handy_ttest(group_a: np.ndarray, group_b: np.ndarray) -> np.ndarray:
    """t-test engine"""
    mean_a = np.mean(group_a, axis=0)
    mean_b = np.mean(group_b, axis=0)
    var_a = np.var(group_a, axis=0)
    var_b = np.var(group_b, axis=0)
    dec_1 = var_a / group_a.shape[0]
    dec_2 = var_b / group_b.shape[0]
    dec = np.sqrt(dec_1 + dec_2)
    num = mean_a - mean_b
    return num / dec



def main():
    """main function"""
    LEN_TRACES = 50000
    LEN_SAMPLES = 500
    VALUE_UNPROTECT = 10
    traces_unprotect = simulator(len_traces=LEN_TRACES, len_samples=LEN_SAMPLES, value=VALUE_UNPROTECT, randomize=True)
    VALUE_PROTECT = 20
    traces_protect = simulator(len_traces=LEN_TRACES, len_samples=LEN_SAMPLES, value=VALUE_PROTECT, randomize=True)
    traces_protect = simulator(len_traces=LEN_TRACES, len_samples=LEN_SAMPLES, value=VALUE_PROTECT, randomize=True)
    t_test = handy_ttest(traces_protect, traces_unprotect)
    dif_mean = abs(np.average(traces_unprotect, axis=0) - np.average(traces_protect, axis=0))
    plt.style.use("dark_background")
    fig, (ax0, ax1) = plt.subplots(2)
    ax0.plot(traces_unprotect[0:100].T)
    ax0.plot(traces_protect[0:100].T)
    ax1.plot(dif_mean) # t_test
    plt.show()
    
    
    labels_0 = np.zeros(LEN_TRACES)
    labels_1 = np.ones(LEN_TRACES)
    leakages = np.concatenate((traces_unprotect, traces_protect), axis=0)
    metadata = np.concatenate((labels_0, labels_1), axis=0)
    x_train = normalization((leakages), feature_range=(-1, 1))
    GUESS_RANGE = 2
    model = model_mlp(LEN_SAMPLES, GUESS_RANGE)
        # Train the model
    profile_engine = Profile(model, leakage_model=leakage_model)
    EPOCHS = 5
    profile_engine.data_augmentation(aug_mixup)
    profile_engine.train(
        x_train=x_train,
        metadata=metadata,
        guess_range=GUESS_RANGE,
        epochs=EPOCHS,
        batch_size=10,
        validation_split=0.1,
        data_augmentation=False,
    )

    # plt.plot(profile_engine.history.history['loss'], 'r')
    # plt.plot(profile_engine.history.history['val_loss'], 'b')
    # plt.show()

    model_wo_sm = innvestigate.model_wo_softmax(model)
    # gradient_analyzer = innvestigate.analyzer.Gradient(model_wo_sm)
    gradient_analyzer = innvestigate.analyzer.DeepTaylor(model_wo_sm)
    vis_trace = np.zeros(LEN_SAMPLES)

    for i in range(LEN_TRACES):
        trace_sample = traces_protect[i]
        trace = trace_sample.reshape(1, LEN_SAMPLES)
        prob = model.predict(trace)
        vis_trace = gradient_analyzer.analyze(trace)[0]
        fig, (ax0, ax1, ax2) = plt.subplots(3)
        ax0.plot(traces_protect[0:100].T)
        ax0.plot(traces_unprotect[0:100].T)
        ax1.plot(trace_sample, 'blue')
        ax2.plot(abs(vis_trace), 'red')
        plt.show()

if __name__ == "__main__":
    main()
