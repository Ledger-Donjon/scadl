import sys
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, Dense, Flatten
from scadl.non_profile import NonProfile
from scadl.tools import sbox, normalization, remove_avg


TARGET_BYTE = 0


def mlp_non_profiling(len_smaples):
    """It retrurns an MLP model"""
    model = Sequential()
    model.add(Dense(20, input_dim=len_smaples, activation=tf.nn.relu))
    model.add(Dense(10, activation=tf.nn.relu))
    model.add(Dense(2, activation=tf.nn.softmax))
    model.compile(optimizer="adam", loss="mean_squared_error", metrics=["accuracy"])
    return model


def mlp_best(node=200, inner_layers=4):
    """It retrurns an MLP model"""
    model = Sequential()
    model.add(Dense(node, activation="relu"))  # 28   #node
    for i in range(inner_layers):
        model.add(Dense(node, activation="relu"))
        # Dropout(0.1)  # Dropout(0.01)
        # BatchNormalization()
    model.add(Dense(2, activation="softmax"))
    optimizer = "adam"  # keras.optimizers.Adam(learning_rate=0.01) #RMSprop(lr=0.00001)# 'adam'#RMSprop(lr=0.00001)
    model.compile(
        loss="mean_squared_error", optimizer=optimizer, metrics=["accuracy"]
    )  # categorical_crossentropy  mean_squared_error
    return model


def cnn_best(len_samples, guess_range):
    """It retrurns a CNN model"""
    model = Sequential()
    model.add(Conv1D(filters=20, kernel_size=5, input_shape=(len_samples, 1)))
    model.add(MaxPooling1D(pool_size=5))
    model.add(Flatten())
    model.add(Dense(200, activation="relu"))
    model.add(Dense(guess_range, activation="softmax"))
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


def leakage_model(metadata, guess):
    """It returns the leakage function"""
    # return 1 & ((sbox[metadata["plaintext"][TARGET_BYTE] ^ guess]) >> 7) #msb
    return 1 & ((sbox[metadata["plaintext"][TARGET_BYTE] ^ guess]))  # lsb
    # return hw(sbox[metadata['plaintext'][TARGET_BYTE] ^ guess]) #hw


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Need to specify the location of training data")
        exit()

    DIR = sys.argv[1]
    leakages = np.load(DIR + "/test/traces.npy")[0:3000]
    metadata = np.load(DIR + "/test/combined_test.npy")[0:3000]
    correct_key = metadata["key"][0][0]

    """Subtracting average from traces + normalization"""
    avg = remove_avg(leakages[:, 1315:1325])
    x_train = normalization(avg)  # normalization(avg)

    """Selecting the model"""
    model_dl = mlp_non_profiling(x_train.shape[1])
    # model_dl = mlp_non_profiling(x_train.shape)
    # model = cnn_best(x_train.shape[1], key_range)

    """Non-profiling DL"""
    profile_engine = NonProfile(model_dl, leakage_model=leakage_model)
    acc = profile_engine.train(
        x_train=x_train,
        metadata=metadata,
        hist_acc="accuracy",
        key_range=range(0, 256),
        num_classes=2,
        epochs=200,
        batch_size=1000,
    )

    """Selecting the key with the highest accuracy key"""
    guessed_key = np.argmax(np.max(acc, axis=1))
    print(f"guessed key = {guessed_key}")
    plt.plot(acc.T, "grey", linewidth=2)
    plt.plot(acc[correct_key], "black", linewidth=2)
    plt.xlabel("Number of epochs", fontsize=40)
    plt.ylabel("Accuracy ", fontsize=40)
    plt.xticks(fontsize=25)
    plt.yticks(fontsize=25)
    plt.show()
