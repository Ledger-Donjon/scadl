
from keras.models import Sequential, Model
from keras.layers import Conv1D, Dense, Flatten
from keras.layers import Dropout
from keras.optimizers import RMSprop
import keras
from keras.layers import Input, AveragePooling1D
from keras.layers import BatchNormalization


# In case of using the overall 700 samples
def mlp_ascad(node=200, layer_nb=5):
    model = Sequential()
    model.add(Dense(node, input_dim=700, activation="relu"))
    for i in range(layer_nb):
        model.add(Dense(node, activation="relu"))
        # Dropout(0.2)    #Dropout(0.01)
        # BatchNormalization()
    model.add(Dense(256, activation="softmax"))

    optimizer = RMSprop(learning_rate=0.00001)
    model.compile(
        loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"]
    )
    return model


### In case of using only snr peaks
def mlp_short(len_samples):
    model = Sequential()
    model.add(Dense(20, input_dim=len_samples, activation="relu"))
    # GaussianNoise(stddev)
    BatchNormalization()
    model.add(Dense(50, activation="relu"))
    model.add(Dense(256, activation="softmax"))
    optimizer = RMSprop(lr=0.00001)
    model.compile(
        loss=keras.losses.CategoricalCrossentropy(),
        optimizer=optimizer,
        metrics=["accuracy"],
    )
    return model


def cnn_ascad():
    # From VGG16 design
    input_shape = (700, 1)
    img_input = Input(shape=input_shape)
    # Block 1
    x = Conv1D(32, 32, activation="relu", padding="same", name="block1_conv1")(
        img_input
    )  # 8
    Dropout(0.1)
    x = AveragePooling1D(2, strides=2, name="block1_pool")(x)

    # Classification block
    x = Flatten(name="flatten")(x)
    # GaussianNoise(stddev=0.2)
    x = Dense(400, activation="relu", name="fc1")(x)
    x = Dense(400, activation="relu", name="fc2")(x)

    x = Dense(256, activation="softmax", name="predictions")(x)
    inputs = img_input
    # Create model.
    model = Model(inputs, x, name="cnn_test")
    optimizer = RMSprop(lr=0.00001)  # SGD(lr=INIT_LR, momentum=0.9)#'adam'
    model.compile(
        loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"]
    )
    return model
