from keras.layers import BatchNormalization, Dense, Input
from keras.models import Model
from keras.optimizers import Adam


def mlp_short_multi(len_samples: int, nb_bytes: int) -> Model:
    input_layer = Input(shape=(len_samples,))

    internal_layer = Dense(100, activation="relu")(input_layer)
    internal_layer = BatchNormalization()(internal_layer)

    output_layers = []
    for i in range(nb_bytes):
        output_layer = Dense(256, activation="softmax", name=f"byte_{i}")(
            internal_layer
        )
        output_layers.append(output_layer)
    model = Model(inputs=input_layer, outputs=output_layers)

    model.compile(
        loss=["categorical_crossentropy" for _ in range(nb_bytes)],
        optimizer=Adam(learning_rate=0.001),
        metrics=["accuracy" for _ in range(nb_bytes)],
    )
    return model


def show_loss_history(history):
    plt.plot(history.history["loss"], label="Training loss")
    plt.plot(history.history["val_loss"], label="Validation loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()

    plt.show()


if __name__ == "__main__":
    import sys
    from pathlib import Path

    import matplotlib.pyplot as plt
    import numpy as np
    from keras.callbacks import ModelCheckpoint
    from keras.utils import to_categorical
    from sklearn.model_selection import train_test_split

    from scadl.tools import sbox, standardize

    NB_BYTES = 16

    if len(sys.argv) != 2:
        print("Need to specify the location of the dataset")
        exit()

    # Load traces and metadata for training
    dataset_dir = Path(sys.argv[1])
    traces = np.load(dataset_dir / "train/traces.npy")
    metadata = np.load(dataset_dir / "train/combined_train.npy")

    # Prepare inputs and labels
    x = traces
    y = metadata

    x_train, x_test, metadata_train, metadata_test = train_test_split(
        x, y, test_size=0.1
    )

    x_train = standardize(x_train)
    x_test = standardize(x_test)

    sbox_vectorized = np.vectorize(lambda x: sbox[x], otypes=[np.uint8])

    def leakage_model_vectorized(metadata: np.ndarray) -> np.ndarray:
        return sbox_vectorized(np.bitwise_xor(metadata["plaintext"], metadata["key"]))

    y_train = [
        to_categorical(
            leakage_model_vectorized(metadata_train[:, i]),
            num_classes=256,
        )
        for i in range(NB_BYTES)
    ]

    y_test = [
        to_categorical(
            leakage_model_vectorized(metadata_test[:, i]),
            num_classes=256,
        )
        for i in range(NB_BYTES)
    ]

    # Build the model
    model = mlp_short_multi(x.shape[1], NB_BYTES)
    model.summary()

    callbacks = [
        ModelCheckpoint(
            "model.checkpoint.keras",
            monitor="val_loss",
            save_best_only=True,
        ),
    ]
    history = model.fit(
        x_train,
        y_train,
        epochs=200,
        batch_size=256,
        verbose=True,
        validation_data=(x_test, y_test),
        callbacks=callbacks,
    )

    show_loss_history(history)

    model.save("model.keras")
