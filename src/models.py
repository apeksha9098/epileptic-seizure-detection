"""
models.py  -  the three deep-learning models of the project.

They are built exactly as described in the project report (Tables 1, 2 and 3):

    LSTM      10 LSTM layers + 2 dense layers          input shape (1, 178)
    CNN       3 convolution blocks + 2 dense layers    input shape (178, 1)
    CNN-LSTM  2 conv layers + 5 LSTM layers + 2 dense  input shape (178, 1)    <- best model

Small differences to the report (and why):
    * Last layer = softmax with 2 neurons. The report wrote "sigmoid"; softmax is the
      mathematically correct partner of "categorical cross-entropy" and gives
      probabilities that add up to 1. The number of parameters stays the same.
"""
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

from . import config

# Which input shape every model needs
INPUT_KIND = {"LSTM": "lstm", "CNN": "cnn", "CNN-LSTM": "cnn"}


def build_lstm(n_features: int = config.N_FEATURES) -> keras.Model:
    """LSTM model - Table 1 of the report (197,090 trainable parameters)."""
    units_and_dropout = [          # (LSTM neurons, dropout after it)
        (64, 0.6), (34, 0.5), (64, 0.5), (32, 0.5), (32, 0.5),
        (32, 0.5), (32, 0.6), (50, 0.6), (50, 0.6),
    ]
    model = keras.Sequential(name="LSTM")
    model.add(keras.Input(shape=(1, n_features)))
    for units, drop in units_and_dropout:
        model.add(layers.LSTM(units, activation="relu", return_sequences=True))
        model.add(layers.Dropout(drop))
    model.add(layers.LSTM(50))                       # 10th LSTM layer (returns one vector)
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(32, name="features"))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(2, activation="softmax"))
    return model


def build_cnn(n_features: int = config.N_FEATURES) -> keras.Model:
    """1D-CNN model - Table 2 of the report (8,626 trainable parameters)."""
    model = keras.Sequential(name="CNN")
    model.add(keras.Input(shape=(n_features, 1)))

    # block 1:  64 filters, size 5, stride 2
    model.add(layers.Conv1D(64, 5, strides=2, padding="valid"))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation("relu"))
    model.add(layers.MaxPooling1D(2))
    model.add(layers.Dropout(0.6))

    # block 2 and 3:  24 filters, size 3, stride 2
    for drop in (0.5, 0.5):
        model.add(layers.Conv1D(24, 3, strides=2, padding="valid"))
        model.add(layers.BatchNormalization())
        model.add(layers.Activation("relu"))
        model.add(layers.MaxPooling1D(2))
        model.add(layers.Dropout(drop))

    model.add(layers.Flatten())
    model.add(layers.Dense(32, name="features"))
    model.add(layers.Activation("relu"))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(2, activation="softmax"))
    return model


def build_cnn_lstm(n_features: int = config.N_FEATURES) -> keras.Model:
    """CNN-LSTM hybrid - Table 3 of the report (127,578 trainable parameters)."""
    model = keras.Sequential(name="CNN-LSTM")
    model.add(keras.Input(shape=(n_features, 1)))

    # Part 1: convolution (finds small shapes in the signal)  +  LSTM (remembers the order)
    model.add(layers.Conv1D(64, 5, strides=2, padding="valid"))
    model.add(layers.LSTM(34, activation="tanh", return_sequences=True))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation("relu"))
    model.add(layers.MaxPooling1D(2))
    model.add(layers.Dropout(0.2))

    # Part 2: the same idea once more
    model.add(layers.Conv1D(64, 3, strides=2))
    model.add(layers.LSTM(34, activation="tanh", return_sequences=True))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation("relu"))
    model.add(layers.MaxPooling1D(2))
    model.add(layers.Dropout(0.3))

    # Part 3: three more LSTM layers
    model.add(layers.LSTM(64, activation="tanh", return_sequences=True))
    model.add(layers.Dropout(0.3))
    model.add(layers.LSTM(64, activation="relu", return_sequences=True))
    model.add(layers.Dropout(0.3))
    model.add(layers.LSTM(64))
    model.add(layers.Dropout(0.2))

    # Part 4: classifier
    model.add(layers.Flatten())
    model.add(layers.Dense(32, name="features"))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(2, activation="softmax"))
    return model


BUILDERS = {"LSTM": build_lstm, "CNN": build_cnn, "CNN-LSTM": build_cnn_lstm}


def build(name: str) -> keras.Model:
    """Build a model by its name: 'LSTM', 'CNN' or 'CNN-LSTM'."""
    return BUILDERS[name]()


def count_params(model: keras.Model) -> dict:
    """Trainable / non-trainable parameter counts (the report quotes the trainable number)."""
    trainable = int(sum(int(np.prod(w.shape)) for w in model.trainable_weights))
    total = int(model.count_params())
    return {"trainable": trainable, "total": total}
