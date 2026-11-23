"""
train.py  -  train a deep-learning model and use it for predictions.

How training works (simple words)
---------------------------------
* The model sees the TRAINING samples again and again. One full pass = one "epoch".
* After every epoch it is checked on the VALIDATION samples (data it does not learn from).
* "Early stopping": if the validation loss has not improved for several epochs we stop and
  go back to the best epoch  ->  this prevents over-fitting (learning the training data by heart).
* "Reduce learning rate": if progress stalls, the optimiser (Adam) takes smaller steps.
"""
import time
from pathlib import Path

import numpy as np
from tensorflow import keras

from . import config, models
from .data import as_cnn_input, as_lstm_input
from .evaluate import slug  # noqa: F401  (re-exported for convenience)


def reshape_for(name: str, X: np.ndarray) -> np.ndarray:
    """Give the data the shape the model expects."""
    return as_lstm_input(X) if models.INPUT_KIND[name] == "lstm" else as_cnn_input(X)


def train_dl_model(name: str, splits, epochs: int = config.EPOCHS,
                   batch_size: int = config.BATCH_SIZE, lr: float = config.LEARNING_RATE,
                   patience: int = config.EARLY_STOP_PATIENCE, seed: int = config.SEED,
                   verbose: int = 2):
    """
    Build + train one model.  Returns (trained_model, info_dict).

    info_dict contains the learning curves ("history"), the training time, the number of
    epochs that were really run, the best epoch and the parameter counts.
    """
    keras.utils.set_random_seed(seed)                       # repeatable results
    model = models.build(name)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr),
                  loss="categorical_crossentropy", metrics=["accuracy"])

    x_train, x_val = reshape_for(name, splits.X_train), reshape_for(name, splits.X_val)
    y_train = keras.utils.to_categorical(splits.y_train, 2)
    y_val = keras.utils.to_categorical(splits.y_val, 2)

    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=patience,
                                      restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                          patience=config.LR_PATIENCE, min_lr=1e-5),
    ]
    t0 = time.time()
    hist = model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=epochs,
                     batch_size=batch_size, callbacks=callbacks, verbose=verbose)
    seconds = time.time() - t0

    history = {k: [float(v) for v in vals] for k, vals in hist.history.items()}
    info = {"history": history, "train_seconds": seconds,
            "epochs_run": len(history["loss"]),
            "best_epoch": int(np.argmin(history["val_loss"])) + 1,
            **{f"params_{k}": v for k, v in models.count_params(model).items()}}
    return model, info


def predict_prob(name: str, model, X: np.ndarray) -> np.ndarray:
    """Probability of SEIZURE (a number between 0 and 1) for every row of X."""
    return model.predict(reshape_for(name, X), batch_size=256, verbose=0)[:, 1]


def save_model(model, path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    model.save(path)


def load_model(path):
    return keras.models.load_model(path, compile=False)
