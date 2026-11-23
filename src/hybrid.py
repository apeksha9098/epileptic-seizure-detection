"""
hybrid.py  -  COMBINE machine learning and deep learning
(one of the goals written in the abstract of the project report).

Two simple ways to combine them:

1. CNN-LSTM + Random Forest
   The trained CNN-LSTM is used as a "feature extractor": the 32 numbers of its dense layer
   ("features") describe every EEG sample. A Random Forest then makes the final decision
   using these 32 learned features instead of the 178 raw values.

2. Ensemble (soft voting)
   Several models vote. We average their seizure-probabilities - a model that is very sure
   counts more than one that is unsure.
"""
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from tensorflow import keras

from . import config
from .data import as_cnn_input


def feature_extractor(model: keras.Model) -> keras.Model:
    """A copy of the network that stops at the layer called 'features'."""
    return keras.Model(inputs=model.inputs, outputs=model.get_layer("features").output)


def cnn_lstm_plus_rf(cnn_lstm: keras.Model, splits, quick: bool = False) -> dict:
    """Train a Random Forest on the features learned by the CNN-LSTM."""
    fe = feature_extractor(cnn_lstm)
    feats = lambda X: fe.predict(as_cnn_input(X), batch_size=256, verbose=0)  # noqa: E731
    f_train, f_val, f_test = feats(splits.X_train), feats(splits.X_val), feats(splits.X_test)
    rf = RandomForestClassifier(n_estimators=50 if quick else 300, max_features="sqrt",
                                min_samples_leaf=1, random_state=config.SEED, n_jobs=-1)
    rf.fit(f_train, splits.y_train)
    return {"prob_val": rf.predict_proba(f_val)[:, 1], "prob_test": rf.predict_proba(f_test)[:, 1]}


def soft_vote(prob_list) -> np.ndarray:
    """Average the seizure-probabilities of several models."""
    return np.mean(np.vstack(prob_list), axis=0)
