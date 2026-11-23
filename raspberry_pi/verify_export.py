#!/usr/bin/env python
"""
verify_export.py  -  check that the exported (NumPy) model gives the SAME answers as the
original Keras model.  Run it on your PC / Colab after export_for_raspberry_pi.py.

    python raspberry_pi/verify_export.py --model results/random_split/models/cnn_lstm.keras \
                                         --prefix raspberry_pi/model/cnn_lstm
"""
import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

import numpy as np                       # noqa: E402
from tensorflow import keras             # noqa: E402

from rpi_predict import NumpyModel       # noqa: E402
from src import data                     # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="trained .keras model")
    ap.add_argument("--prefix", required=True, help="exported model prefix (no extension)")
    ap.add_argument("--n", type=int, default=300, help="how many EEG segments to compare")
    a = ap.parse_args()

    df = data.load_dataframe().sample(a.n, random_state=0)
    X = df[data.FEATURE_COLUMNS].to_numpy(np.float32)
    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-6)          # any scaling is fine for this comparison

    keras_model = keras.models.load_model(a.model, compile=False)
    shape = (len(X),) + tuple(keras_model.input_shape[1:])
    p_keras = keras_model.predict(X.reshape(shape), verbose=0)
    p_numpy = NumpyModel(Path(a.prefix)).predict_proba(X)

    diff = float(np.abs(p_keras - p_numpy).max())
    same = float((p_keras.argmax(1) == p_numpy.argmax(1)).mean())
    print(f"largest difference in probability : {diff:.2e}")
    print(f"same decision on {100 * same:.1f}% of the {len(X)} segments")
    print("RESULT:", "OK - the Raspberry Pi version behaves like the original model." if diff < 1e-3
          else "PROBLEM - outputs differ, do not use this export.")


if __name__ == "__main__":
    main()
