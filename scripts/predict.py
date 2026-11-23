#!/usr/bin/env python
"""
predict.py  -  use a trained model on EEG data.

Try it on 10 random samples of the dataset (shows the true answer next to the prediction):
    python scripts/predict.py --demo

Use it on YOUR OWN file (a CSV where every row = 178 EEG numbers, ID/label columns are ignored):
    python scripts/predict.py --csv my_eeg.csv

Choose another model:
    python scripts/predict.py --demo --model results/random_split/models/cnn.keras

NOTE: research prototype - it is NOT a medical device.
"""
import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np                    # noqa: E402
import pandas as pd                   # noqa: E402

from src import config, data          # noqa: E402

DEFAULT_DIR = config.RESULTS_DIR / "random_split" / "models"


def read_segments(path: Path) -> np.ndarray:
    """Read a CSV with 178 EEG numbers per row. Columns called X1..X178 are used if they exist."""
    df = pd.read_csv(path)
    if all(c in df.columns for c in data.FEATURE_COLUMNS):
        return df[data.FEATURE_COLUMNS].to_numpy(np.float32)
    try:                                              # file without a header line
        df = pd.read_csv(path, header=None)
    except Exception:                                 # noqa: BLE001
        pass
    num = df.apply(pd.to_numeric, errors="coerce").dropna(axis=1, how="all")
    if num.shape[1] == 178:
        return num.to_numpy(np.float32)
    if num.shape[1] == 179:                           # 178 values + label
        return num.iloc[:, :178].to_numpy(np.float32)
    raise SystemExit(f"Each row needs 178 numbers, but your file has {num.shape[1]} numeric columns.")


def main():
    ap = argparse.ArgumentParser(description="Seizure prediction with a trained model.")
    ap.add_argument("--csv", help="CSV file with EEG segments")
    ap.add_argument("--demo", action="store_true", help="try 10 random samples from data/data.csv")
    ap.add_argument("--model", default=str(DEFAULT_DIR / "cnn_lstm.keras"))
    ap.add_argument("--scaler", default=None, help="scaler.json (default: next to the model)")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--out", help="save the predictions to this CSV file")
    args = ap.parse_args()
    if not args.csv and not args.demo:
        raise SystemExit("Use  --demo  or  --csv FILE")

    from src import train as T                        # needs TensorFlow

    model_path = Path(args.model)
    scaler_path = Path(args.scaler) if args.scaler else model_path.parent / "scaler.json"
    name = {"lstm": "LSTM", "cnn": "CNN", "cnn_lstm": "CNN-LSTM"}.get(model_path.stem, "CNN-LSTM")
    model = T.load_model(model_path)
    mean, std = data.load_scaler_arrays(scaler_path)

    truth = None
    if args.demo:
        df = data.load_dataframe().sample(10, random_state=int.from_bytes(os.urandom(2), "little"))
        X, truth = df[data.FEATURE_COLUMNS].to_numpy(np.float32), df["y"].to_numpy()
    else:
        X = read_segments(Path(args.csv))

    prob = T.predict_prob(name, model, (X - mean) / std)
    result = pd.DataFrame({"seizure_probability": prob.round(4),
                           "decision": np.where(prob >= args.threshold, "SEIZURE", "normal")})
    if truth is not None:
        result.insert(0, "true_class", truth)
    print(f"\nModel: {model_path.name}   (research prototype - not a medical device)\n")
    print(result.to_string())
    if truth is not None:
        print("\ntrue_class: 1 = seizure, 2-5 = no seizure (2/3 = patient without seizure, 4/5 = healthy person)")
    if args.out:
        result.to_csv(args.out, index=False)
        print("\nsaved:", args.out)


if __name__ == "__main__":
    main()
