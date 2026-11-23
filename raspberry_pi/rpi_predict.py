#!/usr/bin/env python3
"""
rpi_predict.py  -  seizure detection on a Raspberry Pi (needs only NumPy; no TensorFlow).

It reads EEG segments (178 numbers = 1 second each), runs the trained network and prints
"normal" or "SEIZURE" for every second. Optionally it switches an LED / buzzer on.

Usage (on the Raspberry Pi):
    python3 rpi_predict.py --demo                          # test with the example file
    python3 rpi_predict.py --csv my_eeg.csv                # your own file, one segment per row
    python3 rpi_predict.py --csv my_eeg.csv --led-pin 17   # also switch an LED on GPIO17 on a seizure

Files needed in the same folder (or in ./model):
    cnn_lstm.npz  cnn_lstm.json  scaler.json     (created by export_for_raspberry_pi.py)

NOTE: research prototype. It is NOT a medical device.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


# ======================================================================================
# A tiny neural-network engine written with plain NumPy
# (it re-implements the layers of our models: Conv1D, LSTM, BatchNorm, MaxPool, Dense ...)
# ======================================================================================
def _activation(name: str, x: np.ndarray) -> np.ndarray:
    if name == "relu":
        return np.maximum(x, 0.0)
    if name == "tanh":
        return np.tanh(x)
    if name == "sigmoid":
        return 1.0 / (1.0 + np.exp(-x))
    if name == "softmax":
        e = np.exp(x - x.max(axis=-1, keepdims=True))
        return e / e.sum(axis=-1, keepdims=True)
    return x                                                   # "linear"


def _conv1d(x, W, b, stride):
    """x: (batch, time, channels)   W: (kernel, channels, filters)   'valid' padding."""
    k = W.shape[0]
    out_len = (x.shape[1] - k) // stride + 1
    out = np.zeros((x.shape[0], out_len, W.shape[2]), dtype=np.float32)
    for i in range(k):
        out += np.tensordot(x[:, i:i + stride * (out_len - 1) + 1:stride, :], W[i], axes=([2], [0]))
    return out + b


def _lstm(x, kernel, recurrent, bias, act, rec_act, return_sequences):
    """x: (batch, time, features). Gate order (as in Keras): input, forget, cell, output."""
    batch, steps, _ = x.shape
    units = recurrent.shape[0]
    h = np.zeros((batch, units), dtype=np.float32)
    c = np.zeros((batch, units), dtype=np.float32)
    xw = np.tensordot(x, kernel, axes=([2], [0])) + bias       # all time steps at once
    outputs = []
    for t in range(steps):
        z = xw[:, t, :] + h @ recurrent
        i = _activation(rec_act, z[:, :units])
        f = _activation(rec_act, z[:, units:2 * units])
        g = _activation(act, z[:, 2 * units:3 * units])
        o = _activation(rec_act, z[:, 3 * units:])
        c = f * c + i * g
        h = o * _activation(act, c)
        outputs.append(h)
    return np.stack(outputs, axis=1) if return_sequences else h


def _maxpool(x, pool, stride):
    out_len = (x.shape[1] - pool) // stride + 1
    cols = [x[:, i * stride:i * stride + pool, :].max(axis=1) for i in range(out_len)]
    return np.stack(cols, axis=1)


class NumpyModel:
    """Loads <prefix>.json + <prefix>.npz and predicts like the original Keras model."""

    def __init__(self, prefix: Path):
        self.meta = json.loads(Path(prefix).with_suffix(".json").read_text())
        self.w = dict(np.load(Path(prefix).with_suffix(".npz")))
        self.input_shape = tuple(self.meta["input_shape"])     # e.g. (178, 1) or (1, 178)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        """x: (batch, 178) already normalised  ->  probabilities (batch, 2)."""
        x = x.astype(np.float32).reshape((x.shape[0],) + self.input_shape)
        for L in self.meta["layers"]:
            i, kind = L["id"], L["type"]
            if kind == "Conv1D":
                x = _activation(L["activation"], _conv1d(x, self.w[f"{i}_W"], self.w[f"{i}_b"], L["stride"]))
            elif kind == "LSTM":
                x = _lstm(x, self.w[f"{i}_kernel"], self.w[f"{i}_recurrent"], self.w[f"{i}_bias"],
                          L["activation"], L["recurrent_activation"], L["return_sequences"])
            elif kind == "BatchNormalization":
                x = (x - self.w[f"{i}_mean"]) / np.sqrt(self.w[f"{i}_var"] + L["epsilon"])
                x = x * self.w[f"{i}_gamma"] + self.w[f"{i}_beta"]
            elif kind == "Activation":
                x = _activation(L["activation"], x)
            elif kind == "MaxPooling1D":
                x = _maxpool(x, L["pool"], L["stride"])
            elif kind == "Flatten":
                x = x.reshape(x.shape[0], -1)
            elif kind == "Dense":
                x = _activation(L["activation"], x @ self.w[f"{i}_W"] + self.w[f"{i}_b"])
        return x


# ======================================================================================
# Reading data, LED, main loop
# ======================================================================================
def read_segments(path: Path):
    """Read a CSV: every row = one segment of 178 numbers.
    Returns (segments, labels). `labels` is None unless every row has a 179th number (the class 1-5)."""
    import csv
    rows = []
    with open(path, newline="") as fh:
        for row in csv.reader(fh):
            try:
                vals = [float(v) for v in row]
            except ValueError:
                continue                                       # header or ID text -> skip that row
            rows.append(vals)
    if not rows:
        raise SystemExit("No numeric rows found in the CSV file.")
    lengths = {len(r) for r in rows}
    if lengths == {178}:
        return np.array(rows, dtype=np.float32), None
    if lengths == {179}:                                       # 178 values + the class (1 = seizure) at the end
        arr = np.array(rows, dtype=np.float32)
        return arr[:, :178], arr[:, 178].astype(int)
    raise SystemExit(f"Each row must have 178 numbers, but found rows with {sorted(lengths)} numbers.")


def make_led(pin):
    """Return a function that switches an LED on/off (does nothing if not on a Raspberry Pi)."""
    if pin is None:
        return lambda on: None
    try:
        from gpiozero import LED
        led = LED(pin)
        return lambda on: led.on() if on else led.off()
    except Exception as err:                                   # noqa: BLE001
        print(f"(LED not available: {err}) - continuing without LED")
        return lambda on: None


def main():
    ap = argparse.ArgumentParser(description="EEG seizure detection on Raspberry Pi (NumPy only).")
    ap.add_argument("--csv", help="CSV file with EEG segments (178 numbers per row)")
    ap.add_argument("--demo", action="store_true", help="use the small example file next to this script")
    ap.add_argument("--model", default=str(HERE / "model" / "cnn_lstm"), help="model prefix (no extension)")
    ap.add_argument("--scaler", default=str(HERE / "model" / "scaler.json"))
    ap.add_argument("--threshold", type=float, default=0.5, help="probability above which we call it a seizure")
    ap.add_argument("--delay", type=float, default=0.0, help="seconds to wait between segments (1.0 = real time)")
    ap.add_argument("--led-pin", type=int, default=None, help="GPIO (BCM) pin of an LED/buzzer, e.g. 17")
    ap.add_argument("--max-rows", type=int, default=None)
    args = ap.parse_args()

    csv_path = Path(args.csv) if args.csv else (HERE / "example_segments.csv" if args.demo else None)
    if csv_path is None:
        raise SystemExit("Give --csv FILE or use --demo")

    model = NumpyModel(Path(args.model))
    sc = json.loads(Path(args.scaler).read_text())
    mean, std = np.array(sc["mean"], dtype=np.float32), np.array(sc["std"], dtype=np.float32)
    segments, labels = read_segments(csv_path)
    if args.max_rows:
        segments = segments[:args.max_rows]
    led = make_led(args.led_pin)

    print(f"Loaded {len(segments)} segments from {csv_path.name}.  (research prototype - not a medical device)\n")
    print(" #    seizure-probability   decision" + ("          true class" if labels is not None else ""))
    n_alert = 0
    t0 = time.time()
    for k, seg in enumerate(segments, 1):
        prob = float(model.predict_proba(((seg - mean) / std)[None, :])[0, 1])
        alert = prob >= args.threshold
        n_alert += alert
        led(alert)
        truth = "" if labels is None else f"      {'SEIZURE' if labels[k - 1] == 1 else 'normal'} (class {labels[k - 1]})"
        print(f"{k:3d}   {prob:8.4f}              {('*** SEIZURE ***' if alert else 'normal'):16s}{truth}")
        if args.delay:
            time.sleep(args.delay)
    ms = 1000 * (time.time() - t0) / len(segments) - 1000 * args.delay
    print(f"\n{n_alert} of {len(segments)} segments flagged as seizure.  Average compute time: {ms:.1f} ms per segment.")
    led(False)


if __name__ == "__main__":
    main()
