#!/usr/bin/env python
"""
export_for_raspberry_pi.py  -  run this on your PC / Google Colab (where TensorFlow is installed).

It converts a trained Keras model (.keras) into two small files that the Raspberry Pi can use
WITHOUT installing TensorFlow (installing TensorFlow on a Raspberry Pi is slow and often fails):

    <name>.npz   the learned numbers (weights)
    <name>.json  a short description of the layers

Example:
    python raspberry_pi/export_for_raspberry_pi.py \
        --model results/random_split/models/cnn_lstm.keras \
        --out   raspberry_pi/model/cnn_lstm
"""
import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import numpy as np                      # noqa: E402
from tensorflow import keras            # noqa: E402


def _act_name(layer) -> str:
    act = getattr(layer, "activation", None)
    return getattr(act, "__name__", "linear") if act is not None else "linear"


def export(model_path: str, out_prefix: str) -> None:
    model = keras.models.load_model(model_path, compile=False)
    arch, arrays = [], {}
    for idx, layer in enumerate(model.layers):
        kind = layer.__class__.__name__
        entry = {"type": kind, "id": idx}
        w = layer.get_weights()
        if kind == "Conv1D":
            entry.update(stride=int(layer.strides[0]), padding=layer.padding, activation=_act_name(layer))
            arrays[f"{idx}_W"], arrays[f"{idx}_b"] = w
            assert layer.padding == "valid", "only 'valid' padding is supported"
        elif kind == "LSTM":
            entry.update(units=int(layer.units), activation=_act_name(layer),
                         recurrent_activation=getattr(layer.recurrent_activation, "__name__", "sigmoid"),
                         return_sequences=bool(layer.return_sequences))
            arrays[f"{idx}_kernel"], arrays[f"{idx}_recurrent"], arrays[f"{idx}_bias"] = w
        elif kind == "BatchNormalization":
            entry.update(epsilon=float(layer.epsilon))
            (arrays[f"{idx}_gamma"], arrays[f"{idx}_beta"],
             arrays[f"{idx}_mean"], arrays[f"{idx}_var"]) = w
        elif kind == "Dense":
            entry.update(activation=_act_name(layer))
            arrays[f"{idx}_W"], arrays[f"{idx}_b"] = w
        elif kind == "Activation":
            entry.update(activation=_act_name(layer))
        elif kind == "MaxPooling1D":
            entry.update(pool=int(layer.pool_size[0]), stride=int(layer.strides[0]))
        elif kind in ("Flatten", "Dropout"):
            pass                                    # Dropout does nothing at prediction time
        else:
            raise ValueError(f"Layer type {kind} is not supported by the NumPy engine.")
        if kind != "Dropout":
            arch.append(entry)

    out_prefix = Path(out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_prefix.with_suffix(".npz"), **{k: v.astype(np.float32) for k, v in arrays.items()})
    meta = {"input_shape": [int(d) for d in model.input_shape[1:]], "layers": arch}
    out_prefix.with_suffix(".json").write_text(json.dumps(meta, indent=1))
    print(f"saved {out_prefix.with_suffix('.npz')}  and  {out_prefix.with_suffix('.json')}")
    print("input shape:", meta["input_shape"], "| layers:", len(arch))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="path of the trained .keras model")
    ap.add_argument("--out", required=True, help="output name without extension")
    a = ap.parse_args()
    export(a.model, a.out)
