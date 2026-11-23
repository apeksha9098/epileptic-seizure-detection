"""
evaluate.py  -  measure how good a model is, and draw the pictures.

IMPORTANT - how we calculate Precision / Recall / F1
----------------------------------------------------
We always calculate them for the SEIZURE class (the thing we want to find):

    Precision    = TP / (TP + FP)   "when the model says SEIZURE, how often is it right?"
    Recall       = TP / (TP + FN)   "how many of the real seizures did the model find?"
                                    (in medicine this is called SENSITIVITY)
    Specificity  = TN / (TN + FP)   "how many normal signals were correctly called normal?"
    F1           = 2 * Precision * Recall / (Precision + Recall)
    Accuracy     = (TP + TN) / all samples

(The report used scikit-learn's average='micro'. With micro-averaging, precision = recall =
 F1 = accuracy, which hides how well seizures are really detected. That is why the numbers
 in this project can look different from the draft report.)
"""
import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve

# Class names used in pictures
CLASS_NAMES = ["Non-seizure", "Seizure"]
ORIGINAL_CLASS_NAMES = {
    1: "1 - Seizure (ictal)",
    2: "2 - Seizure-free, epileptogenic zone",
    3: "3 - Seizure-free, hippocampal area",
    4: "4 - Healthy, eyes closed",
    5: "5 - Healthy, eyes open",
}
COLORS = {"LSTM": "#1f77b4", "CNN": "#2ca02c", "CNN-LSTM": "#d62728",
          "Random Forest": "#9467bd", "k-NN": "#8c564b", "Logistic Regression": "#7f7f7f",
          "CNN-LSTM + RF": "#ff7f0e", "Ensemble (all models)": "#17becf"}


def slug(name: str) -> str:
    """'CNN-LSTM + RF' -> 'cnn_lstm_rf'  (safe text for file names)."""
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


# ----------------------------------------------------------------------------------------
# Numbers
# ----------------------------------------------------------------------------------------
def binary_metrics(y_true, prob, threshold: float = 0.5) -> dict:
    """All metrics for one model. `prob` = probability of SEIZURE for each sample."""
    y_true = np.asarray(y_true).astype(int)
    prob = np.asarray(prob, dtype=float)
    y_pred = (prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    div = lambda a, b: float(a / b) if b else 0.0  # noqa: E731
    precision = div(tp, tp + fp)
    recall = div(tp, tp + fn)
    try:
        auc = float(roc_auc_score(y_true, prob))
    except ValueError:
        auc = float("nan")
    return {
        "accuracy": div(tp + tn, tp + tn + fp + fn),
        "precision": precision,
        "recall_sensitivity": recall,
        "specificity": div(tn, tn + fp),
        "f1": div(2 * precision * recall, precision + recall),
        "roc_auc": auc,
        "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
    }


def bootstrap_ci(y_true, prob, groups, n_boot: int = 2000, seed: int = 0, threshold: float = 0.5) -> dict:
    """
    95 % confidence intervals for accuracy, precision, recall and F1.

    "Cluster bootstrap": we draw whole RECORDINGS (with replacement) many times and recompute the
    metrics each time. (The 23 pieces of one recording are not independent, so we must not draw single rows.)
    The interval tells us how much the number could change with a different set of recordings.
    """
    y_true = np.asarray(y_true).astype(int)
    pred = (np.asarray(prob) >= threshold).astype(int)
    _, inv = np.unique(np.asarray(groups), return_inverse=True)
    n_g = inv.max() + 1
    cell = lambda mask: np.bincount(inv, weights=mask.astype(float), minlength=n_g)  # noqa: E731
    counts = np.stack([cell((y_true == 0) & (pred == 0)), cell((y_true == 0) & (pred == 1)),
                       cell((y_true == 1) & (pred == 0)), cell((y_true == 1) & (pred == 1))], axis=1)
    idx = np.random.default_rng(seed).integers(0, n_g, size=(n_boot, n_g))
    tn, fp, fn, tp = counts[idx].sum(axis=1).T
    acc = (tp + tn) / (tp + tn + fp + fn)
    prec = tp / np.maximum(tp + fp, 1)
    rec = tp / np.maximum(tp + fn, 1)
    f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-12)
    lo_hi = lambda v: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)))  # noqa: E731
    return {"accuracy": lo_hi(acc), "precision": lo_hi(prec), "recall": lo_hi(rec), "f1": lo_hi(f1)}


def add_confidence_intervals(table: pd.DataFrame, y_true, probs: dict, groups) -> pd.DataFrame:
    """Add the columns 'Accuracy 95% CI' and 'F1 95% CI' (as text) to a results table."""
    acc_ci, f1_ci = [], []
    for name in table["Model"]:
        ci = bootstrap_ci(y_true, probs[name], groups)
        acc_ci.append(f"{100 * ci['accuracy'][0]:.1f}-{100 * ci['accuracy'][1]:.1f}")
        f1_ci.append(f"{ci['f1'][0]:.3f}-{ci['f1'][1]:.3f}")
    table = table.copy()
    table["Accuracy 95% CI"] = acc_ci
    table["F1 95% CI"] = f1_ci
    return table


def error_breakdown(y5_test, prob, threshold: float = 0.5) -> pd.DataFrame:
    """Which ORIGINAL classes (1-5) does the model get wrong?"""
    y_pred = (np.asarray(prob) >= threshold).astype(int)
    y_true = (np.asarray(y5_test) == 1).astype(int)
    rows = []
    for c in sorted(ORIGINAL_CLASS_NAMES):
        mask = np.asarray(y5_test) == c
        wrong = int((y_pred[mask] != y_true[mask]).sum())
        rows.append({"original_class": ORIGINAL_CLASS_NAMES[c], "samples": int(mask.sum()),
                     "mistakes": wrong, "accuracy": 1 - wrong / max(mask.sum(), 1)})
    return pd.DataFrame(rows)


def results_table(results: dict) -> pd.DataFrame:
    """results = {model_name: metrics_dict}  ->  tidy table (percentages), best first."""
    rows = []
    for name, m in results.items():
        rows.append({"Model": name,
                     "Accuracy %": 100 * m["accuracy"], "Precision %": 100 * m["precision"],
                     "Recall % (sensitivity)": 100 * m["recall_sensitivity"],
                     "Specificity %": 100 * m["specificity"], "F1": m["f1"],
                     "ROC-AUC": m["roc_auc"], "TN": m["TN"], "FP": m["FP"], "FN": m["FN"], "TP": m["TP"]})
    return pd.DataFrame(rows).sort_values("F1", ascending=False).reset_index(drop=True)


def dataframe_to_markdown(df: pd.DataFrame, floatfmt: str = "{:.2f}") -> str:
    """Small markdown-table writer (no extra libraries needed)."""
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            if isinstance(v, (float, np.floating)):
                v = floatfmt.format(v) if c != "F1" and c != "ROC-AUC" else f"{v:.4f}"
            cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


# ----------------------------------------------------------------------------------------
# Pictures
# ----------------------------------------------------------------------------------------
def _finish(fig, path, show):
    """Save the picture (if a path is given). Keep it open for display only if wanted.

    show=None  -> show it when no file path is given (notebook use), otherwise just save + close.
    show=True  -> save AND keep it open so that a notebook displays it.
    """
    if path is not None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=130, bbox_inches="tight")
    if show is None:
        show = path is None
    if not show:
        plt.close(fig)


def plot_history(history: dict, title: str, path=None, show=None):
    """Training curves: loss and accuracy for train and validation data."""
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ep = np.arange(1, len(history["loss"]) + 1)
    ax[0].plot(ep, history["loss"], label="train")
    ax[0].plot(ep, history["val_loss"], label="validation")
    ax[0].set(title=f"{title} - loss", xlabel="epoch", ylabel="loss")
    ax[1].plot(ep, history["accuracy"], label="train")
    ax[1].plot(ep, history["val_accuracy"], label="validation")
    ax[1].set(title=f"{title} - accuracy", xlabel="epoch", ylabel="accuracy")
    for a in ax:
        a.grid(alpha=0.3)
        a.legend()
    fig.tight_layout()
    _finish(fig, path, show)


def plot_history_grid(histories: dict, title: str, path=None, show=None):
    """Learning curves of several models in one picture (one row per model)."""
    n = len(histories)
    fig, axes = plt.subplots(n, 2, figsize=(11, 3.2 * n), squeeze=False)
    for row, (name, h) in zip(axes, histories.items()):
        ep = np.arange(1, len(h["loss"]) + 1)
        best = int(np.argmin(h["val_loss"])) + 1
        row[0].plot(ep, h["loss"], label="train")
        row[0].plot(ep, h["val_loss"], label="validation")
        row[0].set(title=f"{name} - loss", xlabel="epoch", ylabel="loss")
        row[1].plot(ep, h["accuracy"], label="train")
        row[1].plot(ep, h["val_accuracy"], label="validation")
        row[1].set(title=f"{name} - accuracy", xlabel="epoch", ylabel="accuracy")
        for a in row:
            a.axvline(best, color="gray", ls=":", label=f"best epoch = {best}")
            a.grid(alpha=0.3)
            a.legend(fontsize=8)
    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    _finish(fig, path, show)


def _draw_confusion(ax, m: dict, title: str):
    """Draw one confusion matrix (counts + % of each true class) on a given axis."""
    cm = np.array([[m["TN"], m["FP"]], [m["FN"], m["TP"]]])
    pct = cm / cm.sum(axis=1, keepdims=True) * 100
    ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]}\n({pct[i, j]:.1f}%)", ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=11)
    ax.set_xticks([0, 1], ["Non-seizure", "Seizure"], fontsize=8)
    ax.set_yticks([0, 1], ["Non-seizure", "Seizure"], fontsize=8)
    ax.set_xlabel("Predicted", fontsize=9)
    ax.set_ylabel("True", fontsize=9)
    ax.set_title(title, fontsize=10)


def plot_confusion(m: dict, title: str, path=None, show=None):
    """Confusion matrix of ONE model."""
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    _draw_confusion(ax, m, title)
    fig.tight_layout()
    _finish(fig, path, show)


def plot_confusion_grid(metrics: dict, title: str, path=None, show=None, ncols: int = 4):
    """Confusion matrices of ALL models in one picture."""
    names = list(metrics)
    ncols = min(ncols, len(names))
    nrows = math.ceil(len(names) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.7 * ncols, 3.6 * nrows), squeeze=False)
    for ax, name in zip(axes.ravel(), names):
        _draw_confusion(ax, metrics[name], name)
    for ax in axes.ravel()[len(names):]:
        ax.axis("off")
    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    _finish(fig, path, show)


def plot_roc(y_true, probs: dict, title: str, path=None, show=None):
    """ROC curves of several models: full view (left) and a zoom on the top-left corner (right)."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, zoom in zip(axes, (False, True)):
        for name, p in probs.items():
            fpr, tpr, _ = roc_curve(y_true, p)
            ax.plot(fpr, tpr, color=COLORS.get(name), lw=1.6,
                    label=f"{name} (AUC={roc_auc_score(y_true, p):.4f})")
        if not zoom:
            ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
            ax.set(xlim=(0, 1), ylim=(0, 1.02), title=title)
            ax.legend(fontsize=8, loc="lower right")
        else:
            ax.set(xlim=(0, 0.15), ylim=(0.8, 1.005), title="Zoom: top-left corner")
        ax.set_xlabel("False positive rate (1 - specificity)")
        ax.set_ylabel("True positive rate (sensitivity)")
        ax.grid(alpha=0.3)
    fig.tight_layout()
    _finish(fig, path, show)


def plot_comparison(table: pd.DataFrame, title: str, path=None, show=None):
    """Bar chart: Accuracy, Precision, Recall and F1 of all models."""
    metrics = [("Accuracy %", "Accuracy"), ("Precision %", "Precision"),
               ("Recall % (sensitivity)", "Recall"), ("F1", "F1 x100")]
    names = table["Model"].tolist()
    x = np.arange(len(names))
    w = 0.2
    fig, ax = plt.subplots(figsize=(max(8, 1.3 * len(names)), 4.8))
    for k, (col, label) in enumerate(metrics):
        vals = table[col] * (100 if col == "F1" else 1)
        ax.bar(x + (k - 1.5) * w, vals, w, label=label)
    lo = max(0, float(min(table["Accuracy %"].min(), table["Recall % (sensitivity)"].min(),
                          table["Precision %"].min())) - 5)
    ax.set_ylim(lo, 100.5)
    ax.set_xticks(x, names, rotation=25, ha="right")
    ax.set_ylabel("%")
    ax.set_title(title, pad=26)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(ncol=4, loc="lower center", bbox_to_anchor=(0.5, 1.0), fontsize=8, frameon=False)
    fig.tight_layout()
    _finish(fig, path, show)


def plot_class_distribution(df: pd.DataFrame, path=None, show=None):
    """How many samples does every class have?"""
    counts = df["y"].value_counts().sort_index()
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].bar([ORIGINAL_CLASS_NAMES[c].replace(", ", ",\n") for c in counts.index], counts.values,
              color=["#d62728", "#1f77b4", "#1f77b4", "#1f77b4", "#1f77b4"])
    ax[0].set(title="Original 5 classes (2300 samples each)", ylabel="samples")
    ax[0].tick_params(axis="x", labelsize=7)
    b = [counts[1], counts[[2, 3, 4, 5]].sum()]
    ax[1].bar(CLASS_NAMES[::-1], b[::-1], color=["#1f77b4", "#d62728"])
    for i, v in enumerate(b[::-1]):
        ax[1].text(i, v + 100, str(v), ha="center")
    ax[1].set(title="Binary problem used in the project", ylabel="samples")
    fig.tight_layout()
    _finish(fig, path, show)


def plot_sample_eeg(df: pd.DataFrame, feature_columns, path=None, seed: int = 1, show=None):
    """One example EEG signal (1 second) for each of the 5 classes."""
    rng = np.random.default_rng(seed)
    fig, axes = plt.subplots(5, 1, figsize=(9, 9), sharex=True)
    for ax, c in zip(axes, [1, 2, 3, 4, 5]):
        rows = df[df["y"] == c]
        row = rows.iloc[int(rng.integers(len(rows)))]
        ax.plot(row[feature_columns].to_numpy(dtype=float), color="#d62728" if c == 1 else "#1f77b4", lw=1)
        ax.set_ylabel("uV", fontsize=8)
        ax.set_title(ORIGINAL_CLASS_NAMES[c], fontsize=9, loc="left")
        ax.grid(alpha=0.3)
    axes[-1].set_xlabel("sample number (178 samples = 1 second)")
    fig.tight_layout()
    _finish(fig, path, show)


def save_json(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=2, default=float))
