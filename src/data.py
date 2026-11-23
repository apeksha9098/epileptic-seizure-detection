"""
data.py  -  load the EEG dataset and prepare it for the models.

The dataset (UCI "Epileptic Seizure Recognition", made from the Bonn University EEG data):
    * 11,500 rows. Every row = 178 EEG values (1 second of brain signal) + a label.
    * Original label y:  1 = seizure,  2/3/4/5 = NO seizure.
    * The first column is an ID such as  "X21.V1.791".
      The part after "X21." (here "V1.791") tells us WHICH ORIGINAL RECORDING the row came
      from.  Every recording was cut into 23 pieces of 1 second  ->  500 recordings x 23 = 11,500.

We make the problem binary (as in the report):   seizure = 1   /   non-seizure = 0.

Two ways of splitting the data into train / validation / test (7 : 1 : 2):

    "random"     Rows are shuffled and split (this is what the report and most papers do).
                 Problem: pieces of the SAME recording end up in train AND test, and
                 neighbouring pieces are almost continuous -> the test is "too easy".

    "recording"  All 23 pieces of a recording stay together (all in train, or all in test).
                 This tells us how the model behaves on EEG recordings it has NEVER seen.
                 This is the honest (leak-free) check.
"""
import io
import zipfile
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from . import config

FEATURE_COLUMNS = [f"X{i}" for i in range(1, config.N_FEATURES + 1)]

# Best-effort automatic download (the dataset is already included in the repo's data/ folder).
UCI_ZIP_URL = "https://archive.ics.uci.edu/static/public/388/epileptic+seizure+recognition.zip"

MANUAL_DOWNLOAD_HELP = """
Could not find or download the dataset file  data/data.csv.

What to do (2 minutes):
  1. Open  https://archive.ics.uci.edu/dataset/388/epileptic+seizure+recognition
     (or search Kaggle for "Epileptic Seizure Recognition").
  2. Download the file and unzip it so that you get  data.csv  (size about 7.6 MB).
  3. Put it in the folder  data/  of this project  ->  data/data.csv
     (in Google Colab: use the folder icon on the left and drag the file into  data/ ).
"""


# ----------------------------------------------------------------------------------------
# 1. Loading
# ----------------------------------------------------------------------------------------
def ensure_dataset(path=None) -> Path:
    """Return the path of data.csv. If it is missing, try to download it."""
    path = Path(path or config.DATA_FILE)
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        print("data.csv not found -> trying to download it from the UCI repository ...")
        with urllib.request.urlopen(UCI_ZIP_URL, timeout=60) as response:
            blob = response.read()
        with zipfile.ZipFile(io.BytesIO(blob)) as zf:
            csv_name = [n for n in zf.namelist() if n.lower().endswith(".csv")][0]
            path.write_bytes(zf.read(csv_name))
        print("Download finished:", path)
        return path
    except Exception as err:  # noqa: BLE001  (we want to catch everything and explain)
        raise FileNotFoundError(MANUAL_DOWNLOAD_HELP) from err


def load_dataframe(path=None) -> pd.DataFrame:
    """Read data.csv and check that it looks like the correct dataset."""
    df = pd.read_csv(ensure_dataset(path))

    # First column is the ID (it is called "Unnamed: 0" in the original file)
    if df.columns[0] not in FEATURE_COLUMNS and df.columns[0] != "y":
        df = df.rename(columns={df.columns[0]: "id"})
    else:
        df.insert(0, "id", [f"row{i}" for i in range(len(df))])

    missing = [c for c in FEATURE_COLUMNS + ["y"] if c not in df.columns]
    if missing:
        raise ValueError(f"data.csv is missing the columns: {missing[:5]} ...")
    if df[FEATURE_COLUMNS + ["y"]].isnull().any().any():
        raise ValueError("data.csv contains missing values.")
    if set(df["y"].unique()) != {1, 2, 3, 4, 5}:
        raise ValueError("Column y must contain the classes 1,2,3,4,5.")
    return df


def recording_ids(ids: pd.Series) -> np.ndarray:
    """'X21.V1.791' -> 'V1.791'.  Rows with the same result come from the same recording."""
    return ids.astype(str).str.replace(r"^X\d+\.", "", regex=True).to_numpy()


def quick_subsample(df: pd.DataFrame, fraction: float = 0.2, seed: int = config.SEED):
    """Keep only a fraction of the RECORDINGS (for a fast test run)."""
    groups = recording_ids(df["id"])
    rec = pd.DataFrame({"g": groups, "y": df["y"].to_numpy()}).drop_duplicates("g")
    keep, _ = train_test_split(rec["g"], train_size=fraction, stratify=rec["y"], random_state=seed)
    return df[np.isin(groups, keep)].reset_index(drop=True)


# ----------------------------------------------------------------------------------------
# 2. Splitting
# ----------------------------------------------------------------------------------------
def _split_random(y5: np.ndarray, seed: int):
    """Shuffle rows, keep the class proportions equal in all three parts."""
    idx = np.arange(len(y5))
    idx_rest, idx_test = train_test_split(
        idx, test_size=config.TEST_FRAC, stratify=y5, random_state=seed)
    idx_train, idx_val = train_test_split(
        idx_rest, test_size=config.VAL_FRAC / (1 - config.TEST_FRAC),
        stratify=y5[idx_rest], random_state=seed)
    return idx_train, idx_val, idx_test


def _split_by_recording(groups: np.ndarray, y5: np.ndarray, seed: int):
    """Split RECORDINGS (not rows). All pieces of one recording stay together."""
    rec = pd.DataFrame({"g": groups, "y": y5}).drop_duplicates("g").reset_index(drop=True)
    if rec.groupby("g")["y"].nunique().max() != 1:
        raise ValueError("A recording has more than one class - cannot split by recording.")
    g_rest, g_test = train_test_split(
        rec["g"], test_size=config.TEST_FRAC, stratify=rec["y"], random_state=seed)
    y_rest = rec.set_index("g").loc[g_rest, "y"]
    g_train, g_val = train_test_split(
        g_rest, test_size=config.VAL_FRAC / (1 - config.TEST_FRAC),
        stratify=y_rest.to_numpy(), random_state=seed)
    to_idx = lambda g: np.where(np.isin(groups, g))[0]  # noqa: E731
    return to_idx(g_train), to_idx(g_val), to_idx(g_test)


# ----------------------------------------------------------------------------------------
# 3. Everything the models need, in one object
# ----------------------------------------------------------------------------------------
@dataclass
class Splits:
    """Holds the prepared data. X_* are scaled arrays of shape (n, 178)."""
    name: str
    X_train: np.ndarray
    X_val: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray          # 0 = non-seizure, 1 = seizure
    y_val: np.ndarray
    y_test: np.ndarray
    y5_test: np.ndarray          # original 5-class label of the test rows (for error analysis)
    scaler: StandardScaler
    info: dict = field(default_factory=dict)
    groups_test: np.ndarray = None    # recording ID of every test row (used for confidence intervals)


def prepare(df: pd.DataFrame, split: str = "random", seed: int = config.SEED) -> Splits:
    """
    Turn the DataFrame into train / validation / test arrays.

    split = "random"     -> shuffle rows            (as in the report)
    split = "recording"  -> keep recordings together (leak-free check)
    """
    X = df[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    y5 = df["y"].to_numpy()
    y = (y5 == 1).astype(np.int64)                       # seizure = 1, everything else = 0
    groups = recording_ids(df["id"])

    if split == "random":
        i_tr, i_va, i_te = _split_random(y5, seed)
    elif split == "recording":
        i_tr, i_va, i_te = _split_by_recording(groups, y5, seed)
    else:
        raise ValueError("split must be 'random' or 'recording'")

    # Normalisation: mean 0 and standard deviation 1 for each of the 178 columns.
    # The numbers (mean/std) are learned ONLY from the training data.
    scaler = StandardScaler().fit(X[i_tr])
    scale = lambda idx: scaler.transform(X[idx]).astype(np.float32)  # noqa: E731

    n_rec = lambda idx: int(len(np.unique(groups[idx])))  # noqa: E731
    overlap = len(set(groups[i_tr]) & set(groups[i_te]))
    info = {
        "split": split,
        "seed": seed,
        "rows": {"train": int(len(i_tr)), "val": int(len(i_va)), "test": int(len(i_te))},
        "seizure_rows": {"train": int(y[i_tr].sum()), "val": int(y[i_va].sum()), "test": int(y[i_te].sum())},
        "recordings": {"train": n_rec(i_tr), "val": n_rec(i_va), "test": n_rec(i_te)},
        "recordings_shared_between_train_and_test": int(overlap),
    }
    return Splits(split, scale(i_tr), scale(i_va), scale(i_te),
                  y[i_tr], y[i_va], y[i_te], y5[i_te], scaler, info, groups[i_te])


# ----------------------------------------------------------------------------------------
# 4. Small helpers
# ----------------------------------------------------------------------------------------
def as_lstm_input(X: np.ndarray) -> np.ndarray:
    """(n, 178) -> (n, 1, 178): one time-step that contains 178 values (as in the report)."""
    return X.reshape(X.shape[0], 1, X.shape[1])


def as_cnn_input(X: np.ndarray) -> np.ndarray:
    """(n, 178) -> (n, 178, 1): 178 time-steps with 1 value each (a 1-D signal)."""
    return X.reshape(X.shape[0], X.shape[1], 1)


def save_scaler(scaler: StandardScaler, path) -> None:
    """Save mean/std as plain JSON (works on every computer, also on Raspberry Pi)."""
    import json
    Path(path).write_text(json.dumps({"mean": scaler.mean_.tolist(), "std": scaler.scale_.tolist()}))


def load_scaler_arrays(path):
    """Read the JSON written by save_scaler. Returns (mean, std) numpy arrays."""
    import json
    d = json.loads(Path(path).read_text())
    return np.array(d["mean"], dtype=np.float32), np.array(d["std"], dtype=np.float32)
