# Dataset

**File:** `data.csv` (7.6 MB) – **UCI “Epileptic Seizure Recognition”** data set.

| Property | Value |
|---|---|
| Rows | 11,500 (500 recordings × 23 pieces of 1 second) |
| Columns | `Unnamed: 0` (ID, e.g. `X21.V1.791`), `X1 … X178` (EEG values), `y` (class 1–5) |
| Classes | 1 = seizure, 2–5 = no seizure (2,300 rows each) |
| Sampling | 173.61 Hz → 178 values ≈ 1 second |
| Missing values | none |

The **ID column** is useful: the part after `X21.` (here `V1.791`) is the same for all 23 pieces of one recording.
`src/data.py` uses this to keep recordings together in the leak-free “recording split”.

## Where does it come from? (please give credit)

* Original recordings: R. G. Andrzejak, K. Lehnertz, F. Mormann, C. Rieke, P. David, C. E. Elger,
  *“Indications of nonlinear deterministic and finite-dimensional structures in time series of brain electrical activity:
  Dependence on recording region and brain state”*, **Physical Review E 64, 061907 (2001)** – Department of Epileptology, University of Bonn.
* Re-structured 178-column version: Qiuyi Wu and Ernest Fokoué, *Epileptic Seizure Recognition*,
  **UCI Machine Learning Repository** (2017). https://archive.ics.uci.edu/dataset/388/epileptic+seizure+recognition
  (the older link from the project slides also works: https://archive.ics.uci.edu/ml/datasets/Epileptic+Seizure+Recognition;
  the data is also on Kaggle – search “Epileptic Seizure Recognition”).

The data is published for research and education. Please check the terms on the UCI page.
If you prefer not to redistribute the file, delete `data/data.csv` before uploading to GitHub – the program
then tries to download it, or tells you exactly how to get it (see `src/data.py`).

## Integrity check (optional)

```
SHA-256  8e32c6751af5e67373051c3ae3356020f854e1835232005071362c6d5fbabf9f
MD5      8470df1188f79a7d9b970a1b42e348a4
```
(`sha256sum data/data.csv` on Linux/Mac, `certutil -hashfile data\data.csv SHA256` on Windows.)

## A note on the class descriptions

Some copies of the dataset describe classes 2 and 3 as “tumor area / healthy area”. In the original Bonn description,
these are **seizure-free (interictal) recordings of epilepsy patients**: class 2 from the epileptogenic zone, class 3 from the
hippocampal formation of the opposite hemisphere. Class 4 and 5 are healthy volunteers with eyes closed / open.
