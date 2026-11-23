# Extras – the team's earlier experiments (kept for reference)

## `original_BEED_experiments.ipynb`

This is the notebook from the project folder (originally called *“Bonn dataset.ipynb”*), kept **unchanged**.

**Important: it uses a different dataset.** It loads `BEED_Data.csv`, the
*Bangalore EEG Epilepsy Dataset (BEED)* – 8,000 rows, **16 channels** (`X1…X16`), **4 classes**
(0 healthy, 1 generalised seizure, 2 focal seizure, 3 seizure events) –
and **not** the Bonn / UCI “Epileptic Seizure Recognition” data (178 values per row, 5 classes) that the project
report and slides are based on. The main project in this repository follows the **report**, i.e. the UCI/Bonn data.

What the notebook contains (from its saved outputs):

* 19 classical ML models on BEED (binary and 4-class) – e.g. Extra Trees ≈ 98.4 % test accuracy (4 classes).
* LSTM, BiLSTM, CNN, CNN-BiLSTM and an “optimised CNN + batch-norm” – the best reached about **98.9 %** (4 classes).

Things to check before using these numbers in a report:

1. **Possible leakage.** In the printed rows at the end of the file (class 3), neighbouring rows are almost identical,
   which suggests consecutive time samples. With a random split, near-duplicates can land in train *and* test and
   inflate the scores – the same issue that `docs/3_HOW_IT_WORKS.md` (section 3.6) explains for the main data set.
2. **The “Train-test overlap” cell is not a real duplicate check.** It computes
   `len(set(train)) & len(set(test))`, which is a bit-wise AND of two *numbers* (that is where “1088” comes from).
   A correct check is `len(set(map(tuple, X_train)) & set(map(tuple, X_test)))`.
3. Three cells stop with an error in the saved outputs: cell 13 (`SyntaxError`), cell 22 (it looks for a ZIP file that is not present)
   and the last cell 60 (a `LearningRateSchedule` cannot be changed by `ReduceLROnPlateau` – use a fixed learning rate with that callback).

If you want BEED as a second full experiment in this repository, the code in `src/` can be adapted
(input length 16 instead of 178, 4 outputs instead of 2).
