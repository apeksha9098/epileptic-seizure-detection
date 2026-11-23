# 4. Notes for your report and viva

This page lists (1) the final numbers, (2) **corrections** to the draft report that you should make before submission, and
(3) questions an examiner may ask. Everything here was checked against your draft and against the code in this repository.

## 4.1 Final results (copy into Chapter 4)

Protocol A – random split 70 : 10 : 20, as in the draft (test = 2300 EEG seconds, 460 of them seizures):

| Model | Accuracy % | Precision % | Recall % | Specificity % | F1 | ROC-AUC |
|---|---|---|---|---|---|---|
| CNN-LSTM + RF | 98.96 | 96.38 | 98.48 | 99.08 | 0.974 | 0.997 |
| CNN-LSTM | 98.96 | 96.38 | 98.48 | 99.08 | 0.974 | 0.999 |
| Ensemble (all models) | 98.70 | 97.36 | 96.09 | 99.35 | 0.967 | 0.999 |
| Random Forest | 97.43 | 94.46 | 92.61 | 98.64 | 0.935 | 0.996 |
| LSTM | 97.35 | 95.44 | 91.09 | 98.91 | 0.932 | 0.995 |
| CNN | 96.26 | 97.95 | 83.04 | 99.57 | 0.899 | 0.996 |
| k-NN | 94.87 | 98.31 | 75.65 | 99.67 | 0.855 | 0.877 |
| Logistic Regression | 82.39 | 96.61 | 12.39 | 99.89 | 0.220 | 0.534 |

Protocol B – recording split (test = 100 unseen recordings):

| Model | Accuracy % | Precision % | Recall % | Specificity % | F1 | ROC-AUC |
|---|---|---|---|---|---|---|
| Ensemble (all models) | 98.52 | 96.71 | 95.87 | 99.18 | 0.963 | 0.999 |
| CNN-LSTM | 98.30 | 95.66 | 95.87 | 98.91 | 0.958 | 0.992 |
| CNN | 98.30 | 96.26 | 95.22 | 99.08 | 0.957 | 0.997 |
| CNN-LSTM + RF | 98.17 | 95.24 | 95.65 | 98.80 | 0.954 | 0.980 |
| LSTM | 98.09 | 92.28 | 98.70 | 97.93 | 0.954 | 0.995 |
| Random Forest | 97.91 | 94.21 | 95.43 | 98.53 | 0.948 | 0.998 |
| k-NN | 96.70 | 97.29 | 85.87 | 99.40 | 0.912 | 0.926 |
| Logistic Regression | 83.26 | 100.00 | 16.30 | 100.00 | 0.280 | 0.501 |

Suggested text for the conclusion:
> *The CNN-LSTM hybrid gave the best result with an accuracy of 98.96 %, a sensitivity of 98.5 %, a precision of 96.4 % and an F1-score of 0.974 on the test data, using only 127,578 trainable parameters. When the test recordings were completely excluded from training, the deep models still reached 98.1–98.3 % accuracy. Our result is comparable with the previous CNN-LSTM study (99.39 %), with about 17 times fewer parameters.*

## 4.2 Corrections for the draft report

### (1) Precision, recall and F1 in Tables 4–8 equal the accuracy – use seizure-class metrics
The code uses `average='micro'`. For a single-label problem, micro-averaged precision = recall = F1 = **accuracy**, so the tables show the same number four times
(e.g. CNN-LSTM: 99.13 % / 99.1 % / 99.1 % / 0.991). They do not tell how well *seizures* are found.
Recomputed **from the confusion matrices printed in your own draft** (seizure class):

| Model (draft) | Accuracy % | Precision % | Recall % | F1 |
|---|---|---|---|---|
| LSTM | 97.39 | 96.98 | 88.94 | 0.928 |
| CNN | 98.30 | 98.53 | 92.40 | 0.954 |
| CNN-LSTM | 99.13 | 98.14 | 97.24 | 0.977 |
| Random Forest | 97.57 | 93.15 | 94.01 | 0.936 |
| k-NN | 94.52 | 99.68 | 71.20 | 0.831 |

Consequences:
* The sentence that your model beats the previous study (precision 98.39 %, recall 98.79 %, F1 0.9859) **is not supported**: with the correct seizure-class calculation the draft's CNN-LSTM has precision 98.14 %, recall 97.24 %, F1 0.977, and its accuracy (99.13 %) is below 99.39 %.
* k-NN is not "0.945 recall": its confusion matrix [[1865, 1], [125, 309]] gives a **seizure recall of only 71.2 %** (it misses 125 of 434 seizures).
* In this repository every table uses seizure-class precision, recall (= sensitivity), specificity and F1, so the numbers are not comparable with the draft's Tables 4–8 – replace them.

### (2) Logistic Regression: wrong name and wrong accuracy
* The code trains `LogisticRegression`, but the text calls it "Linear Regression" (a regression method, not a classifier). Rename it **Logistic Regression**.
* Table 8 of the draft prints accuracy 0.9734 next to precision = recall = 0.827. With micro-averaging these three must be identical, so one is wrong. In the appendix code the LR cell
  computes `accuracy_score(expected_classes, predict_classes)` – `predict_classes` belongs to the *previous* cell – while precision/recall use `predicted_classes`.
  **The real accuracy of Logistic Regression is about 82–83 %** (this repository: 82.4 %, ROC-AUC 0.53). Correct the table and the sentence "accuracy of 0.97".

### (3) Data split and scaling
* The text says 7 : 2 : 1, the code gives 72 % / 8 % / 20 %. Here it is exactly 70 % train / 10 % validation / 20 % test.
* The scaler was fitted on **all** data before splitting (a small leak). Here it is fitted on the training part only.
* Random split vs recording split: see `docs/3_HOW_IT_WORKS.md`, section 3.6 – the check is described there, and the numbers hold up.

### (4) Smaller points
* **Output layer:** the draft says *sigmoid* with 2 neurons and categorical cross-entropy; this project uses *softmax* (the matching choice). Parameter counts do not change.
* **Parameter count:** the best CNN-LSTM in Table 4 has 1,17,242 parameters, but the architecture in Table 3 and in the appendix code has **127,578** trainable parameters. Use one consistent description.
* **Class descriptions:** some copies of the data set call classes 2 and 3 "tumor" recordings. In the original Bonn description they are *seizure-free recordings of epilepsy patients* (epileptogenic zone / hippocampal area). Class 4 and 5 are healthy volunteers (eyes closed / open).
* **Abbreviations list:** "EEG: Electrocardiogram" should be **Electroencephalogram** (ECG is the heart).
* Table 7 gives 99.3 % for the CNN-LSTM, Table 4 gives 99.13 % – make them agree.
* "Date of submission" on the title page is empty, "Ghant chart" should read "Gantt chart", and Chapter 4 appears twice in the numbering.
* The **Raspberry Pi** part is now implemented in `raspberry_pi/` (NumPy-only inference; verified against Keras, not yet run on real Pi hardware).

## 4.3 What the draft aimed for – and what was reached

| Aim in the draft | Result |
|---|---|
| Train LSTM, CNN, CNN-LSTM, RF, k-NN, LR on the UCI data | done (see 4.1) |
| Combine machine learning and deep learning | done: CNN-LSTM + RF and an ensemble – no gain over CNN-LSTM alone |
| Exceed 99.6 % / 99.54 % accuracy of earlier CNN / LSTM work | **not reached** (best 98.96 %) |
| Run on a Raspberry Pi | code ready and verified on a PC; hardware test still to do |

## 4.4 Questions an examiner may ask (short answers)

1. **Why CNN-LSTM?** The convolution layers find short wave shapes (spikes, sharp waves); the LSTM layers follow how these shapes change over the second. It was the best model and needs far fewer parameters than the previous study's model.
2. **Why a validation set?** To choose settings and to stop training at the right time without touching the test data. The test set is used once, for the final numbers.
3. **Why not only accuracy?** 80 % of the data is "no seizure"; a program that always says "no seizure" has 80 % accuracy and finds no seizure. Sensitivity (recall) and precision tell the real story.
4. **What is over-fitting and what did you do against it?** Learning the training data by heart. Dropout, batch normalisation, early stopping and a separate validation set.
5. **Is there data leakage?** In a random split, pieces of one recording are in both train and test. We therefore ran a recording split; the results were very close (CNN-LSTM 98.96 % vs 98.30 %). Remaining limit: several recordings come from the same few patients and the data has no patient IDs.
6. **Is the difference between your models real?** Look at the 95 % confidence intervals. In protocol B the intervals of the deep models overlap almost completely, so we do not claim a winner there.
7. **Can it be used in hospitals?** Not yet. One public data set, 1-second single-channel segments, no clinical validation.
8. **What would you do next?** Multi-channel data (e.g. CHB-MIT), patient-independent testing, seizure *prediction* (before onset), real-time acquisition hardware with the Raspberry Pi, model explanation.
9. **How does the Raspberry Pi version work?** The trained weights are exported to a file; a small NumPy program recomputes the same network on the Pi (about 7 ms per second of EEG on our test machine), and can light an LED when a seizure is found.
