"""
config.py  -  ALL the settings of the project in ONE place.

Beginners: you normally do not need to edit anything here.
If you want to experiment, change a number and run the project again.
"""
from pathlib import Path

# ----------------------------------------------------------------------------
# Folders
# ----------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent          # the project folder
DATA_FILE = ROOT / "data" / "data.csv"                  # the EEG dataset
RESULTS_DIR = ROOT / "results"                          # everything we create goes here

# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------
N_FEATURES = 178            # every sample = 178 EEG values (= 1 second of EEG)
SEED = 42                   # makes results repeatable

# Train : Test : Validation = 7 : 2 : 1   (as written in the project report)
TRAIN_FRAC = 0.70
TEST_FRAC = 0.20
VAL_FRAC = 0.10

# ----------------------------------------------------------------------------
# Deep-learning training
# ----------------------------------------------------------------------------
EPOCHS = 60                 # maximum number of passes over the training data
BATCH_SIZE = 64             # samples processed together in one step
LEARNING_RATE = 1e-3        # Adam optimiser start value
EARLY_STOP_PATIENCE = 12    # stop if validation loss does not improve for 12 epochs
LR_PATIENCE = 4             # halve learning rate if no improvement for 4 epochs

# Names of the three deep-learning models (same as in the report)
DL_MODELS = ["LSTM", "CNN", "CNN-LSTM"]
