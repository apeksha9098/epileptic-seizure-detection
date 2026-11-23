# Raspberry Pi version

The report’s future-work section says *“we are working to run our algorithm codes in Raspberry Pi”*.
This folder does exactly that – **without installing TensorFlow on the Pi**.

**How it works:** on your PC/Colab the trained network is exported to two small files (weights + layer list).
On the Raspberry Pi a tiny program written with **NumPy only** (`rpi_predict.py`) re-builds the same network
(Conv1D, LSTM, batch-norm, pooling, dense) and predicts.
`verify_export.py` proves that it gives the same answers as the original Keras model
(difference below 0.000001 in our test).

> Why not TensorFlow Lite? The plain CNN converts, but LSTM layers need “Flex” operators that the small Raspberry Pi
> runtime does not contain. The NumPy engine runs **all three models**, also the best one (CNN-LSTM).

> ⚠️ **Honest limits.** (1) The code was tested on a PC; we could not test on real Raspberry Pi hardware here – the steps below are
> standard, but please try them and tell us if something differs. (2) The model is trained on the Bonn data set
> (single channel, 173.61 Hz). Real EEG from another device needs the same sampling rate and filtering before it can
> be used – and **this is a research prototype, not a medical device.**

## What you need

| Item | Notes |
|---|---|
| Raspberry Pi 4 Model B (2–4 GB) | Pi 3 also works, only slower |
| microSD card ≥ 16 GB + power supply | |
| Raspberry Pi OS (64-bit or 32-bit) | install with **Raspberry Pi Imager** (https://www.raspberrypi.com/software/) |
| *(optional)* LED + 330 Ω resistor or a buzzer | shows the “seizure” alarm on a GPIO pin |

## Step 1 – Prepare the files (on your PC or Google Colab)

```bash
pip install tensorflow            # only on the PC/Colab, not on the Pi
python raspberry_pi/export_for_raspberry_pi.py --model results/random_split/models/cnn_lstm.keras --out raspberry_pi/model/cnn_lstm
cp results/random_split/models/scaler.json raspberry_pi/model/scaler.json
python raspberry_pi/verify_export.py --model results/random_split/models/cnn_lstm.keras --prefix raspberry_pi/model/cnn_lstm
```
The last command must print `RESULT: OK`. (The folder `raspberry_pi/model/` with the three exported files is already included in this repository.)

## Step 2 – Set up the Raspberry Pi

1. Boot the Pi, connect to Wi-Fi, open **Terminal**.
2. Install the two small packages:
   ```bash
   sudo apt update
   sudo apt install -y python3-numpy python3-gpiozero
   ```
3. Copy the folder `raspberry_pi` to the Pi: with a USB stick, or from your PC:
   ```bash
   scp -r raspberry_pi pi@raspberrypi.local:~/
   ```
   (or download the repository on the Pi: `git clone https://github.com/YOUR-GITHUB-USERNAME/epileptic-seizure-detection.git`)

## Step 3 – Run

```bash
cd ~/raspberry_pi
python3 rpi_predict.py --demo                       # 20 example EEG seconds: 10 seizure + 10 non-seizure
python3 rpi_predict.py --csv my_eeg.csv --delay 1   # your own file, one segment per second ("real time")
python3 rpi_predict.py --demo --led-pin 17          # LED on GPIO17 lights up on a seizure
```
You will see one line per second, e.g. `  3   0.9987   *** SEIZURE ***`, and at the end the average computing time per segment.

**LED wiring (BCM numbering):** GPIO17 (physical pin 11) → 330 Ω resistor → LED long leg (+); LED short leg (–) → GND (physical pin 9).

## Files

| File | Runs on | Purpose |
|---|---|---|
| `export_for_raspberry_pi.py` | PC / Colab | converts a `.keras` model to `.npz` + `.json` |
| `verify_export.py` | PC / Colab | checks that both versions agree |
| `rpi_predict.py` | **Raspberry Pi** (NumPy only) | the detector |
| `model/` | – | exported CNN-LSTM + `scaler.json` |
| `example_segments.csv` | – | 20 EEG seconds for the demo (from the data set) |
