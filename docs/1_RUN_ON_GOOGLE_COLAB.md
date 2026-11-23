# 1. Run the project on Google Colab (step by step)

Google Colab is a free website where you can run Python code **without installing anything**.
You only need a Google account (Gmail).

> **Time needed:** about 5 minutes for a test run, about 20–30 minutes for the full training.

---

## Step 1 – Open Google Colab

1. Open **https://colab.research.google.com** in Chrome.
2. Sign in with your Google account.
3. If a window "Open notebook" appears, you will use it in Step 2.

## Step 2 – Open the project notebook (choose ONE way)

### Way A – Without GitHub (easiest, do this first)

1. In Colab click **File → Upload notebook**.
2. Click **Browse** and select the file `Epileptic_Seizure_Detection_Colab.ipynb`
   (it is in the folder `notebooks/` of the project).
3. The notebook opens. Continue with Step 3.
   *Later, when the notebook asks for the project files, you will upload the project ZIP file (explained in Step 4).*

### Way B – From your own GitHub repository (after you have pushed the project)

1. In Colab click **File → Open notebook → GitHub** tab.
2. Paste your repository link, for example `https://github.com/YOUR-GITHUB-USERNAME/epileptic-seizure-detection`
3. Click the notebook `Epileptic_Seizure_Detection_Colab.ipynb`.

> Or simply click the **“Open in Colab”** badge at the top of the `README.md` of your repository.
> In the notebook’s **Settings** cell type your GitHub username into the box (explained in Step 4).

## Step 3 – (Optional) choose a faster computer

* Click **Runtime → Change runtime type**.
* Choose **T4 GPU** and click **Save**.
* A GPU is *optional*. The project also runs on the normal CPU, only a little slower.

## Step 4 – Run the notebook

1. Find the cell called **“Settings”**. In Colab it shows **two check-boxes and two text boxes** (no coding needed):
   * ☑ **QUICK_TEST** ticked → 3–5 minute test (only to see that everything works).
     **Unticked** → the **real** training (about 20–30 minutes).
     **Recommended: run once ticked, then untick it and run again.**
   * ☑ **RUN_LEAK_FREE_CHECK** → also repeats everything with the stricter “recording split” (leave it ticked).
   * **GITHUB_USERNAME** → type your GitHub username **only** if you opened the notebook from your own repository (Way B). Otherwise leave it.
   * **REPO_NAME** → leave it as it is.
2. Click **Runtime → Run all** (or press **Ctrl + F9**).
3. In the cell **“1. Get the project files”**:
   * If you typed your GitHub username (Way B), the project is downloaded automatically.
   * Otherwise (Way A) a button **“Choose Files”** appears. Click it and select
     `epileptic-seizure-detection.zip` (the project ZIP). The notebook unzips it by itself.
4. Wait. You will see progress bars and numbers. **Do not close the browser tab.**

## Step 5 – What you will see (and what it means)

| Notebook step | What happens | What you will see |
|---|---|---|
| Load data | reads `data/data.csv` (11,500 EEG pieces) | a table and pictures of EEG signals |
| Prepare data | seizure = 1 / non-seizure = 0, split 70 % train / 10 % validation / 20 % test, normalise | numbers of rows in train / validation / test |
| Machine learning | trains Random Forest, k-NN, Logistic Regression | accuracy of each |
| Deep learning | trains LSTM, 1D-CNN, CNN-LSTM | lines like `Epoch 12/60 … val_accuracy: 0.98` and learning curves |
| Evaluation | tests all models on data they never saw | results table, confusion matrices, ROC curves |
| Hybrid | combines machine learning + deep learning | two extra rows in the table |
| Honest check | repeats everything with recordings kept together | a second table (numbers are usually close to the first one – see `docs/3_HOW_IT_WORKS.md`) |
| Save | creates `results_colab.zip` | the file is downloaded to your computer |

## Step 6 – Download your results

* The last cell downloads `results_colab.zip` automatically. If your browser blocks it, click
  the **folder icon** on the left of Colab → open `results_colab.zip` → three dots → **Download**.
* **Important:** Colab deletes everything when you close it. Download the results before you leave!

---

## Common problems

| Problem | Solution |
|---|---|
| `ModuleNotFoundError: No module named 'src'` | You are not inside the project folder. Run the cell “Step 1 – get the project files” again and upload the ZIP. |
| `FileNotFoundError … data.csv` | The dataset is missing. Download it from https://archive.ics.uci.edu/dataset/388/epileptic+seizure+recognition (or Kaggle), unzip, and drag `data.csv` into the `data` folder (left sidebar). |
| “Runtime disconnected” | Free Colab disconnects after idle time. Keep the tab open, press **Reconnect**, then **Runtime → Run all** again. |
| It is very slow | Use `QUICK_TEST = True` first. Or change the runtime to GPU (Step 3). |
| My numbers differ a little from the README | Normal. Neural networks start with random numbers, so results change by about ±0.5 %. |
| Red warning text about `cuda` / `oneDNN` | Harmless – you can ignore it. |

## Run on your own computer (optional)

```bash
git clone https://github.com/YOUR-GITHUB-USERNAME/epileptic-seizure-detection.git
cd epileptic-seizure-detection
python -m venv .venv
# Windows:  .venv\Scripts\activate        Mac/Linux:  source .venv/bin/activate
pip install -r requirements.txt
python scripts/run_all.py --quick      # 2-minute test
python scripts/run_all.py --split both # full run (both ways of splitting)
```
