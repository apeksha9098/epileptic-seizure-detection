# 2. Put the project on GitHub (step by step)

GitHub is a website that stores your project online, so teachers, examiners and employers can see it with one link.
Choose **ONE** of the three methods. **Method A needs no installation – start there.**

| Method | Needs | Difficulty |
|---|---|---|
| **A. Website upload (drag and drop)** | only a browser | ⭐ easiest |
| **B. GitHub Desktop** | a free program | ⭐⭐ |
| **C. Git commands** | Git installed | ⭐⭐⭐ |

---

## Before you start

1. **Create a GitHub account** at https://github.com → *Sign up* → confirm your e-mail.
   (Your username becomes part of your project link, e.g. `github.com/sabita-k/…` – choose something professional.)
2. **Unzip** `epileptic-seizure-detection.zip` on your computer (right-click → *Extract all*).
   You now have a folder `epileptic-seizure-detection` that contains `README.md`, `src`, `data`, `docs`, …
3. Decide if you want the **team names** in `README.md` to be public. A public repository can be seen by everybody.
   You can edit or delete the names later.

---

## Method A – Upload with the website (no installation)

### A1. Create an empty repository
1. Log in to GitHub. Click the **“+”** (top right) → **New repository**.
2. **Repository name:** `epileptic-seizure-detection`
3. **Description:** `Epileptic seizure detection from EEG using LSTM, 1D-CNN and CNN-LSTM (Python, TensorFlow)`
4. Choose **Public**.
5. **Do NOT** tick “Add a README”, “.gitignore” or “license” (the project already has them).
6. Click **Create repository**.

### A2. Upload the files
1. On the new page click the blue link **“uploading an existing file”**.
2. Open the unzipped project folder on your computer. Select **everything inside it**
   (`README.md`, `LICENSE`, `requirements.txt`, `.gitignore`, and the folders `data`, `src`, `scripts`, `notebooks`, `results`, `docs`, `raspberry_pi`, `extras`)
   and **drag them into the browser window**.
   * Use **Chrome or Edge** – they keep the folder structure when you drag folders.
   * Drag the **contents** of the folder, not the ZIP file and not the outer folder itself.
3. Wait until every file shows a green tick (the dataset is 7.6 MB – give it a minute).
4. Scroll down. In the box *“Commit changes”* write: `Initial commit: seizure detection project`
5. Click **Commit changes**.

> The project has about 70 files, so one upload is normally enough. The file `.gitignore` is hidden on Mac – if you cannot see it, simply skip it (nothing breaks).
> If GitHub says there are too many files (limit ≈ 100 files per upload), upload the folders in two or three rounds:
> first `src`, `scripts`, `docs`, `notebooks`, then `data`, then `results`, `raspberry_pi`, `extras` and the loose files.
> (Each round: **Add file → Upload files → drag → Commit**.)

### A3. Write your username into the README
The “Open in Colab” button in the README must know **your** GitHub username.

1. In your repository click `README.md` → click the **pencil icon** (Edit).
2. Replace every `YOUR-GITHUB-USERNAME` with your real username (there are 2 places: the “Open in Colab” badge and the `git clone` line).
   Click **Commit changes**.
3. The notebook itself needs **no editing**: when you open it in Colab you just type your username into the
   **GITHUB_USERNAME** box of the *Settings* cell.

### A4. Check your work ✅
* Your repository page shows the **README** with the title, tables and pictures.
* Click the **Open in Colab** badge → the notebook opens in Colab.
* Copy the page address (e.g. `https://github.com/yourname/epileptic-seizure-detection`) – **this is the link you give to your teachers**.

---

## Method B – GitHub Desktop

1. Download **GitHub Desktop** from https://desktop.github.com → install → sign in with your GitHub account.
2. **File → Add local repository…** → choose the unzipped project folder.
   If it says “This directory does not appear to be a Git repository”, click **create a repository**, keep the defaults, click **Create repository**.
3. At the bottom left write a summary: `Initial commit` → click **Commit to main**.
4. Click **Publish repository** (top bar). Name: `epileptic-seizure-detection`.
   **Untick** “Keep this code private” if you want it public. Click **Publish repository**.
5. Do step A3 (replace the username in `README.md`) – in GitHub Desktop: edit the file in any text editor, then commit and click **Push origin**.

---

## Method C – Git commands (Windows: *Git Bash*, Mac/Linux: *Terminal*)

1. Install Git: https://git-scm.com/downloads
2. One-time setup (use your own name and e-mail):
   ```bash
   git config --global user.name  "Your Name"
   git config --global user.email "you@example.com"
   ```
3. Create the empty repository on GitHub (step A1), then in the terminal:
   ```bash
   cd path/to/epileptic-seizure-detection
   git init
   git add .
   git commit -m "Initial commit: seizure detection project"
   git branch -M main
   git remote add origin https://github.com/YOUR-GITHUB-USERNAME/epileptic-seizure-detection.git
   git push -u origin main
   ```
4. A window asks you to **sign in to GitHub** – do it in the browser.
   If it asks for a *password* in the terminal: GitHub no longer accepts your normal password. Use a **Personal Access Token**:
   GitHub → profile picture → **Settings → Developer settings → Personal access tokens → Tokens (classic) → Generate new token**,
   tick **repo**, copy the token and paste it as the password. **Never share the token or put it into a file.**

### Update later (after you changed something)
```bash
git add .
git commit -m "Describe what you changed"
git push
```

---

## Make the repository look professional (5 minutes)

* **About box** (gear icon on the right of the repository page): add *Topics*:
  `epilepsy` `eeg` `seizure-detection` `deep-learning` `lstm` `cnn` `tensorflow` `raspberry-pi`
* Keep the `README.md` results tables – examiners read them first.
* Do **not** upload passwords, tokens or private patient data.
* The dataset in `data/` is public research data; its source and credit are in `data/README.md`.

## Common problems

| Message / problem | What to do |
|---|---|
| “File is too large” (web upload) | Web upload accepts files up to 25 MB. All project files are smaller than that. Do not add big new files. |
| Folders appear as one flat list of files | You used a browser that does not support folder drag-and-drop. Use **Chrome/Edge** or Method B. |
| `remote: Permission denied` / `Authentication failed` | Wrong username or you used your password. Use the browser sign-in or a Personal Access Token (Method C step 4). |
| `failed to push some refs … fetch first` | The online repository has something your computer does not. Run `git pull --rebase origin main`, then `git push`. |
| `remote origin already exists` | Run `git remote set-url origin https://github.com/YOUR-GITHUB-USERNAME/epileptic-seizure-detection.git` |
| README pictures do not show | A folder was uploaded without its sub-folders. Re-upload the `results` folder. |

## Final checklist

- [ ] The repository is **Public** and has the link in your report/presentation.
- [ ] `README.md` shows the title, results and pictures.
- [ ] The **Open in Colab** badge works.
- [ ] `YOUR-GITHUB-USERNAME` does not appear in `README.md` any more.
- [ ] You can explain the project in 2 minutes (read `docs/3_HOW_IT_WORKS.md`).
