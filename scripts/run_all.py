#!/usr/bin/env python
"""
run_all.py  -  run the WHOLE project with one command.

    python scripts/run_all.py                    # report-style random split (7:2:1)
    python scripts/run_all.py --split recording  # leak-free check (recordings kept together)
    python scripts/run_all.py --split both       # both of the above
    python scripts/run_all.py --quick            # 2-minute smoke test (to check everything works)

What it does:   load data -> split -> train ML models -> train deep-learning models
                -> hybrid ML+DL -> evaluate -> save tables, pictures and trained models.
Everything is saved in the folder  results/
"""
import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")          # keep the console quiet
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib                                           # noqa: E402
matplotlib.use("Agg")                                       # draw pictures without a screen
import numpy as np                                          # noqa: E402
import pandas as pd                                         # noqa: E402

from src import config, data, evaluate as ev, ml            # noqa: E402


def run_split(split_name: str, df: pd.DataFrame, args) -> pd.DataFrame:
    """Run everything for ONE way of splitting the data. Returns the results table."""
    out = Path(args.out) / f"{split_name}_split"
    figs, tabs, mods = out / "figures", out / "tables", out / "models"
    for d in (figs, tabs, mods):
        d.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 78)
    print(f" SPLIT = {split_name.upper()}   "
          f"({'rows shuffled - as in the report' if split_name == 'random' else 'recordings kept together - leak-free'})")
    print("=" * 78)
    splits = data.prepare(df, split_name, seed=args.seed)
    ev.save_json(splits.info, tabs / "split_summary.json")
    data.save_scaler(splits.scaler, mods / "scaler.json")
    i = splits.info
    print(f"rows       train/val/test : {i['rows']['train']} / {i['rows']['val']} / {i['rows']['test']}")
    print(f"seizures   train/val/test : {i['seizure_rows']['train']} / {i['seizure_rows']['val']} / {i['seizure_rows']['test']}")
    print(f"recordings train/val/test : {i['recordings']['train']} / {i['recordings']['val']} / {i['recordings']['test']}")
    print(f"recordings shared between train and test: {i['recordings_shared_between_train_and_test']}")

    prob_val, prob_test, extra = {}, {}, {}
    search_tables = []

    # ---------------- classical machine learning ----------------
    if not args.skip_ml:
        print("\n--- Machine-learning models (Random Forest, k-NN, Logistic Regression) ---")
        for name, r in ml.run_ml_models(splits, quick=args.quick).items():
            prob_val[name], prob_test[name] = r["prob_val"], r["prob_test"]
            g = r["grid_table"].copy()
            g.insert(0, "Model", name)
            search_tables.append(g)
            extra[name] = {"Train time (s)": r["seconds"], "Settings": json.dumps(r["params"])}
        pd.concat(search_tables).to_csv(tabs / "ml_hyperparameter_search.csv", index=False)

    # ---------------- deep learning ----------------
    histories, summaries, dl = {}, [], {}
    hist_file = tabs / "training_histories.json"
    if not args.skip_dl:
        from src import train as T, hybrid as H            # imported here: needs TensorFlow
        print("\n--- Deep-learning models ---")
        saved = json.loads(hist_file.read_text()) if hist_file.exists() else {}
        for name in args.models:
            m_path = mods / f"{ev.slug(name)}.keras"
            if args.resume and m_path.exists() and name in saved:
                model, info = T.load_model(m_path), saved[name]
                print(f"[DL] {name}: loaded from disk (--resume)")
            else:
                print(f"\n[DL] training {name} ...")
                model, info = T.train_dl_model(name, splits, epochs=args.epochs,
                                               batch_size=args.batch_size, seed=args.seed,
                                               patience=3 if args.quick else config.EARLY_STOP_PATIENCE,
                                               verbose=2)
                T.save_model(model, m_path)
                saved[name] = info
                ev.save_json(saved, hist_file)             # saved after EVERY model (safe if stopped)
            dl[name] = model
            histories[name] = info["history"]
            prob_val[name] = T.predict_prob(name, model, splits.X_val)
            prob_test[name] = T.predict_prob(name, model, splits.X_test)
            extra[name] = {"Trainable params": info["params_trainable"],
                           "Train time (s)": info["train_seconds"], "Epochs run": info["epochs_run"],
                           "Best epoch": info["best_epoch"]}
            lines = []
            model.summary(print_fn=lambda line, **kw: lines.append(line))
            summaries.append(f"################  {name}  ################\n" + "\n".join(lines))
        (tabs / "model_summaries.txt").write_text("\n\n".join(summaries))
        ev.plot_history_grid(histories, f"Learning curves ({split_name} split)", figs / "training_curves.png")

        # ---------------- ML + DL combined ----------------
        if "CNN-LSTM" in dl and not args.skip_ml:
            print("\n--- Hybrid: machine learning + deep learning ---")
            hy = H.cnn_lstm_plus_rf(dl["CNN-LSTM"], splits, quick=args.quick)
            prob_val["CNN-LSTM + RF"], prob_test["CNN-LSTM + RF"] = hy["prob_val"], hy["prob_test"]
            extra["CNN-LSTM + RF"] = {"Settings": "Random Forest (300 trees) on the 32 features of the CNN-LSTM"}
            members = [n for n in ("LSTM", "CNN", "CNN-LSTM", "Random Forest") if n in prob_test]
            if len(members) >= 3:
                prob_test["Ensemble (all models)"] = H.soft_vote([prob_test[n] for n in members])
                prob_val["Ensemble (all models)"] = H.soft_vote([prob_val[n] for n in members])
                extra["Ensemble (all models)"] = {"Settings": "average of: " + ", ".join(members)}

    # ---------------- evaluation on the TEST set ----------------
    metrics = {n: ev.binary_metrics(splits.y_test, p) for n, p in prob_test.items()}
    table = ev.results_table(metrics)
    table = ev.add_confidence_intervals(table, splits.y_test, prob_test, splits.groups_test)
    for col in ("Trainable params", "Train time (s)", "Epochs run", "Best epoch", "Settings"):
        table[col] = [extra.get(n, {}).get(col, "") for n in table["Model"]]
    table.to_csv(tabs / "results_test.csv", index=False)
    (tabs / "results_test.md").write_text(ev.dataframe_to_markdown(
        table.drop(columns=["Settings", "TN", "FP", "FN", "TP"])))
    ev.save_json(metrics, tabs / "metrics_test.json")
    pd.DataFrame({"y_true": splits.y_test, "original_class": splits.y5_test,
                  **{n: np.round(p, 5) for n, p in prob_test.items()}}).to_csv(
        tabs / "predictions_test.csv", index=False)
    err = []
    for n in prob_test:
        e = ev.error_breakdown(splits.y5_test, prob_test[n])
        e.insert(0, "Model", n)
        err.append(e)
    pd.concat(err).to_csv(tabs / "errors_by_original_class.csv", index=False)

    ev.plot_confusion_grid(metrics, f"Confusion matrices - test set ({split_name} split)",
                           figs / "confusion_matrices.png")
    ev.plot_roc(splits.y_test, prob_test, f"ROC curves - test set ({split_name} split)", figs / "roc_curves.png")
    ev.plot_comparison(table, f"Test-set results ({split_name} split)", figs / "comparison.png")

    print("\n" + "-" * 78 + f"\n TEST RESULTS ({split_name} split) - seizure class\n" + "-" * 78)
    show = table[["Model", "Accuracy %", "Precision %", "Recall % (sensitivity)", "Specificity %", "F1"]]
    print(show.to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    return table


def main():
    ap = argparse.ArgumentParser(description="Epileptic seizure detection - run everything.")
    ap.add_argument("--split", choices=["random", "recording", "both"], default="random")
    ap.add_argument("--models", nargs="+", default=config.DL_MODELS, choices=config.DL_MODELS)
    ap.add_argument("--epochs", type=int, default=config.EPOCHS)
    ap.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    ap.add_argument("--seed", type=int, default=config.SEED)
    ap.add_argument("--quick", action="store_true", help="tiny fast test run (2 epochs, 20%% of data)")
    ap.add_argument("--resume", action="store_true", help="re-use already trained models in results/")
    ap.add_argument("--skip-dl", action="store_true", help="do not train deep-learning models")
    ap.add_argument("--skip-ml", action="store_true", help="do not train machine-learning models")
    ap.add_argument("--out", default=str(config.RESULTS_DIR), help="output folder")
    args = ap.parse_args()

    t_start = time.time()
    if args.quick:
        args.epochs = min(args.epochs, 2)
    df = data.load_dataframe()
    if args.quick:
        df = data.quick_subsample(df, 0.2, args.seed)
        print(f"QUICK MODE: using {len(df)} rows and {args.epochs} epochs (only for testing!)")

    # pictures of the raw data (once)
    fig_dir = Path(args.out) / "figures"
    ev.plot_class_distribution(df, fig_dir / "dataset_class_distribution.png")
    ev.plot_sample_eeg(df, data.FEATURE_COLUMNS, fig_dir / "dataset_eeg_examples.png")

    splits_to_run = ["random", "recording"] if args.split == "both" else [args.split]
    for s in splits_to_run:
        run_split(s, df, args)

    info = {"args": vars(args), "python": platform.python_version(),
            "numpy": np.__version__, "pandas": pd.__version__,
            "total_minutes": round((time.time() - t_start) / 60, 2)}
    if not args.skip_dl:
        import tensorflow as tf
        info["tensorflow"] = tf.__version__
    ev.save_json(info, Path(args.out) / f"run_info_{args.split}.json")
    print(f"\nDONE in {info['total_minutes']} minutes.  Results are in:  {args.out}")


if __name__ == "__main__":
    main()
