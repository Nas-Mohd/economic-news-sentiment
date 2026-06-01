"""
train.py
────────
Entry point. Runs:
  1. Data prep  (cleaning, splitting, DataLoaders)
  2. Baseline   (ProsusAI/finbert + 6-way head)
  3. Fine-tune  (Modern-FinBERT-large + 6-way head)
  4. Threshold tuning on val set for each model
  5. Final evaluation on held-out test set
  6. Side-by-side comparison printed + saved as CSV

Usage:
    python train.py --data path/to/labels.csv

Your CSV must have:
  - a 'text' column
  - columns named exactly: Monetary_Financial, Inflation_Prices,
    Real_Economic_Activity, Labor_Consumption, Fiscal_Government, External_Sector
  - values in [0, 1]  (the Snorkel probabilities)
"""

import argparse
import pandas as pd
import torch

from config import DATA_CFG, MODEL_CFG, TRAIN_CFG
from data.dataset import prepare_splits, make_loaders
from models.model import AspectClassifier, BaselineClassifier
from utils.loss import build_loss_fn
from utils.trainer import Trainer
from utils.metrics import (
    collect_predictions,
    tune_thresholds,
    evaluate,
    print_comparison,
    results_to_dataframe,
)


def main(data_path: str):
    # ── 0. Load raw data ─────────────────────────────────────────────
    print(f"\n[main] Loading data from {data_path}")
    df = pd.read_csv(data_path)
    print(f"[main] {len(df)} sentences, columns: {list(df.columns)}")

    # ── 1. Prepare splits ────────────────────────────────────────────
    DATA_CFG.data_path = data_path
    df_train, df_val, df_test = prepare_splits(df, DATA_CFG)

    # ── 2. Build DataLoaders ─────────────────────────────────────────
    train_loader, val_loader, test_loader = make_loaders(
        df_train, df_val, df_test, DATA_CFG, MODEL_CFG, TRAIN_CFG
    )

    # ── 3. Loss function (shared by both models) ─────────────────────
    loss_fn = build_loss_fn(TRAIN_CFG)

    # ── 4. Train BASELINE ────────────────────────────────────────────
    print("\n" + "█" * 60)
    print("  BASELINE: ProsusAI/finbert")
    print("█" * 60)

    baseline = BaselineClassifier(MODEL_CFG)
    baseline_trainer = Trainer(
        model=baseline,
        loss_fn=loss_fn,
        train_loader=train_loader,
        val_loader=val_loader,
        cfg=TRAIN_CFG,
        model_name="baseline_finbert",
    )
    baseline = baseline_trainer.fit()

    # ── 5. Train FINE-TUNE ───────────────────────────────────────────
    print("\n" + "█" * 60)
    print("  FINE-TUNE: Modern-FinBERT-large")
    print("█" * 60)

    finetune = AspectClassifier(MODEL_CFG)
    ft_trainer = Trainer(
        model=finetune,
        loss_fn=loss_fn,
        train_loader=train_loader,
        val_loader=val_loader,
        cfg=TRAIN_CFG,
        model_name="modern_finbert_large",
    )
    finetune = ft_trainer.fit()

    # ── 6. Threshold tuning on val set ───────────────────────────────
    print("\n[main] Tuning thresholds on validation set...")

    bl_val_probs, val_targets = collect_predictions(baseline, val_loader, TRAIN_CFG.device)
    ft_val_probs, _           = collect_predictions(finetune,  val_loader, TRAIN_CFG.device)

    bl_thresholds = tune_thresholds(bl_val_probs, val_targets,
                                    TRAIN_CFG.threshold_search, DATA_CFG.aspect_cols)
    ft_thresholds = tune_thresholds(ft_val_probs, val_targets,
                                    TRAIN_CFG.threshold_search, DATA_CFG.aspect_cols)

    # ── 7. Final evaluation on TEST set ─────────────────────────────
    print("\n[main] Evaluating on test set...")

    bl_test_probs, test_targets = collect_predictions(baseline, test_loader, TRAIN_CFG.device)
    ft_test_probs, _            = collect_predictions(finetune,  test_loader, TRAIN_CFG.device)

    bl_results = evaluate(bl_test_probs, test_targets,
                          thresholds=bl_thresholds,
                          aspect_cols=DATA_CFG.aspect_cols,
                          label="Baseline (FinBERT)")

    ft_results = evaluate(ft_test_probs, test_targets,
                          thresholds=ft_thresholds,
                          aspect_cols=DATA_CFG.aspect_cols,
                          label="Modern-FinBERT-large")

    # ── 8. Print + save comparison ───────────────────────────────────
    print_comparison([bl_results, ft_results], DATA_CFG.aspect_cols)

    out_df = results_to_dataframe([bl_results, ft_results], DATA_CFG.aspect_cols)
    out_path = f"{TRAIN_CFG.output_dir}/comparison_results.csv"
    out_df.to_csv(out_path, index=False)
    print(f"\n[main] Results saved → {out_path}")

    # ── 9. Save training logs ────────────────────────────────────────
    import json
    for trainer, name in [(baseline_trainer, "baseline"), (ft_trainer, "finetune")]:
        log_path = f"{TRAIN_CFG.output_dir}/{name}_training_log.json"
        with open(log_path, "w") as f:
            json.dump(trainer.log, f, indent=2)
        print(f"[main] Training log → {log_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        type=str,
        default=DATA_CFG.data_path,
        help="Path to CSV with text + 6 aspect probability columns",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Where to save checkpoints and results (overrides config.py)",
    )
    args = parser.parse_args()
    if args.output_dir:
        TRAIN_CFG.output_dir = args.output_dir
    main(args.data)
