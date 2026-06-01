"""
utils/metrics.py
────────────────
All evaluation logic:
  • Per-aspect F1, precision, recall
  • Macro / Micro F1
  • Exact match ratio
  • Per-aspect threshold tuning on validation set
  • Pretty-print comparison table (fine-tune vs baseline)
"""

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, precision_score, recall_score
from typing import Dict, List, Tuple
import pandas as pd


ASPECT_COLS = [
    "Monetary_Financial",
    "Inflation_Prices",
    "Real_Economic_Activity",
    "Labor_Consumption",
    "Fiscal_Government",
    "External_Sector",
]


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Inference — collect logits and targets from a DataLoader
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def collect_predictions(model, loader: DataLoader, device: str):
    """Returns (probs, targets) as numpy arrays, shape [N, 6]."""
    model.eval()
    all_probs, all_targets = [], []

    for batch in loader:
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        targets        = batch["labels"].cpu().numpy()

        logits = model(input_ids, attention_mask)
        probs  = torch.sigmoid(logits).cpu().numpy()

        all_probs.append(probs)
        all_targets.append(targets)

    return np.concatenate(all_probs), np.concatenate(all_targets)


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Threshold tuning — find best per-aspect threshold on val set
# ─────────────────────────────────────────────────────────────────────────────

def tune_thresholds(
    probs: np.ndarray,          # [N, 6]  validation probabilities
    targets: np.ndarray,        # [N, 6]  soft labels (binarised with 0.5 for eval)
    candidates: List[float],
    aspect_cols: List[str] = ASPECT_COLS,
) -> Dict[str, float]:
    """
    Search for the threshold that maximises F1 for each aspect independently.
    Returns a dict: {aspect_name: best_threshold}
    """
    hard_targets = (targets >= 0.5).astype(int)
    best = {}

    for j, col in enumerate(aspect_cols):
        best_f1, best_t = -1.0, 0.5
        for t in candidates:
            preds = (probs[:, j] >= t).astype(int)
            f1 = f1_score(hard_targets[:, j], preds, zero_division=0)
            if f1 > best_f1:
                best_f1, best_t = f1, t
        best[col] = best_t

    print("\n[threshold tuning]")
    for col, t in best.items():
        print(f"  {col:<30} → {t:.2f}")

    return best


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Full evaluation report
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(
    probs: np.ndarray,
    targets: np.ndarray,
    thresholds: Dict[str, float] = None,
    aspect_cols: List[str] = ASPECT_COLS,
    label: str = "Model",
) -> Dict:
    """
    Compute per-aspect and aggregate metrics.
    If thresholds is None, uses 0.5 for all aspects.
    """
    hard_targets = (targets >= 0.5).astype(int)
    thresh_vec   = np.array([
        thresholds[c] if thresholds else 0.5
        for c in aspect_cols
    ])
    preds = (probs >= thresh_vec).astype(int)

    results = {"model": label}

    # Per-aspect
    per_aspect = {}
    for j, col in enumerate(aspect_cols):
        per_aspect[col] = {
            "f1":        f1_score(hard_targets[:, j], preds[:, j], zero_division=0),
            "precision": precision_score(hard_targets[:, j], preds[:, j], zero_division=0),
            "recall":    recall_score(hard_targets[:, j], preds[:, j], zero_division=0),
            "support":   int(hard_targets[:, j].sum()),
        }
    results["per_aspect"] = per_aspect

    # Aggregate
    results["macro_f1"] = f1_score(hard_targets, preds, average="macro",  zero_division=0)
    results["micro_f1"] = f1_score(hard_targets, preds, average="micro",  zero_division=0)
    results["exact_match"] = float((preds == hard_targets).all(axis=1).mean())

    return results


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Pretty-print comparison table
# ─────────────────────────────────────────────────────────────────────────────

def print_comparison(results_list: List[Dict], aspect_cols: List[str] = ASPECT_COLS):
    """
    Prints a side-by-side comparison table for multiple models.
    Pass a list of dicts returned by evaluate().
    """
    models = [r["model"] for r in results_list]
    header = f"{'Aspect':<30}" + "".join(f"  {m:<22}" for m in models)
    print("\n" + "=" * len(header))
    print(header)
    print("=" * len(header))

    for col in aspect_cols:
        row = f"{col:<30}"
        for r in results_list:
            f1  = r["per_aspect"][col]["f1"]
            sup = r["per_aspect"][col]["support"]
            row += f"  F1={f1:.3f}  (n={sup:<5})"
        print(row)

    print("-" * len(header))
    macro_row = f"{'Macro F1':<30}" + "".join(
        f"  {r['macro_f1']:.3f}{'':19}" for r in results_list
    )
    micro_row = f"{'Micro F1':<30}" + "".join(
        f"  {r['micro_f1']:.3f}{'':19}" for r in results_list
    )
    exact_row = f"{'Exact Match':<30}" + "".join(
        f"  {r['exact_match']:.3f}{'':19}" for r in results_list
    )
    print(macro_row)
    print(micro_row)
    print(exact_row)
    print("=" * len(header))


def results_to_dataframe(results_list: List[Dict], aspect_cols: List[str] = ASPECT_COLS) -> pd.DataFrame:
    """Convert results list to a tidy DataFrame for further analysis."""
    rows = []
    for r in results_list:
        for col in aspect_cols:
            rows.append({
                "model":   r["model"],
                "aspect":  col,
                "f1":      r["per_aspect"][col]["f1"],
                "precision": r["per_aspect"][col]["precision"],
                "recall":  r["per_aspect"][col]["recall"],
                "support": r["per_aspect"][col]["support"],
            })
        rows.append({
            "model":   r["model"],
            "aspect":  "MACRO",
            "f1":      r["macro_f1"],
            "precision": None,
            "recall":  None,
            "support": None,
        })
    return pd.DataFrame(rows)
