"""
Central configuration for all experiments.
Edit this file to change hyperparameters, paths, or model choices.
"""

from dataclasses import dataclass, field
from typing import List, Optional
import torch


@dataclass
class DataConfig:
    data_path: str = "data/labels.csv"          # CSV with 'text' + 6 aspect columns
    text_col: str = "text"
    aspect_cols: List[str] = field(default_factory=lambda: [
        "Monetary_Financial",
        "Inflation_Prices",
        "Real_Economic_Activity",
        "Labor_Consumption",
        "Fiscal_Government",
        "External_Sector",
    ])
    train_frac: float = 0.80
    val_frac:   float = 0.10
    test_frac:  float = 0.10
    seed: int = 42

    # Soft-label cleaning
    abstain_low:  float = 0.35   # mask BCE for labels in (abstain_low, abstain_high)
    abstain_high: float = 0.55
    # Override per aspect — None means use the defaults above
    abstain_overrides: dict = field(default_factory=lambda: {
        "Fiscal_Government": (0.35, 0.45),  # tighter upper bound → keep more positives
    })
    zero_out_all_negative: bool = True   # hard-zero confirmed off-domain rows


@dataclass
class ModelConfig:
    # ── Fine-tune target ──────────────────────────────────────────────
    model_name: str = "beethogedeon/Modern-FinBERT-large"
    num_labels: int = 6
    max_length: int = 128       # sentences are short; 128 is safe & efficient
    dropout: float = 0.1

    # ── Baseline ─────────────────────────────────────────────────────
    baseline_model_name: str = "ProsusAI/finbert"   # classic FinBERT, 3-class head


@dataclass
class TrainConfig:
    output_dir: str = "outputs"
    batch_size: int = 16
    eval_batch_size: int = 32
    num_epochs_phase1: int = 3    # head warmup  (backbone frozen)
    num_epochs_phase2: int = 8    # top-4 layers + head
    num_epochs_phase3: int = 3    # full fine-tune (optional, use if val F1 plateaus)

    lr_phase1: float = 3e-4       # higher LR fine for head-only
    lr_phase2: float = 2e-5
    lr_phase3: float = 5e-6

    warmup_ratio: float = 0.06    # fraction of total steps used for linear warmup
    weight_decay: float = 0.01
    grad_clip: float = 1.0

    # pos_weight for BCEWithLogitsLoss — derived from hard prevalence
    # order matches aspect_cols above
    # Fiscal cap at 8 (uncapped ~14.4 is too aggressive)
    pos_weight: List[float] = field(default_factory=lambda: [
        2.33,   # Monetary_Financial
        1.92,   # Inflation_Prices
        0.76,   # Real_Economic_Activity
        1.55,   # Labor_Consumption
        12.00,   # Fiscal_Government  (capped from ~14.4)
        1.28,   # External_Sector
    ])

    # Early stopping
    patience: int = 4             # epochs without val macro-F1 improvement
    min_delta: float = 1e-4

    # Threshold tuning (applied post-training on val set)
    threshold_search: List[float] = field(default_factory=lambda: [
        0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6
    ])

    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    fp16: bool = True             # used by baseline (FinBERT)
    fp16_finetune: bool = False   # ModernBERT: cast to fp32 instead
    save_best_only: bool = True
    log_every_n_steps: int = 50


# Convenience singleton
DATA_CFG  = DataConfig()
MODEL_CFG = ModelConfig()
TRAIN_CFG = TrainConfig()
