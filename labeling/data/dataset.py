"""
data/dataset.py
───────────────
Handles:
  • Soft-label cleaning  (zero out confirmed off-domain rows, abstain masking)
  • Stratified multi-label train/val/test split  (iterative stratification)
  • PyTorch Dataset + collator
"""

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from typing import Tuple, Dict
from sklearn.model_selection import train_test_split

from config import DataConfig, ModelConfig


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Label Cleaning
# ─────────────────────────────────────────────────────────────────────────────

def clean_labels(df: pd.DataFrame, cfg: DataConfig) -> pd.DataFrame:
    """
    Two cleaning passes:
      A) Hard-zero confirmed off-domain rows (all aspects below 0.5 threshold).
         We validated manually that these are genuinely non-economic sentences.
      B) Build an abstain mask (stored separately) so masked BCE can ignore
         the [abstain_low, abstain_high] band during loss computation.
    """
    df = df.copy()

    # Pass A — zero out all-negative rows
    if cfg.zero_out_all_negative:
        all_neg = (df[cfg.aspect_cols] < 0.5).all(axis=1)
        df.loc[all_neg, cfg.aspect_cols] = 0.0
        print(f"[clean_labels] Zeroed {all_neg.sum()} off-domain rows")

    # Pass B — abstain mask columns (1 = use in loss, 0 = ignore)
    for col in cfg.aspect_cols:
        mask_col = f"_mask_{col}"
        in_abstain_zone = (
            (df[col] > cfg.abstain_low) & (df[col] < cfg.abstain_high)
        )
        df[mask_col] = (~in_abstain_zone).astype(np.float32)

    abstain_count = sum(
        ((df[col] > cfg.abstain_low) & (df[col] < cfg.abstain_high)).sum()
        for col in cfg.aspect_cols
    )
    print(f"[clean_labels] {abstain_count} aspect-labels masked as abstain "
          f"(in [{cfg.abstain_low}, {cfg.abstain_high}])")

    return df


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Stratified Split
# ─────────────────────────────────────────────────────────────────────────────

def _multilabel_stratified_split(
    df: pd.DataFrame,
    aspect_cols,
    train_frac: float,
    val_frac: float,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Approximate multi-label stratification using hard labels.
    Uses iterative stratification via skmultilearn if available,
    falls back to a simple random split with a printed warning.
    """
    hard = (df[aspect_cols] >= 0.5).astype(int)

    try:
        from skmultilearn.model_selection import iterative_train_test_split

        X = np.arange(len(df)).reshape(-1, 1)
        y = hard.values

        test_frac  = 1.0 - train_frac - val_frac
        X_tv, _, X_test, _ = iterative_train_test_split(X, y, test_size=test_frac)

        # Now split train+val
        tv_idx   = X_tv.flatten()
        test_idx = X_test.flatten()
        df_tv    = df.iloc[tv_idx].reset_index(drop=True)
        df_test  = df.iloc[test_idx].reset_index(drop=True)
        hard_tv  = hard.iloc[tv_idx].values

        val_share = val_frac / (train_frac + val_frac)
        X2 = np.arange(len(df_tv)).reshape(-1, 1)
        X_train, _, X_val, _ = iterative_train_test_split(X2, hard_tv, test_size=val_share)

        df_train = df_tv.iloc[X_train.flatten()].reset_index(drop=True)
        df_val   = df_tv.iloc[X_val.flatten()].reset_index(drop=True)

        print("[split] Used iterative stratification (skmultilearn)")

    except ImportError:
        print("[split] WARNING: skmultilearn not found — using random split. "
              "Install with: pip install scikit-multilearn")
        test_frac = 1.0 - train_frac - val_frac
        df_train, df_temp = train_test_split(df, test_size=(1 - train_frac), random_state=seed)
        rel_val = val_frac / (val_frac + test_frac)
        df_val, df_test = train_test_split(df_temp, test_size=(1 - rel_val), random_state=seed)
        df_train = df_train.reset_index(drop=True)
        df_val   = df_val.reset_index(drop=True)
        df_test  = df_test.reset_index(drop=True)

    print(f"[split] train={len(df_train)}  val={len(df_val)}  test={len(df_test)}")
    return df_train, df_val, df_test


def prepare_splits(df: pd.DataFrame, cfg: DataConfig):
    df = clean_labels(df, cfg)
    return _multilabel_stratified_split(
        df, cfg.aspect_cols, cfg.train_frac, cfg.val_frac, cfg.seed
    )


# ─────────────────────────────────────────────────────────────────────────────
# 3.  PyTorch Dataset
# ─────────────────────────────────────────────────────────────────────────────

class AspectDataset(Dataset):
    """
    Returns:
        input_ids, attention_mask  — tokenized text
        labels                     — soft float tensor [6]
        mask                       — BCE mask tensor   [6]  (1=use, 0=ignore)
    """

    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer: AutoTokenizer,
        cfg_data: DataConfig,
        cfg_model: ModelConfig,
    ):
        self.texts   = df[cfg_data.text_col].tolist()
        self.labels  = torch.tensor(
            df[cfg_data.aspect_cols].values, dtype=torch.float32
        )
        mask_cols = [f"_mask_{c}" for c in cfg_data.aspect_cols]
        self.masks = torch.tensor(df[mask_cols].values, dtype=torch.float32)

        self.tokenizer  = tokenizer
        self.max_length = cfg_model.max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        # Extract base features safely
        input_ids = enc["input_ids"].squeeze(0)
        attention_mask = enc["attention_mask"].squeeze(0)
        
        # Safe fallback if 'token_type_ids' is missing from the tokenizer output
        token_type_ids = enc.get(
            "token_type_ids", 
            torch.zeros_like(input_ids)
        ).squeeze(0)

        return {
            "input_ids":      input_ids,
            "attention_mask": attention_mask,
            "labels":         self.labels[idx],
            "mask":           self.masks[idx],
            #"token_type_ids":   token_type_ids,
        }


# ─────────────────────────────────────────────────────────────────────────────
# 4.  DataLoader factory
# ─────────────────────────────────────────────────────────────────────────────

def make_loaders(
    df_train: pd.DataFrame,
    df_val:   pd.DataFrame,
    df_test:  pd.DataFrame,
    cfg_data:  DataConfig,
    cfg_model: ModelConfig,
    cfg_train,
    tokenizer = None,
) -> Tuple[DataLoader, DataLoader, DataLoader]:

    if tokenizer is None:    # ← and this
        tokenizer = AutoTokenizer.from_pretrained(cfg_model.model_name)

    train_ds = AspectDataset(df_train, tokenizer, cfg_data, cfg_model)
    val_ds   = AspectDataset(df_val,   tokenizer, cfg_data, cfg_model)
    test_ds  = AspectDataset(df_test,  tokenizer, cfg_data, cfg_model)

    train_loader = DataLoader(train_ds, batch_size=cfg_train.batch_size,
                              shuffle=True,  num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=cfg_train.eval_batch_size,
                              shuffle=False, num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=cfg_train.eval_batch_size,
                              shuffle=False, num_workers=2, pin_memory=True)

    return train_loader, val_loader, test_loader
