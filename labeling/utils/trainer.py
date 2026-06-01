"""
utils/trainer.py
────────────────
Reusable Trainer for both AspectClassifier and BaselineClassifier.

Phases:
  Phase 1 — head warmup  (backbone frozen,      lr_phase1, epochs_phase1)
  Phase 2 — top layers   (top-4 + head,          lr_phase2, epochs_phase2)
  Phase 3 — full model   (all params,            lr_phase3, epochs_phase3)
             only runs if val macro-F1 is still improving

Features:
  • Linear warmup + cosine LR decay per phase
  • Gradient clipping
  • Mixed precision (fp16)
  • Early stopping on val macro-F1
  • Saves best checkpoint per phase
  • Training log returned as list of dicts for plotting
"""

import os
import copy
import time
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler
from torch.amp import autocast
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup
from typing import List, Dict, Optional
import numpy as np

from config import TrainConfig
from utils.loss import MaskedSoftBCE
from utils.metrics import collect_predictions, evaluate, tune_thresholds


class EarlyStopping:
    def __init__(self, patience: int, min_delta: float = 1e-4):
        self.patience   = patience
        self.min_delta  = min_delta
        self.best_score = -np.inf
        self.counter    = 0
        self.best_state = None

    def __call__(self, score: float, model: nn.Module) -> bool:
        """Returns True if training should stop."""
        if score > self.best_score + self.min_delta:
            self.best_score = score
            self.counter    = 0
            self.best_state = copy.deepcopy(model.state_dict())
            return False
        else:
            self.counter += 1
            return self.counter >= self.patience

    def restore_best(self, model: nn.Module):
        if self.best_state:
            model.load_state_dict(self.best_state)


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        loss_fn: MaskedSoftBCE,
        train_loader: DataLoader,
        val_loader:   DataLoader,
        cfg: TrainConfig,
        model_name: str = "model",
    ):
        self.model        = model.to(cfg.device)
        self.loss_fn      = loss_fn
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.cfg          = cfg
        self.model_name   = model_name
        self.log: List[Dict] = []

        self.scaler = GradScaler(enabled=cfg.fp16)

    # ─────────────────────────────────────────────────────────────────
    # Public entry point
    # ─────────────────────────────────────────────────────────────────

    def fit(self) -> nn.Module:
        """Run all three training phases. Returns best model."""

        print(f"\n{'='*60}")
        print(f"  Training: {self.model_name}")
        print(f"  Device:   {self.cfg.device}   FP16: {self.cfg.fp16}")
        print(f"{'='*60}")

        es = EarlyStopping(self.cfg.patience, self.cfg.min_delta)

        # ── Phase 1: head warmup ─────────────────────────────────────
        self.model.freeze_backbone()
        self._run_phase(
            phase=1,
            lr=self.cfg.lr_phase1,
            n_epochs=self.cfg.num_epochs_phase1,
            early_stopper=None,      # no early stopping in warmup
        )

        # ── Phase 2: top layers ──────────────────────────────────────
        self.model.unfreeze_top_layers(n=4)
        stopped = self._run_phase(
            phase=2,
            lr=self.cfg.lr_phase2,
            n_epochs=self.cfg.num_epochs_phase2,
            early_stopper=es,
        )

        # ── Phase 3: full fine-tune (only if still improving) ────────
        if not stopped:
            print("\n[Phase 3] Val F1 still improving — running full fine-tune")
            self.model.unfreeze_all()
            self._run_phase(
                phase=3,
                lr=self.cfg.lr_phase3,
                n_epochs=self.cfg.num_epochs_phase3,
                early_stopper=es,
            )

        es.restore_best(self.model)
        print(f"\n[trainer] Best val macro-F1: {es.best_score:.4f}")
        self._save_checkpoint("best")
        return self.model

    # ─────────────────────────────────────────────────────────────────
    # Phase runner
    # ─────────────────────────────────────────────────────────────────

    def _run_phase(
        self,
        phase: int,
        lr: float,
        n_epochs: int,
        early_stopper: Optional[EarlyStopping],
    ) -> bool:
        """Returns True if early stopping fired."""
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=lr,
            weight_decay=self.cfg.weight_decay,
        )
        total_steps  = n_epochs * len(self.train_loader)
        warmup_steps = int(total_steps * self.cfg.warmup_ratio)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, warmup_steps, total_steps
        )

        print(f"\n── Phase {phase} │ lr={lr} │ epochs={n_epochs} "
              f"│ steps={total_steps} ──")

        for epoch in range(1, n_epochs + 1):
            t0        = time.time()
            train_loss = self._train_epoch(optimizer, scheduler)
            val_loss, val_f1 = self._val_epoch()

            elapsed = time.time() - t0
            self.log.append({
                "phase": phase, "epoch": epoch,
                "train_loss": train_loss, "val_loss": val_loss,
                "val_macro_f1": val_f1,
            })

            print(f"  Ph{phase} Ep{epoch:02d}/{n_epochs:02d}  "
                  f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
                  f"val_macro_F1={val_f1:.4f}  [{elapsed:.0f}s]")

            if early_stopper is not None:
                if early_stopper(val_f1, self.model):
                    print(f"  [early stop] No improvement for {self.cfg.patience} epochs")
                    return True

        return False

    # ─────────────────────────────────────────────────────────────────
    # Train / val steps
    # ─────────────────────────────────────────────────────────────────

    def _train_epoch(self, optimizer, scheduler) -> float:
        self.model.train()
        total_loss, n_steps = 0.0, 0

        for step, batch in enumerate(self.train_loader):
            input_ids      = batch["input_ids"].to(self.cfg.device)
            attention_mask = batch["attention_mask"].to(self.cfg.device)
            labels         = batch["labels"].to(self.cfg.device)
            mask           = batch["mask"].to(self.cfg.device)
            token_type_ids = batch.get("token_type_ids")
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(self.cfg.device)


            optimizer.zero_grad()

            with autocast(device_type='cuda', enabled=self.cfg.fp16):
                logits = self.model(input_ids, attention_mask)
                loss   = self.loss_fn(logits, labels, mask)

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.cfg.grad_clip
            )
            self.scaler.step(optimizer)
            self.scaler.update()
            scheduler.step()

            total_loss += loss.item()
            n_steps    += 1

            if (step + 1) % self.cfg.log_every_n_steps == 0:
                print(f"    step {step+1}/{len(self.train_loader)}  "
                      f"loss={total_loss/n_steps:.4f}  "
                      f"lr={scheduler.get_last_lr()[0]:.2e}")

        return total_loss / max(n_steps, 1)

    @torch.no_grad()
    def _val_epoch(self) -> tuple:
        self.model.eval()
        total_loss, n_steps = 0.0, 0
        all_probs, all_targets = [], []

        for batch in self.val_loader:
            input_ids      = batch["input_ids"].to(self.cfg.device)
            attention_mask = batch["attention_mask"].to(self.cfg.device)
            labels         = batch["labels"].to(self.cfg.device)
            mask           = batch["mask"].to(self.cfg.device)
            token_type_ids = batch.get("token_type_ids")
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(self.cfg.device)

            with autocast(device_type='cuda', enabled=self.cfg.fp16):
                logits = self.model(input_ids, attention_mask)
                loss   = self.loss_fn(logits, labels, mask)

            total_loss += loss.item()
            n_steps    += 1

            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.append(probs)
            all_targets.append(labels.cpu().numpy())

        probs   = np.concatenate(all_probs)
        targets = np.concatenate(all_targets)

        from sklearn.metrics import f1_score
        hard_targets = (targets >= 0.5).astype(int)
        preds        = (probs >= 0.5).astype(int)
        macro_f1     = f1_score(hard_targets, preds, average="macro", zero_division=0)

        return total_loss / max(n_steps, 1), macro_f1

    # ─────────────────────────────────────────────────────────────────
    # Checkpoint
    # ─────────────────────────────────────────────────────────────────

    def _save_checkpoint(self, tag: str):
        os.makedirs(self.cfg.output_dir, exist_ok=True)
        path = os.path.join(
            self.cfg.output_dir, f"{self.model_name}_{tag}.pt"
        )
        torch.save(self.model.state_dict(), path)
        print(f"[trainer] Saved checkpoint → {path}")
