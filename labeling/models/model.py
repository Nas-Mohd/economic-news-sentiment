"""
models/model.py
───────────────
Two models:
  1. AspectClassifier  — Modern-FinBERT-large with 6-way sigmoid head
  2. BaselineClassifier — classic ProsusAI/finbert re-purposed with 6-way head
                          trained identically for apples-to-apples comparison
"""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
from config import ModelConfig


# ─────────────────────────────────────────────────────────────────────────────
# Shared classification head
# ─────────────────────────────────────────────────────────────────────────────

class MultiLabelHead(nn.Module):
    """Dropout → Linear → (sigmoid applied in loss, not here)"""

    def __init__(self, hidden_size: int, num_labels: int, dropout: float):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_labels)
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

    def forward(self, cls_token: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.dropout(cls_token))


# ─────────────────────────────────────────────────────────────────────────────
# 1. Main model — Modern-FinBERT-large
# ─────────────────────────────────────────────────────────────────────────────

class AspectClassifier(nn.Module):
    """
    Loads Modern-FinBERT-large backbone, discards its 3-class sentiment head,
    attaches a fresh 6-way multi-label head.

    Freezing API:
        model.freeze_backbone()          → Phase 1 (head warmup)
        model.unfreeze_top_layers(n=4)   → Phase 2 (top-n transformer layers)
        model.unfreeze_all()             → Phase 3 (full fine-tune)
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg

        # Load backbone weights only (ignore the sentiment classifier head)
        self.backbone = AutoModel.from_pretrained(cfg.model_name)
        self.backbone = self.backbone.float() 
        hidden_size = self.backbone.config.hidden_size

        self.head = MultiLabelHead(hidden_size, cfg.num_labels, cfg.dropout)

    # ── Forward ──────────────────────────────────────────────────────────────

    def forward(self, input_ids, attention_mask, token_type_ids = None) -> torch.Tensor:
        out = self.backbone(
            input_ids=input_ids, 
            attention_mask=attention_mask
        )
        # Use [CLS] token representation
        cls = out.last_hidden_state[:, 0, :]   # [batch, hidden]
        return self.head(cls)                   # [batch, 6]  — raw logits

    # ── Freezing helpers ─────────────────────────────────────────────────────

    def freeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = False
        print("[freeze] Backbone frozen — training head only")

    def unfreeze_top_layers(self, n: int = 4):
        """Unfreeze the last n transformer encoder layers + embeddings remain frozen."""
        # First re-freeze everything
        for p in self.backbone.parameters():
            p.requires_grad = False

        # ModernBERT uses .layers; classic BERT uses .encoder.layer
        # Try both naming conventions
        layers = None
        if hasattr(self.backbone, "encoder") and hasattr(self.backbone.encoder, "layer"):
            layers = self.backbone.encoder.layer           # BERT-style
        elif hasattr(self.backbone, "layers"):
            layers = self.backbone.layers                  # ModernBERT-style
        elif hasattr(self.backbone, "model") and hasattr(self.backbone.model, "layers"):
            layers = self.backbone.model.layers            # wrapped ModernBERT

        if layers is None:
            print("[freeze] WARNING: could not find transformer layers — unfreezing all")
            self.unfreeze_all()
            return

        total = len(layers)
        for layer in layers[total - n:]:
            for p in layer.parameters():
                p.requires_grad = True

        # Also unfreeze the final layer norm if present
        for name in ["norm", "final_layer_norm", "ln_f"]:
            module = getattr(self.backbone, name, None)
            if module is not None:
                for p in module.parameters():
                    p.requires_grad = True

        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total_p   = sum(p.numel() for p in self.parameters())
        print(f"[freeze] Top {n}/{total} layers unfrozen — "
              f"{trainable:,} / {total_p:,} params trainable "
              f"({100*trainable/total_p:.1f}%)")

    def unfreeze_all(self):
        for p in self.parameters():
            p.requires_grad = True
        trainable = sum(p.numel() for p in self.parameters())
        print(f"[freeze] All params unfrozen — {trainable:,} trainable")


# ─────────────────────────────────────────────────────────────────────────────
# 2. Baseline model — classic ProsusAI/finbert
# ─────────────────────────────────────────────────────────────────────────────

class BaselineClassifier(nn.Module):
    """
    Identical architecture to AspectClassifier but built on ProsusAI/finbert
    (BERT-base sized, 768 hidden). Trained with the same loss, same schedule,
    same thresholds — provides a fair apples-to-apples comparison.

    Note: ProsusAI/finbert also has a 3-class head we discard.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(cfg.baseline_model_name)
        hidden_size = self.backbone.config.hidden_size
        self.head = MultiLabelHead(hidden_size, cfg.num_labels, cfg.dropout)

    def forward(self, input_ids, attention_mask, token_type_ids = None) -> torch.Tensor:
        if input_ids.size(1) > 512:
            input_ids = input_ids[:, :512]
            attention_mask = attention_mask[:, :512]
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0, :]
        return self.head(cls)

    def freeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = False

    def unfreeze_top_layers(self, n: int = 4):
        for p in self.backbone.parameters():
            p.requires_grad = False
        layers = self.backbone.encoder.layer
        total  = len(layers)
        for layer in layers[total - n:]:
            for p in layer.parameters():
                p.requires_grad = True

    def unfreeze_all(self):
        for p in self.parameters():
            p.requires_grad = True
