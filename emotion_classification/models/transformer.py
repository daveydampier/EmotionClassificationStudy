"""Transformer tier: fine-tune a pretrained encoder (HuggingFace + PyTorch).

Wraps ``AutoModelForSequenceClassification`` in multi-label mode (sigmoid +
``BCEWithLogitsLoss``) behind the common :class:`EmotionModel` interface, so a
fine-tuned BERT/RoBERTa lands on the same scorecard as the TF-IDF baselines.
Defaults to ``distilbert-base-uncased`` — small enough to fine-tune on CPU for a
smoke run, swap ``model_name`` for ``bert-base-uncased`` / ``roberta-base`` for
the real comparison.

``torch`` and ``transformers`` are imported lazily inside methods.
"""

from __future__ import annotations

import io

import numpy as np

from .base import BaseEmotionModel


class TransformerClassifier(BaseEmotionModel):
    """Fine-tuned transformer encoder for multi-label emotion classification."""

    def __init__(
        self,
        model_name: str = "distilbert-base-uncased",
        *,
        max_len: int = 128,
        epochs: int = 3,
        batch_size: int = 16,
        lr: float = 2e-5,
        device: str | None = None,
    ):
        self.model_name = model_name
        self.name = f"hf_{model_name.split('/')[-1]}"
        self.max_len = max_len
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self._device = device
        self.tokenizer_ = None
        self.model_ = None

    def _resolve_device(self):
        import torch

        if self._device:
            return torch.device(self._device)
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _encode(self, texts: list[str]):
        enc = self.tokenizer_(
            texts,
            truncation=True,
            padding=True,
            max_length=self.max_len,
            return_tensors="pt",
        )
        return enc

    def fit(self, texts: list[str], Y: np.ndarray):
        import torch
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
        )

        Y = np.asarray(Y, dtype=np.float32)
        n_labels = Y.shape[1]
        device = self._resolve_device()

        self.tokenizer_ = AutoTokenizer.from_pretrained(self.model_name)
        self.model_ = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=n_labels,
            problem_type="multi_label_classification",
        ).to(device)

        optimizer = torch.optim.AdamW(self.model_.parameters(), lr=self.lr)
        targets = torch.tensor(Y)
        n = len(texts)

        self.model_.train()
        for _ in range(self.epochs):
            perm = torch.randperm(n)
            for start in range(0, n, self.batch_size):
                idx = perm[start : start + self.batch_size].tolist()
                batch_texts = [texts[i] for i in idx]
                enc = self._encode(batch_texts).to(device)
                labels = targets[idx].to(device)
                optimizer.zero_grad()
                out = self.model_(**enc, labels=labels)
                out.loss.backward()
                optimizer.step()
        return self

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        import torch

        if self.model_ is None:
            raise RuntimeError("call fit() before predict_proba()")
        device = self._resolve_device()
        self.model_.eval()
        out = []
        with torch.no_grad():
            for start in range(0, len(texts), self.batch_size):
                batch_texts = texts[start : start + self.batch_size]
                enc = self._encode(batch_texts).to(device)
                logits = self.model_(**enc).logits
                out.append(torch.sigmoid(logits).cpu().numpy())
        return np.vstack(out) if out else np.zeros((0, 0))

    def size_mb(self) -> float:
        import torch

        if self.model_ is None:
            return 0.0
        buf = io.BytesIO()
        torch.save(self.model_.state_dict(), buf)
        return buf.tell() / (1024 * 1024)
