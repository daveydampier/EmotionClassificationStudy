"""Deep-learning tier: a self-contained BiLSTM classifier (PyTorch).

No pretrained embeddings or external tokenizer required — a whitespace
vocabulary is built from the training texts, embeddings are learned from
scratch, and a bidirectional LSTM feeds a linear multi-label head trained with
``BCEWithLogitsLoss``. This keeps the tier runnable on CPU for the study while
still exercising the PyTorch path end-to-end.

``torch`` is imported lazily inside methods so importing this module (and the
package) does not require torch to be installed.
"""

from __future__ import annotations

import io
import re

import numpy as np

from .base import BaseEmotionModel

_TOKEN_RE = re.compile(r"[a-z0-9']+")
PAD, UNK = "<pad>", "<unk>"


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


class BiLSTMClassifier(BaseEmotionModel):
    """Learned-embedding BiLSTM for multi-label emotion classification."""

    name = "bilstm"

    def __init__(
        self,
        *,
        embed_dim: int = 128,
        hidden_dim: int = 128,
        max_vocab: int = 30_000,
        max_len: int = 64,
        epochs: int = 5,
        batch_size: int = 64,
        lr: float = 1e-3,
        device: str | None = None,
    ):
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.max_vocab = max_vocab
        self.max_len = max_len
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self._device = device
        self.vocab_: dict[str, int] = {}
        self.module_ = None

    # -- vocab / encoding ----------------------------------------------------
    def _build_vocab(self, texts: list[str]) -> None:
        from collections import Counter

        counts: Counter[str] = Counter()
        for t in texts:
            counts.update(_tokenize(t))
        vocab = {PAD: 0, UNK: 1}
        for tok, _ in counts.most_common(self.max_vocab - 2):
            vocab[tok] = len(vocab)
        self.vocab_ = vocab

    def _encode(self, texts: list[str]):
        import torch

        unk = self.vocab_[UNK]
        rows = []
        for t in texts:
            ids = [self.vocab_.get(tok, unk) for tok in _tokenize(t)][: self.max_len]
            if not ids:
                ids = [self.vocab_[PAD]]
            rows.append(ids)
        lengths = torch.tensor([len(r) for r in rows], dtype=torch.long)
        width = int(lengths.max())
        padded = torch.zeros((len(rows), width), dtype=torch.long)
        for i, r in enumerate(rows):
            padded[i, : len(r)] = torch.tensor(r, dtype=torch.long)
        return padded, lengths

    def _resolve_device(self):
        import torch

        if self._device:
            return torch.device(self._device)
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -- API -----------------------------------------------------------------
    def fit(self, texts: list[str], Y: np.ndarray):
        import torch
        from torch import nn

        Y = np.asarray(Y, dtype=np.float32)
        self._build_vocab(texts)
        device = self._resolve_device()
        self.module_ = _BiLSTMModule(
            vocab_size=len(self.vocab_),
            embed_dim=self.embed_dim,
            hidden_dim=self.hidden_dim,
            n_labels=Y.shape[1],
            pad_idx=self.vocab_[PAD],
        ).to(device)

        optimizer = torch.optim.Adam(self.module_.parameters(), lr=self.lr)
        loss_fn = nn.BCEWithLogitsLoss()
        X, lengths = self._encode(texts)
        targets = torch.tensor(Y)
        n = len(texts)

        self.module_.train()
        for _ in range(self.epochs):
            perm = torch.randperm(n)
            for start in range(0, n, self.batch_size):
                idx = perm[start : start + self.batch_size]
                xb = X[idx].to(device)
                lb = lengths[idx]
                yb = targets[idx].to(device)
                optimizer.zero_grad()
                logits = self.module_(xb, lb)
                loss = loss_fn(logits, yb)
                loss.backward()
                optimizer.step()
        return self

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        import torch

        if self.module_ is None:
            raise RuntimeError("call fit() before predict_proba()")
        device = self._resolve_device()
        X, lengths = self._encode(texts)
        self.module_.eval()
        out = []
        with torch.no_grad():
            for start in range(0, len(texts), self.batch_size):
                xb = X[start : start + self.batch_size].to(device)
                lb = lengths[start : start + self.batch_size]
                logits = self.module_(xb, lb)
                out.append(torch.sigmoid(logits).cpu().numpy())
        return np.vstack(out) if out else np.zeros((0, 0))

    def size_mb(self) -> float:
        import torch

        if self.module_ is None:
            return 0.0
        buf = io.BytesIO()
        torch.save(self.module_.state_dict(), buf)
        return buf.tell() / (1024 * 1024)


try:  # Define the nn.Module only when torch is available.
    import torch
    from torch import nn

    class _BiLSTMModule(nn.Module):
        def __init__(self, vocab_size, embed_dim, hidden_dim, n_labels, pad_idx):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
            self.lstm = nn.LSTM(
                embed_dim, hidden_dim, batch_first=True, bidirectional=True
            )
            self.dropout = nn.Dropout(0.3)
            self.fc = nn.Linear(hidden_dim * 2, n_labels)

        def forward(self, x, lengths):
            emb = self.embedding(x)
            packed = nn.utils.rnn.pack_padded_sequence(
                emb, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            _, (h, _) = self.lstm(packed)
            # concat final forward + backward hidden states
            h = torch.cat((h[-2], h[-1]), dim=1)
            return self.fc(self.dropout(h))

except ImportError:  # pragma: no cover - torch not installed
    _BiLSTMModule = None  # type: ignore[assignment]
