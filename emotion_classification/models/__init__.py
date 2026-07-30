"""Model tiers for the study, all behind one interface.

Every model implements :class:`~emotion_classification.models.base.EmotionModel`
(``fit`` / ``predict_proba`` / ``size_mb``), so the experiment runner and
scorecard treat a TF-IDF SVM and a fine-tuned transformer identically.

Concrete models live in submodules and import their heavy dependencies (torch,
transformers) lazily, so importing this package stays cheap:

* :mod:`.classical`   — scikit-learn: NB / LogReg / LinearSVM / RandomForest (TF-IDF, BoW)
* :mod:`.deep`        — PyTorch BiLSTM
* :mod:`.transformer` — HuggingFace BERT / DistilBERT fine-tuning
"""

from .base import EmotionModel

__all__ = ["EmotionModel"]
