"""Model registry: every model configuration the study can run, by short name.

One :class:`ModelSpec` per configuration records how to build it, which *route*
to a lightweight model it belongs to, and whether its training is stochastic.
The pipeline and the command line build models through :func:`build_model`, so
adding a model is one :func:`register` call, with no change to the runner.

Routes (thesis spec, section 2.2): ``S`` small by design, ``F`` frozen encoder
plus linear head, ``C`` compressed transformer, ``ceiling`` full-size reference,
``hosted`` no-training hosted model. ``unassigned`` marks a HuggingFace
checkpoint requested ad hoc as ``hf_<checkpoint>`` and not yet classified.

Builders import their model classes lazily, so importing this module needs
neither torch nor transformers.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from ..errors import ConfigError

ROUTES: tuple = ("S", "F", "C", "ceiling", "hosted", "unassigned")
HF_PREFIX = "hf_"


@dataclass(frozen=True)
class ModelSpec:
    """How to build one model configuration, and what kind of model it is."""

    name: str                    # registry key and command-line short name
    builder: Callable            # builder(**params) -> EmotionModel
    route: str                   # one of ROUTES
    stochastic: bool             # does training vary with the seed?
    takes_features: bool = False  # accepts features="tfidf" | "bow"
    takes_epochs: bool = False    # accepts epochs=<int>

    def __post_init__(self) -> None:
        if self.route not in ROUTES:
            raise ValueError(f"unknown route {self.route!r}; choose from {', '.join(ROUTES)}")


def _classical(class_name: str) -> Callable:
    def build(**params):
        from . import classical

        return getattr(classical, class_name)(**params)

    return build


def _bilstm(**params):
    from .deep import BiLSTMClassifier

    return BiLSTMClassifier(**params)


def _transformer(checkpoint: str) -> Callable:
    def build(**params):
        from .transformer import TransformerClassifier

        return TransformerClassifier(model_name=checkpoint, **params)

    return build


MODELS: dict = {}
ALIASES: dict = {
    "hf_distilbert": "hf_distilbert-base-uncased",
    "transformer": "hf_distilbert-base-uncased",
}


def register(spec: ModelSpec) -> ModelSpec:
    """Add ``spec`` to the registry. A name can be registered only once."""
    if spec.name in MODELS:
        raise ValueError(f"model {spec.name!r} is already registered")
    MODELS[spec.name] = spec
    return spec


# The stochastic flags are read off results/paper_final/data/go_native_seeds_summary.csv:
# Naive Bayes, LogReg and LinearSVM show zero spread over five seeds; Random Forest,
# the BiLSTM and the transformers do not.
register(ModelSpec("naive_bayes", _classical("NaiveBayes"), route="S",
                   stochastic=False, takes_features=True))
register(ModelSpec("logreg", _classical("LogisticReg"), route="S",
                   stochastic=False, takes_features=True))
register(ModelSpec("linsvm", _classical("LinearSVM"), route="S",
                   stochastic=False, takes_features=True))
register(ModelSpec("random_forest", _classical("RandomForest"), route="S",
                   stochastic=True, takes_features=True))
register(ModelSpec("bilstm", _bilstm, route="S", stochastic=True, takes_epochs=True))
register(ModelSpec("hf_distilbert-base-uncased", _transformer("distilbert-base-uncased"),
                   route="C", stochastic=True, takes_epochs=True))
register(ModelSpec("hf_bert-base-uncased", _transformer("bert-base-uncased"),
                   route="ceiling", stochastic=True, takes_epochs=True))


def model_names() -> list:
    """Names of every registered model, sorted."""
    return sorted(MODELS)


def get_model_spec(name: str) -> ModelSpec:
    """Look up a model by short name, alias, or ``hf_<checkpoint>``."""
    name = ALIASES.get(name, name)
    if name in MODELS:
        return MODELS[name]
    if name.startswith(HF_PREFIX) and len(name) > len(HF_PREFIX):
        return ModelSpec(name, _transformer(name[len(HF_PREFIX):]), route="unassigned",
                         stochastic=True, takes_epochs=True)
    known = ", ".join(sorted(MODELS))
    raise ConfigError(f"unknown model {name!r}; known models: {known}, or hf_<checkpoint>")


def build_model(name: str, *, features: str = "tfidf", epochs=None, **params):
    """Instantiate a model by short name.

    ``features`` goes to models with a text-feature backend (the classical tier).
    ``epochs``, if set, overrides the training-epoch default of the models that
    train by epochs and is ignored by the rest. ``params`` are passed straight to
    the model's constructor.
    """
    spec = get_model_spec(name)
    kwargs = dict(params)
    if spec.takes_features:
        kwargs["features"] = features
    if spec.takes_epochs and epochs:
        kwargs["epochs"] = epochs
    return spec.builder(**kwargs)
