"""Tests for the model registry.

No torch needed: the BiLSTM and transformer constructors import it lazily.
"""

import pytest

from emotion_classification.errors import ConfigError
from emotion_classification.models import registry
from emotion_classification.models.registry import (
    ModelSpec, build_model, get_model_spec,
)

pytest.importorskip("sklearn")


def test_registry_lists_the_study_models():
    assert registry.model_names() == [
        "bilstm", "hf_bert-base-uncased", "hf_distilbert-base-uncased",
        "linsvm", "logreg", "naive_bayes", "random_forest",
    ]


def test_routes_and_stochastic_flags():
    assert get_model_spec("logreg").route == "S"
    assert get_model_spec("bilstm").route == "S"
    assert get_model_spec("hf_distilbert-base-uncased").route == "C"
    assert get_model_spec("hf_bert-base-uncased").route == "ceiling"
    deterministic = {n for n in registry.model_names() if not get_model_spec(n).stochastic}
    assert deterministic == {"naive_bayes", "logreg", "linsvm"}


def test_classical_models_get_the_feature_backend():
    assert build_model("naive_bayes", features="bow").name == "naive_bayes_bow"
    assert build_model("logreg").name == "logreg_tfidf"


def test_extra_params_reach_the_constructor():
    model = build_model("random_forest", n_estimators=7)
    assert model.head.estimator.n_estimators == 7


def test_aliases_resolve_to_distilbert():
    assert get_model_spec("hf_distilbert").name == "hf_distilbert-base-uncased"
    assert get_model_spec("transformer").name == "hf_distilbert-base-uncased"
    assert build_model("hf_distilbert").model_name == "distilbert-base-uncased"


def test_any_hf_checkpoint_resolves_dynamically():
    spec = get_model_spec("hf_google/electra-small-discriminator")
    assert spec.route == "unassigned" and spec.stochastic and spec.takes_epochs
    model = build_model("hf_google/electra-small-discriminator", epochs=4)
    assert model.model_name == "google/electra-small-discriminator"
    assert model.name == "hf_electra-small-discriminator"
    assert model.epochs == 4


def test_epochs_override_only_where_accepted():
    assert build_model("bilstm", epochs=2).epochs == 2
    assert build_model("bilstm").epochs == 5                        # constructor default
    assert build_model("logreg", epochs=9).name == "logreg_tfidf"   # ignored by classical


def test_unknown_model_rejected():
    with pytest.raises(ConfigError, match="unknown model"):
        get_model_spec("gpt")
    with pytest.raises(ConfigError, match="unknown model"):
        get_model_spec("hf_")


def test_spec_rejects_unknown_route():
    with pytest.raises(ValueError, match="route"):
        ModelSpec("x", lambda: None, route="Z", stochastic=False)
