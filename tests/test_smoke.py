"""Smoke tests: the pipeline runs and the pieces behave.

Tests that exercise YOUR Naive Bayes skip themselves until you implement it, then start
enforcing correctness. Run them often: `make test`.
"""
import numpy as np
import pytest
from scipy import sparse

from src.data import load_split, read_corpus, deduplicate
from src.evaluate import classification_metrics
from src.features import fit_transform_split
from src.naive_bayes import NaiveBayesReference, NaiveBayesScratch

CFG = {"seed": 0,
       "data": {"source": "sample", "deduplicate": True, "train_frac": 0.5},
       "features": {"kind": "counts", "ngram_range": [1, 1], "min_df": 1}}

# Tiny hand-checkable problem: 2 words, class 1 = spam.
X = np.array([[2, 0], [1, 1], [0, 2], [0, 1]], dtype=float)
y = np.array([1, 1, 0, 0])


# ---------------------------------------------------------------- provided --
def test_corpus_loads_and_has_both_classes():
    texts, labels = read_corpus(CFG)
    assert len(texts) == len(labels) > 20
    assert set(np.unique(labels)) == {0, 1}


def test_deduplicate_keeps_first_occurrence():
    t, l, removed = deduplicate(["a", "b", "a"], np.array([0, 1, 0]))
    assert t == ["a", "b"] and removed == 1


@pytest.mark.parametrize("protocol", ["paper", "stratified"])
def test_splits_are_disjoint_and_leak_free_after_dedup(protocol):
    tr, ytr, te, yte, info = load_split(CFG, protocol)
    assert len(tr) + len(te) == info["n_messages_raw"] - info["duplicates_removed"]
    assert info["test_messages_also_in_train"] == 0
    assert 0 in set(ytr) and 1 in set(ytr)


def test_vectorizer_is_fit_on_train_only():
    tr, _, te, _, _ = load_split(CFG, "paper")
    Xtr, Xte, vec = fit_transform_split(CFG, tr, te)
    assert Xtr.shape[1] == Xte.shape[1] == len(vec.vocabulary_)
    # a word that appears only in the test text must NOT be in the vocabulary
    only_test = {w for t in te for w in vec.build_analyzer()(t)} - {w for t in tr for w in vec.build_analyzer()(t)}
    assert not (only_test & set(vec.vocabulary_))


def test_metrics_on_known_confusion():
    m = classification_metrics(np.array([1, 1, 0, 0, 0]), np.array([1, 0, 0, 0, 1]))
    assert (m["tp"], m["fp"], m["fn"], m["tn"]) == (1, 1, 1, 2)
    assert np.isclose(m["spam_caught_pct"], 50.0) and np.isclose(m["blocked_ham_pct"], 100 / 3)


def test_reference_naive_bayes_separates_obvious_classes():
    nb = NaiveBayesReference(alpha=1.0).fit(X, y)
    assert list(nb.predict(np.array([[3.0, 0.0], [0.0, 3.0]]))) == [1, 0]


# ------------------------------------------------------- your implementation --
def _scratch_or_skip(alpha=1.0, data=None):
    nb = NaiveBayesScratch(alpha=alpha)
    try:
        nb.fit(*(data if data is not None else (X, y)))
    except NotImplementedError as e:
        pytest.skip(f"not implemented yet: {e}")
    return nb


def test_scratch_learns_the_hand_computed_parameters():
    """alpha=1. Spam word counts: w0=3, w1=1 (total 4) -> theta = (3+1)/(4+2), (1+1)/(4+2) = 2/3, 1/3.
    Ham word counts: w0=0, w1=3 (total 3) -> theta = (0+1)/(3+2), (3+1)/(3+2) = 1/5, 4/5."""
    nb = _scratch_or_skip()
    assert list(nb.classes_) == [0, 1]
    assert np.allclose(np.exp(nb.class_log_prior_), [0.5, 0.5])
    assert np.allclose(np.exp(nb.feature_log_prob_[1]), [2 / 3, 1 / 3])
    assert np.allclose(np.exp(nb.feature_log_prob_[0]), [1 / 5, 4 / 5])


def test_scratch_posteriors_are_normalized_log_probabilities():
    nb = _scratch_or_skip()
    lp = nb.predict_log_proba(np.array([[1.0, 0.0], [0.0, 2.0], [1.0, 1.0]]))
    assert lp.shape == (3, 2)
    assert np.allclose(np.exp(lp).sum(axis=1), 1.0)
    assert (lp <= 1e-12).all()


def test_scratch_predict_matches_argmax_of_posterior():
    nb = _scratch_or_skip()
    Xq = np.array([[1.0, 0.0], [0.0, 2.0], [2.0, 0.0]])
    assert np.array_equal(nb.predict(Xq), nb.classes_[np.argmax(nb.predict_log_proba(Xq), axis=1)])


def test_scratch_accepts_sparse_input():
    nb = _scratch_or_skip(data=(sparse.csr_matrix(X), y))
    dense = _scratch_or_skip()
    assert np.allclose(nb.feature_log_prob_, dense.feature_log_prob_)
    assert np.allclose(nb.predict_log_proba(sparse.csr_matrix(X)), dense.predict_log_proba(X))


def test_scratch_matches_the_reference_implementation():
    tr, ytr, te, _, _ = load_split(CFG, "stratified")
    Xtr, Xte, _ = fit_transform_split(CFG, tr, te)
    nb = _scratch_or_skip(data=(Xtr, ytr))
    ref = NaiveBayesReference(alpha=1.0).fit(Xtr, ytr)
    assert np.allclose(nb.feature_log_prob_, ref.feature_log_prob_, atol=1e-8)
    assert np.allclose(nb.predict_log_proba(Xte), ref.predict_log_proba(Xte), atol=1e-8)
    assert np.array_equal(nb.predict(Xte), ref.predict(Xte))


def test_scratch_log_space_does_not_underflow():
    """Thousands of features: a product of probabilities would underflow to 0; log space must not."""
    rng = np.random.RandomState(0)
    Xb = sparse.csr_matrix((rng.rand(40, 5000) < 0.02).astype(float))
    yb = np.array([0, 1] * 20)
    nb = _scratch_or_skip(data=(Xb, yb))
    lp = nb.predict_log_proba(Xb)
    assert np.isfinite(lp).all()


def test_scratch_smoothing_avoids_zero_probability():
    nb = _scratch_or_skip(alpha=1.0)
    assert np.isfinite(nb.feature_log_prob_).all()      # a word unseen in a class is not impossible


# ----------------------------------------------- exploratory: Enron-Spam --
ENRON = "tests/fixtures/enron_sample"


def test_enron_loader_reads_ham_and_spam_in_arrival_order():
    from src.data import read_enron
    texts, labels = read_enron(ENRON)
    assert len(texts) == 8 and set(labels) == {0, 1}
    assert labels[:4].tolist() == [0, 0, 0, 0]       # ham fixtures are dated earlier than spam


def test_cross_corpus_exploration_runs_both_directions():
    from src.run import run_cross_corpus
    cfg = {"seed": 0, "data": {"source": "sample", "deduplicate": True, "enron_path": ENRON},
           "features": {"kind": "tfidf", "ngram_range": [1, 1], "min_df": 1},
           "naive_bayes": {"implementation": "reference", "alpha": 1.0}, "svm": {"C": 1.0}}
    res = run_cross_corpus(cfg, "results")
    assert set(res["directions"]) == {"sms_to_enron", "enron_to_sms"}
    for d in res["directions"].values():
        assert 0.0 <= d["naive_bayes"]["f1"] <= 1.0 and "mcc" in d["linear_svm"]
