"""Text -> numeric vectors. PROVIDED, but you are expected to explore the options in config.yaml
and to document your preprocessing choices in the report.

The vectorizer is FITTED ON TRAINING TEXT ONLY and then used to transform the test text. Fitting
the vocabulary or the IDF weights on the test set is data leakage (see the course notes on evaluation and text features).
"""
from __future__ import annotations

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


def build_vectorizer(cfg):
    f = cfg["features"]
    common = dict(lowercase=f.get("lowercase", True),
                  ngram_range=tuple(f.get("ngram_range", (1, 1))),
                  min_df=f.get("min_df", 1),
                  max_features=f.get("max_features"))
    if f.get("kind", "tfidf") == "counts":
        return CountVectorizer(**common)
    return TfidfVectorizer(sublinear_tf=f.get("sublinear_tf", False), **common)


def fit_transform_split(cfg, train_texts, test_texts):
    """Fit on train only; return (X_train, X_test, vectorizer)."""
    vec = build_vectorizer(cfg)
    X_tr = vec.fit_transform(train_texts)
    X_te = vec.transform(test_texts)          # transform only, never fit, on test text
    return X_tr, X_te, vec
