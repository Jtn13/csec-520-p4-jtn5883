"""Classifiers other than Naive Bayes, plus the exploratory clustering. PROVIDED.

The linear SVM and KNN use scikit-learn: only the Naive Bayes must be your own code.
"""
from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC

from . import evaluate as ev


def build_svm(cfg):
    return LinearSVC(C=float(cfg["svm"]["C"]), random_state=cfg["seed"])


def build_knn(cfg, k):
    # cosine distance needs brute force on sparse input
    return KNeighborsClassifier(n_neighbors=int(k), metric=cfg["knn"].get("metric", "cosine"),
                                algorithm="brute")


def spam_scores_from_log_proba(log_proba, classes):
    """Spam log-odds from an (n, 2) log-probability array (monotone in P(spam | x))."""
    classes = list(classes)
    return log_proba[:, classes.index(1)] - log_proba[:, classes.index(0)]


def explore_kmeans(cfg, X_train, y_train):
    """Cluster the TRAINING vectors (after LSI) and score the clusters against the labels."""
    kc = cfg["kmeans"]
    n_comp = max(2, min(int(kc.get("lsi_components", 50)), X_train.shape[1] - 1, X_train.shape[0] - 1))
    Z = TruncatedSVD(n_components=n_comp, random_state=cfg["seed"]).fit_transform(X_train)
    impl = kc.get("implementation", "sklearn")
    if impl == "p2":
        try:
            from .p2_kmeans import KMeansScratch      # your Project 2 code, copied into src/
        except ImportError as e:
            raise ImportError("kmeans.implementation = 'p2' needs src/p2_kmeans.py "
                              "defining KMeansScratch (copy it from Project 2)") from e
        est = KMeansScratch(k=int(kc["k"]), seed=cfg["seed"]).fit(Z)
        clusters = np.asarray(est.labels_)
    else:
        clusters = KMeans(n_clusters=int(kc["k"]), n_init=10, random_state=cfg["seed"]).fit_predict(Z)
    out = {"implementation": impl, "k": int(kc["k"]), "lsi_components": int(n_comp)}
    out.update(ev.internal_cluster_metrics(y_train, clusters))
    out["spam_fraction_per_cluster"] = {int(c): float(y_train[clusters == c].mean())
                                        for c in np.unique(clusters)}
    return out

def tune_svm(cfg, train_texts, y_train):
    """Choose C by stratified k-fold CV on the TRAINING messages only.

    The vectorizer sits inside the pipeline, so every CV fold fits its own vocabulary and IDF
    weights on that fold's training part. The validation fold never leaks into the features.
    After the search, the best pipeline is refit on all training messages.
    """
    from sklearn.model_selection import GridSearchCV, StratifiedKFold
    from sklearn.pipeline import Pipeline
    from .features import build_vectorizer

    s = cfg["svm"]
    grid = [float(c) for c in s.get("C_grid", [s["C"]])]
    pipe = Pipeline([
        ("vec", build_vectorizer(cfg)),
        ("svm", LinearSVC(random_state=cfg["seed"], max_iter=10000)),
    ])
    cv = StratifiedKFold(n_splits=int(s.get("cv_folds", 5)), shuffle=True, random_state=cfg["seed"])
    search = GridSearchCV(pipe, {"svm__C": grid}, cv=cv,
                          scoring=s.get("cv_scoring", "f1"), refit=True)
    search.fit(list(train_texts), y_train)

    info = {
        "C": float(search.best_params_["svm__C"]),
        "cv_scoring": s.get("cv_scoring", "f1"),
        "cv_best_score": float(search.best_score_),
        "cv_scores_by_C": {str(c): float(m) for c, m in
                           zip(grid, search.cv_results_["mean_test_score"])},
    }
    return search.best_estimator_, info

def oof_spam_scores(train_texts, y_train, make_vec, make_model, score_fn, n_splits, seed):
    """Out-of-fold spam scores for the TRAINING messages.

    Each fold fits its own vectorizer and model on the other folds, so no training message is
    scored by a model that saw it. The test set is not touched here.
    """
    from sklearn.model_selection import StratifiedKFold
    texts = np.asarray(list(train_texts), dtype=object)
    y = np.asarray(y_train)
    scores = np.empty(len(y), dtype=float)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for fit_idx, val_idx in cv.split(texts, y):
        vec = make_vec()
        X_fit = vec.fit_transform(texts[fit_idx])
        X_val = vec.transform(texts[val_idx])
        model = make_model().fit(X_fit, y[fit_idx])
        scores[val_idx] = score_fn(model, X_val)
    return scores


def choose_threshold(scores, y, max_blocked_ham_pct):
    """Threshold that catches the most spam while blocking at most `max_blocked_ham_pct` of ham.

    A message is called spam when score >= threshold. Ties in spam caught go to the higher
    threshold, because a blocked legitimate message costs more than a missed spam.
    """
    scores = np.asarray(scores, dtype=float)
    y = np.asarray(y)
    ham, spam = scores[y == 0], scores[y == 1]
    best = None
    for t in np.unique(scores):
        bh = 100.0 * np.mean(ham >= t)
        if bh > max_blocked_ham_pct:
            continue
        sc = 100.0 * np.mean(spam >= t)
        if best is None or sc > best[1] or (sc == best[1] and t > best[0]):
            best = (float(t), float(sc), float(bh))
    if best is None:                       # cap unreachable: block nothing
        best = (float(scores.max()) + 1.0, 0.0, 0.0)
    return {"threshold": best[0],
            "train_oof_spam_caught_pct": best[1],
            "train_oof_blocked_ham_pct": best[2]}