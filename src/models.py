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
