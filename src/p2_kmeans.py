"""K-means clustering.

This module holds TWO implementations:

1. ``KMeansReference`` — a thin wrapper around scikit-learn's KMeans. It works
   out of the box so the pipeline runs end-to-end before you write any code,
   and it gives you a **correctness oracle** to check your own results against.

2. ``KMeansScratch`` — YOUR from-scratch implementation. The methods below are
   stubs. Filling them in is the core of Project 2.

Per the handout: the library version is for *checking* only. Your submitted run
must use ``kmeans.implementation: scratch`` in config.yaml.
"""
from __future__ import annotations

import numpy as np

from .distances import euclidean_sqdist, mahalanobis_sqdist


# --------------------------------------------------------------------------
# Provided baseline — do not submit results from this one.
# --------------------------------------------------------------------------
class KMeansReference:
    """scikit-learn KMeans, wrapped in the same interface as KMeansScratch."""

    def __init__(self, k=3, init="kmeans++", n_init=10, max_iter=300, tol=1e-4,
                 seed=42, **_ignored):
        self.k, self.n_init, self.max_iter, self.tol, self.seed = k, n_init, max_iter, tol, seed
        self.init = "k-means++" if init in ("kmeans++", "k-means++") else "random"
        self.centroids_ = None
        self.labels_ = None
        self.inertia_ = None
        self.n_iter_ = None

    def fit(self, X: np.ndarray) -> "KMeansReference":
        from sklearn.cluster import KMeans

        km = KMeans(n_clusters=self.k, init=self.init, n_init=self.n_init,
                    max_iter=self.max_iter, tol=self.tol, random_state=self.seed)
        km.fit(X)
        self.centroids_ = km.cluster_centers_
        self.labels_ = km.labels_.astype("int64")
        self.inertia_ = float(km.inertia_)
        self.n_iter_ = int(km.n_iter_)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return euclidean_sqdist(X, self.centroids_).argmin(axis=1).astype("int64")


# --------------------------------------------------------------------------
# YOUR implementation
# --------------------------------------------------------------------------
class KMeansScratch:
    """K-means (Lloyd's algorithm), implemented from scratch.

    Attributes set by ``fit``:
        centroids_ : (k, d) final centroids
        labels_    : (n,)   cluster index of each point
        inertia_   : float  within-cluster sum of squared distances (the objective J)
        n_iter_    : int    iterations actually run

    Supports two distance metrics, selected at construction:
        "euclidean"   — plain squared Euclidean
        "mahalanobis" — simplified diagonal Mahalanobis (Task 4)
    """

    def __init__(self, k=3, init="kmeans++", n_init=10, max_iter=300, tol=1e-4,
                 seed=42, distance="euclidean", mahalanobis_diag=None):
        self.k = k
        self.init = init
        self.n_init = n_init
        self.max_iter = max_iter
        self.tol = tol
        self.seed = seed
        self.distance = distance
        self.mahalanobis_diag = mahalanobis_diag
        self.centroids_ = None
        self.labels_ = None
        self.inertia_ = None
        self.n_iter_ = None

    # -- distance dispatch (provided) --------------------------------------
    def _sqdist(self, X: np.ndarray, C: np.ndarray) -> np.ndarray:
        """(n, k) squared distances under the configured metric."""
        if self.distance == "euclidean":
            return euclidean_sqdist(X, C)
        if self.distance == "mahalanobis":
            diag = self.mahalanobis_diag
            diag = np.ones(X.shape[1]) if diag is None else np.asarray(diag, dtype="float64")
            if diag.shape != (X.shape[1],):
                raise ValueError(
                    f"mahalanobis_diag must have length {X.shape[1]}, got {diag.shape}"
                )
            if np.any(diag <= 0):
                raise ValueError("mahalanobis_diag entries must be strictly positive")
            return mahalanobis_sqdist(X, C, diag)
        raise ValueError(f"unknown distance {self.distance!r}")

    # -- the three steps you implement -------------------------------------
    def _init_centroids(self, X: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Choose ``self.k`` initial centroids and return them as a (k, d) array.

        Support both settings of ``self.init``:
          "random"   — pick k distinct rows of X uniformly at random.
          "kmeans++" — pick the first centroid at random, then pick each next
                       centroid with probability proportional to its squared
                       distance to the NEAREST already-chosen centroid. This
                       spreads centroids out and avoids bad local optima.

        Use ``rng`` (a numpy Generator) for every random draw so runs are
        reproducible. ``self._sqdist`` gives you the distances you need.

        """
        if self.init == "random":
            return rng.choice(X,size=self.k,replace=False)
        elif self.init == "kmeans++" or self.init == "k-means++":
            centroids = np.empty((self.k,X.shape[1]))
            centroids[0] = rng.choice(X)
            for c in range(1, self.k):
                dist = self._sqdist(X,centroids[:c])
                D = dist.min(axis=1)
                total = D.sum()
                if total == 0:
                    centroids[c] = rng.choice(X)
                    continue
                probs = D/total
                centroids[c] = rng.choice(X,p=probs)
            return centroids     
        else:
            raise ValueError (
                "Unknown init value"
            )
    def _assign(self, X: np.ndarray, centroids: np.ndarray) -> np.ndarray:
        """Assignment step: index of the nearest centroid for each point.

        Returns an int64 array of shape (n,) with values in [0, k).

        """
        dist = self._sqdist(X,centroids)
        result = []
        for row in dist:
            result.append(np.argmin(row))
        result = np.array(result, dtype=np.int64 )
        return result

    def _update(self, X: np.ndarray, labels: np.ndarray, centroids: np.ndarray) -> np.ndarray:
        """Update step: move each centroid to the mean of its assigned points.

        Watch out for **empty clusters** — a centroid with no points assigned has
        no mean. A common fix is to leave it where it is, or re-seed it at the
        point furthest from its centroid. Say which you chose in your report.

        Returns the new (k, d) centroids.

        """
        newcentroids = np.copy(centroids)
        for i in range(self.k):
            mask = labels == i
            points = X[mask]
            if len(points) > 0:
                newcentroids[i] = points.mean(axis=0)
        return newcentroids

    # -- the fit loop you implement ----------------------------------------
    def fit(self, X: np.ndarray) -> "KMeansScratch":
        """Run Lloyd's algorithm with ``self.n_init`` random restarts.

        For each restart:
            1. initialize centroids            (``self._init_centroids``)
            2. repeat up to ``self.max_iter``:
                 a. labels = assignment step   (``self._assign``)
                 b. new_centroids = update     (``self._update``)
                 c. stop when the centroids move less than ``self.tol``
                    (or when the labels stop changing)
            3. compute inertia J = sum of squared distances of each point to
               its assigned centroid

        Keep the restart with the LOWEST inertia, and set:
            self.centroids_, self.labels_, self.inertia_, self.n_iter_

        Return ``self``.
        """
        rng=np.random.default_rng(self.seed)
        b_inertia = np.inf
        b_centroids = None
        b_labels = None
        b_n_iter = 0
        for run in range(self.n_init):
            centroids=self._init_centroids(X,rng)
            for i in range(self.max_iter):
                labels = self._assign(X,centroids)
                new_centroids = self._update(X,labels,centroids)
                shift = np.sum((new_centroids-centroids)**2)
                centroids = new_centroids
                if shift < self.tol:
                    break

            n_iter = i + 1
            labels = self._assign(X,centroids)
            inertia = self._sqdist(X, centroids).min(axis=1).sum()

            if inertia < b_inertia:
                b_inertia = inertia
                b_centroids = centroids
                b_labels = labels
                b_n_iter = n_iter
        
        self.centroids_ = b_centroids
        self.labels_ = b_labels
        self.inertia_ = float(b_inertia)
        self.n_iter_ = b_n_iter
        return self
        

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Assign new points to the fitted centroids."""
        if self.centroids_ is None:
            raise RuntimeError("call fit() before predict()")
        return self._assign(X, self.centroids_)


def build(cfg: dict):
    """Construct the clusterer selected by config.yaml."""
    kc = cfg["kmeans"]
    impl = kc.get("implementation", "sklearn")
    common = dict(k=kc["k"], init=kc.get("init", "kmeans++"),
                  n_init=kc.get("n_init", 10), max_iter=kc.get("max_iter", 300),
                  tol=float(kc.get("tol", 1e-4)), seed=cfg["seed"])
    if impl == "sklearn":
        if kc.get("distance", "euclidean") != "euclidean":
            raise ValueError("The sklearn baseline supports Euclidean only; use scratch for Mahalanobis.")
        return KMeansReference(**common)
    if impl == "scratch":
        return KMeansScratch(distance=kc.get("distance", "euclidean"),
                             mahalanobis_diag=kc.get("mahalanobis_diag"), **common)
    raise ValueError(f"unknown kmeans.implementation {impl!r}")