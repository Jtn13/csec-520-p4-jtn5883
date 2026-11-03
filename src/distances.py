"""Distance functions used by the clustering algorithms.

`euclidean_sqdist` is provided as a worked example of the vectorized NumPy style
we want. `mahalanobis_sqdist` is YOURS to implement (Project 2, Task 4).
"""
from __future__ import annotations

import numpy as np


def euclidean_sqdist(X: np.ndarray, C: np.ndarray) -> np.ndarray:
    """Squared Euclidean distance from every point to every centroid.

    Parameters
    ----------
    X : (n, d) data
    C : (k, d) centroids

    Returns
    -------
    (n, k) array where entry [i, j] is ||x_i - c_j||^2.

    We square-and-sum the broadcast difference. Squared distance is enough for
    K-means: sqrt is monotonic, so argmin is unchanged, and we skip the cost.
    """
    diff = X[:, None, :] - C[None, :, :]          # (n, k, d)
    return np.einsum("nkd,nkd->nk", diff, diff)   # (n, k)


def mahalanobis_sqdist(X: np.ndarray, C: np.ndarray, diag: np.ndarray) -> np.ndarray:
    """Simplified (diagonal) Mahalanobis squared distance.

    With a positive diagonal matrix ``C_mat = diag(c_1, ..., c_d)``:

        d(x, mu)^2 = (x - mu)^T C_mat (x - mu) = sum_f c_f * (x_f - mu_f)^2

    i.e. a *weighted* Euclidean distance: each feature f is scaled by c_f.
    Setting every c_f = 1 recovers plain Euclidean distance.

    Parameters
    ----------
    X    : (n, d) data
    C    : (k, d) centroids
    diag : (d,) strictly positive per-feature weights

    Returns
    -------
    (n, k) array of squared Mahalanobis distances.

    TODO(student): implement this.
      Hint: start from `euclidean_sqdist` above and weight the squared
      per-feature differences by `diag` before summing over d.
      Keep it vectorized — no Python loop over n.
    """
    diff = X[:, None, :] - C[None, :, :]
    return (diff**2*diag).sum(axis=2)