"""Multinomial Naive Bayes for text.

* NaiveBayesReference  - PROVIDED. Wraps scikit-learn's MultinomialNB. It lets the pipeline run before
                         you write code, and it is your CORRECTNESS ORACLE.
* NaiveBayesScratch    - YOURS. Implement it with NumPy only. Do not call a library Naive Bayes in it.

Model (classes c in {0 = ham, 1 = spam}, features = word counts or TF-IDF weights x_j >= 0):

    log P(c | x)  ~  log P(c)  +  sum_j  x_j * log theta_{c,j}

    P(c)        = (number of training messages of class c) / (number of training messages)
    theta_{c,j} = (N_{c,j} + alpha) / (sum_k N_{c,k} + alpha * d)       <- Laplace / add-alpha smoothing
                  where N_{c,j} = sum of feature j over the training messages of class c, d = #features

Work in LOG space (products of thousands of small probabilities underflow) and normalize the
posteriors with the log-sum-exp trick.
"""
from __future__ import annotations


class NaiveBayesReference:
    """Scikit-learn baseline with the same interface as NaiveBayesScratch."""

    def __init__(self, alpha: float = 1.0):
        from sklearn.naive_bayes import MultinomialNB
        # alpha = 0 triggers a warning in sklearn; clip like it does internally.
        self.alpha = alpha
        self._m = MultinomialNB(alpha=max(alpha, 1e-10))

    def fit(self, X, y):
        self._m.fit(X, y)
        self.classes_ = self._m.classes_
        self.class_log_prior_ = self._m.class_log_prior_
        self.feature_log_prob_ = self._m.feature_log_prob_
        return self

    def predict_log_proba(self, X):
        return self._m.predict_log_proba(X)

    def predict(self, X):
        return self._m.predict(X)


class NaiveBayesScratch:
    """YOURS TO IMPLEMENT. After fit() these attributes must exist:

        classes_            sorted array of class labels, e.g. [0, 1]
        class_log_prior_    shape (n_classes,)            log P(c)
        feature_log_prob_   shape (n_classes, n_features) log theta_{c,j}
    """

    def __init__(self, alpha: float = 1.0):
        self.alpha = float(alpha)

    # ---- the pieces to write, roughly in this order --------------------------------
    def _class_log_prior(self, y):
        """Return log P(c) for each class (in the order of self.classes_)."""
        raise NotImplementedError("compute log class priors from the label counts")

    def _feature_log_prob(self, X, y):
        """Return log theta of shape (n_classes, n_features) with add-alpha smoothing.

        X may be a scipy sparse matrix or a dense array. Think about alpha = 0: what happens to
        a word that never occurs in a class? (Report it. Do not hide it.)
        """
        raise NotImplementedError("sum the feature mass per class, smooth with alpha, take logs")

    def fit(self, X, y):
        """Estimate the parameters. Sets classes_, class_log_prior_, feature_log_prob_; returns self."""
        raise NotImplementedError("learn the class priors and the per-class word log-probabilities")

    def predict_log_proba(self, X):
        """Return log P(c | x) for every row of X, shape (n_samples, n_classes), rows normalized
        so that exp(...) sums to 1. Use log space and log-sum-exp (scipy.special.logsumexp is fine)."""
        raise NotImplementedError("joint log likelihood, then normalize in log space")

    def predict(self, X):
        """Return the most probable class label for each row of X."""
        raise NotImplementedError("argmax over the joint log likelihood")


def build_nb(cfg):
    nb = cfg["naive_bayes"]
    cls = NaiveBayesScratch if nb.get("implementation", "reference") == "scratch" else NaiveBayesReference
    return cls(alpha=float(nb.get("alpha", 1.0)))
