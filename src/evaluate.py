"""Metrics and figures. PROVIDED.

Spam = positive class (1). Everything here is computed on the TEST split.
Names follow the SMS-spam literature (Almeida et al., 2011):

    SC%  spam caught   = recall of the spam class        = TP / (TP + FN)
    BH%  blocked ham   = false-positive rate             = FP / (FP + TN)   (legitimate SMS blocked)
    MCC  Matthews correlation coefficient (uses all four cells; robust to imbalance)
"""
from __future__ import annotations

import json
import os

import numpy as np
from sklearn import metrics as skm


def confusion(y_true, y_pred):
    tn, fp, fn, tp = skm.confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return int(tn), int(fp), int(fn), int(tp)


def classification_metrics(y_true, y_pred, scores=None):
    """Binary metrics for the spam class. `scores` = any monotone spam score (for ROC-AUC / PR-AUC)."""
    tn, fp, fn, tp = confusion(y_true, y_pred)
    out = {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "accuracy": float((tp + tn) / max(tp + tn + fp + fn, 1)),
        "precision": float(skm.precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(skm.recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(skm.f1_score(y_true, y_pred, zero_division=0)),
        "mcc": float(skm.matthews_corrcoef(y_true, y_pred)),
        "spam_caught_pct": float(100.0 * tp / max(tp + fn, 1)),
        "blocked_ham_pct": float(100.0 * fp / max(fp + tn, 1)),
    }
    if scores is not None and len(set(y_true)) == 2:
        out["roc_auc"] = float(skm.roc_auc_score(y_true, scores))
        out["pr_auc"] = float(skm.average_precision_score(y_true, scores))
    return out


def internal_cluster_metrics(y_true, clusters):
    """External scores of a clustering against the labels (labels used to EVALUATE only)."""
    hom, com, v = skm.homogeneity_completeness_v_measure(y_true, clusters)
    return {"homogeneity": float(hom), "completeness": float(com), "v_measure": float(v),
            "ari": float(skm.adjusted_rand_score(y_true, clusters))}


def write_metrics(metrics, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)


def save_confusion_png(tn, fp, fn, tp, path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(3.6, 3.2))
    cm = np.array([[tn, fp], [fn, tp]])
    ax.imshow(cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, str(v), ha="center", va="center")
    ax.set_xticks([0, 1], ["ham", "spam"]); ax.set_yticks([0, 1], ["ham", "spam"])
    ax.set_xlabel("predicted"); ax.set_ylabel("actual"); ax.set_title(title, fontsize=9)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def save_pr_curves(curves, path):
    """curves: {name: (y_true, scores)} -> one precision-recall figure."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(4.4, 3.6))
    for name, (y, s) in curves.items():
        p, r, _ = skm.precision_recall_curve(y, s)
        ax.plot(r, p, label=name)
    ax.set_xlabel("recall (spam caught)"); ax.set_ylabel("precision"); ax.legend(fontsize=7)
    ax.set_title("Precision-recall (test)", fontsize=9)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def save_alpha_sweep(alphas, f1s, mccs, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(4.4, 3.4))
    xs = np.arange(len(alphas))
    ax.plot(xs, f1s, marker="o", label="F1"); ax.plot(xs, mccs, marker="s", label="MCC")
    ax.set_xticks(xs, [str(a) for a in alphas]); ax.set_xlabel("smoothing alpha"); ax.legend()
    ax.set_title("Naive Bayes: effect of smoothing (test)", fontsize=9)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)
