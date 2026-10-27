"""Single entry point: `python -m src.run --config config.yaml`.

Reproducibility standard: this command must recreate your reported results. It writes
results/metrics.json and figures to the output directory, for every protocol in config.yaml.
"""
from __future__ import annotations

import argparse
import json
import os
import random

import numpy as np
import yaml
from scipy import sparse

from . import evaluate as ev
from .data import load_split, read_corpus, read_enron, deduplicate
from .features import fit_transform_split
from .naive_bayes import NaiveBayesReference, NaiveBayesScratch, build_nb
from .features import build_vectorizer
from .models import build_svm, build_knn, spam_scores_from_log_proba, explore_kmeans


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


def run_protocol(cfg, protocol, out):
    seed = cfg["seed"]
    tr_t, y_tr, te_t, y_te, info = load_split(cfg, protocol)
    X_tr, X_te, vec = fit_transform_split(cfg, tr_t, te_t)   # vocabulary + IDF from TRAIN only
    info["n_features"] = int(X_tr.shape[1])
    res = {"protocol": protocol, **info, "models": {}}

    def record(name, y_pred, scores, extra=None):
        m = ev.classification_metrics(y_te, y_pred, scores)
        if extra:
            m.update(extra)
        res["models"][name] = m
        curves[name] = (y_te, scores)
        return m

    curves = {}

    # ---- Naive Bayes (yours when implementation: scratch) ---------------------------------
    nb = build_nb(cfg).fit(X_tr, y_tr)
    lp = nb.predict_log_proba(X_te)
    record("naive_bayes", nb.predict(X_te), spam_scores_from_log_proba(lp, nb.classes_),
           {"implementation": cfg["naive_bayes"].get("implementation", "reference"),
            "alpha": float(cfg["naive_bayes"].get("alpha", 1.0))})
    tn, fp, fn, tp = ev.confusion(y_te, nb.predict(X_te))
    ev.save_confusion_png(tn, fp, fn, tp, os.path.join(out, f"confusion_nb_{protocol}.png"),
                          f"Naive Bayes ({protocol})")

    # ---- does YOUR Naive Bayes agree with scikit-learn? -----------------------------------
    if cfg["naive_bayes"].get("implementation") == "scratch" and cfg.get("compare_to_reference", False):
        ref = NaiveBayesReference(alpha=float(cfg["naive_bayes"].get("alpha", 1.0))).fit(X_tr, y_tr)
        agree = float(np.mean(ref.predict(X_te) == nb.predict(X_te)))
        res["reference_check"] = {
            "prediction_agreement": agree,
            "max_abs_diff_log_prob": float(np.max(np.abs(
                ref.predict_log_proba(X_te) - lp))),
            "max_abs_diff_feature_log_prob": float(np.max(np.abs(
                ref.feature_log_prob_ - nb.feature_log_prob_))),
        }

    # ---- smoothing sweep (alpha = 0 exposes the zero-frequency problem) -------------------
    sweep = []
    for a in cfg["evaluate"].get("alpha_sweep", []):
        cls = NaiveBayesScratch if cfg["naive_bayes"].get("implementation") == "scratch" else NaiveBayesReference
        with np.errstate(divide="ignore", invalid="ignore"):
            try:
                m = cls(alpha=float(a)).fit(X_tr, y_tr)
                lpa = m.predict_log_proba(X_te)
                met = ev.classification_metrics(y_te, m.predict(X_te))
                sweep.append({"alpha": float(a), "f1": met["f1"], "mcc": met["mcc"],
                              "blocked_ham_pct": met["blocked_ham_pct"],
                              "spam_caught_pct": met["spam_caught_pct"],
                              "any_nonfinite_log_proba": bool(not np.isfinite(lpa).all())})
            except NotImplementedError:
                break
    res["naive_bayes_alpha_sweep"] = sweep
    if sweep:
        ev.save_alpha_sweep([s["alpha"] for s in sweep], [s["f1"] for s in sweep],
                            [s["mcc"] for s in sweep], os.path.join(out, f"alpha_sweep_{protocol}.png"))

    # ---- linear SVM ---------------------------------------------------------------
    svm = build_svm(cfg).fit(X_tr, y_tr)
    record("linear_svm", svm.predict(X_te), svm.decision_function(X_te), {"C": float(cfg["svm"]["C"])})

    # ---- KNN, k = 1, 3, 5 as in the paper ----------------------------------------------
    for k in cfg["knn"]["ks"]:
        knn = build_knn(cfg, k).fit(X_tr, y_tr)
        proba = knn.predict_proba(X_te)[:, 1] if len(knn.classes_) == 2 else None
        record(f"knn_k{k}", knn.predict(X_te), proba, {"k": int(k)})

    # ---- exploratory clustering (labels only to EVALUATE) -------------------------------
    res["kmeans_exploration"] = explore_kmeans(cfg, X_tr, y_tr)

    ev.save_pr_curves({k: v for k, v in curves.items() if v[1] is not None},
                      os.path.join(out, f"pr_curves_{protocol}.png"))
    return res


def run_cross_corpus(cfg, out):
    """EXPLORATORY (Enron-Spam). Train on the main corpus, test on Enron, and the reverse.
    Shows how much a spam filter transfers across domains (SMS <-> e-mail). Labels are used to train
    and to score only; each vectorizer is fitted on its own TRAINING corpus."""
    a_t, a_y = read_corpus(cfg)
    if cfg["data"].get("deduplicate", True):
        a_t, a_y, _ = deduplicate(a_t, a_y)
    b_t, b_y = read_enron(cfg["data"]["enron_path"])
    if cfg["data"].get("deduplicate", True):
        b_t, b_y, _ = deduplicate(b_t, b_y)
    names = {"sms_to_enron": (a_t, a_y, b_t, b_y), "enron_to_sms": (b_t, b_y, a_t, a_y)}
    res = {"main_source": cfg["data"]["source"], "enron_path": cfg["data"]["enron_path"], "directions": {}}
    for name, (tr_t, tr_y, te_t, te_y) in names.items():
        vec = build_vectorizer(cfg)
        X_tr = vec.fit_transform(tr_t)
        X_te = vec.transform(te_t)
        nb = build_nb(cfg).fit(X_tr, tr_y)
        svm = build_svm(cfg).fit(X_tr, tr_y)
        res["directions"][name] = {
            "n_train": len(tr_t), "n_test": len(te_t),
            "spam_rate_train": float(tr_y.mean()), "spam_rate_test": float(te_y.mean()),
            "naive_bayes": ev.classification_metrics(
                te_y, nb.predict(X_te), spam_scores_from_log_proba(nb.predict_log_proba(X_te), nb.classes_)),
            "linear_svm": ev.classification_metrics(te_y, svm.predict(X_te), svm.decision_function(X_te)),
        }
    return res


def main(config_path):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    set_seed(cfg["seed"])
    out = cfg["output"]["dir"]
    os.makedirs(out, exist_ok=True)
    metrics = {"seed": cfg["seed"], "data_source": cfg["data"]["source"],
               "implementation": cfg["naive_bayes"].get("implementation", "reference"),
               "config": cfg, "protocols": {}}
    for protocol in cfg["protocols"]:
        print(f"== protocol: {protocol}")
        metrics["protocols"][protocol] = run_protocol(cfg, protocol, out)
    if cfg["evaluate"].get("cross_corpus", False):
        print("== exploratory: cross-corpus (main corpus <-> Enron-Spam)")
        metrics["cross_corpus"] = run_cross_corpus(cfg, out)
    first = metrics["protocols"][cfg["protocols"][0]]
    metrics["summary"] = {name: {k: round(m[k], 4) for k in ("precision", "recall", "f1", "mcc") if k in m}
                          for name, m in first["models"].items()}
    ev.write_metrics(metrics, os.path.join(out, "metrics.json"))
    print("Results written to", out)
    print(json.dumps(metrics["summary"], indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    main(ap.parse_args().config)
