"""Load the SMS Spam Collection and split it. PROVIDED: you may read it, but you should not need to edit it.

Two protocols are supported (see config.yaml -> protocols):

* "paper":      the first `train_frac` of the file for training, the rest for testing
                (the protocol of Almeida et al., 2011, which you compare against).
* "stratified": a seeded random split that keeps the spam rate in both parts.

Labels: spam = 1, ham = 0. Labels are for TRAINING the supervised classifiers and for
EVALUATION; they must never influence the feature extraction (see features.py).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

SAMPLE = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "sample_sms.tsv"


def read_enron(path):
    """Read an Enron-Spam folder (Metsis et al., 2006) with `ham/` and `spam/` subfolders, e.g. the
    extracted `enron1/`. File names look like `0001.2001-01-10.farmer.ham.txt`: we order the messages
    by the date in the name (then the arrival number) so the paper-style split is chronological.
    Returns (texts, labels) with spam = 1."""
    path = Path(path)
    if not (path / "ham").is_dir() and (path / "enron1").is_dir():
        path = path / "enron1"
    items = []
    for lab, sub in ((0, "ham"), (1, "spam")):
        folder = path / sub
        if not folder.is_dir():
            raise FileNotFoundError(f"{folder} not found. See data/README.md (Enron-Spam).")
        for f in folder.glob("*.txt"):
            parts = f.name.split(".")
            date = parts[1] if len(parts) > 2 else ""
            items.append((date, parts[0], lab, f))
    items.sort(key=lambda t: (t[0], t[1], t[2]))
    texts = [f.read_text(encoding="latin-1", errors="ignore") for *_, f in items]
    return texts, np.array([lab for _, _, lab, _ in items], dtype=int)


def read_corpus(cfg):
    """Return (texts, labels) as Python list / np.ndarray, in file (or arrival) order."""
    d = cfg["data"]
    source = d.get("source", "sample")
    if source == "enron":
        return read_enron(d["enron_path"])
    path = SAMPLE if source == "sample" else Path(d["path"])
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. See data/README.md for how to download the SMS Spam Collection.")
    texts, labels = [], []
    with open(path, encoding="latin-1") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip() or "\t" not in line:
                continue
            lab, txt = line.split("\t", 1)
            labels.append(1 if lab.strip().lower() == "spam" else 0)
            texts.append(txt)
    return texts, np.array(labels, dtype=int)


def deduplicate(texts, labels):
    """Keep the FIRST occurrence of every exact-duplicate message. Returns (texts, labels, n_removed)."""
    seen, keep = set(), []
    for i, t in enumerate(texts):
        if t not in seen:
            seen.add(t)
            keep.append(i)
    return [texts[i] for i in keep], labels[keep], len(texts) - len(keep)


def split(texts, labels, protocol, train_frac, seed):
    """Return (train_texts, y_train, test_texts, y_test)."""
    n = len(texts)
    if protocol == "paper":
        cut = int(round(train_frac * n))
        idx_tr, idx_te = np.arange(cut), np.arange(cut, n)
    elif protocol == "stratified":
        rng = np.random.RandomState(seed)
        idx_tr = []
        for c in (0, 1):
            idx = np.where(labels == c)[0]
            rng.shuffle(idx)
            idx_tr.extend(idx[: int(round(train_frac * len(idx)))])
        idx_tr = np.array(sorted(idx_tr))
        idx_te = np.setdiff1d(np.arange(n), idx_tr)
    else:
        raise ValueError(f"unknown protocol: {protocol!r}")
    return ([texts[i] for i in idx_tr], labels[idx_tr],
            [texts[i] for i in idx_te], labels[idx_te])


def load_split(cfg, protocol):
    """Read, optionally de-duplicate, and split. Also returns bookkeeping for the report."""
    texts, labels = read_corpus(cfg)
    n_raw, removed = len(texts), 0
    if cfg["data"].get("deduplicate", True):
        texts, labels, removed = deduplicate(texts, labels)
    tr_t, y_tr, te_t, y_te = split(texts, labels, protocol,
                                   float(cfg["data"]["train_frac"]), cfg["seed"])
    info = {
        "n_messages_raw": n_raw,
        "duplicates_removed": removed,
        "n_train": len(tr_t), "n_test": len(te_t),
        "spam_rate_train": float(y_tr.mean()), "spam_rate_test": float(y_te.mean()),
        # test messages whose exact text also occurs in training = leakage if not removed
        "test_messages_also_in_train": int(sum(t in set(tr_t) for t in te_t)),
        # accuracy of the trivial "everything is ham" filter: the baseline to beat
        "all_ham_accuracy": float(1.0 - y_te.mean()),
    }
    return tr_t, y_tr, te_t, y_te, info
