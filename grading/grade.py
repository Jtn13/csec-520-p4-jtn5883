#!/usr/bin/env python3
"""Automated grading harness for CSEC 520/620: Project 4 (SMS spam).

Runs the OBJECTIVE checks a grading agent needs as evidence, then writes grading/auto_report.json.
The agent reads that file, adds qualitative judgment (discussion quality, correctness, code quality),
and produces the final grade per grading/AGENT_GRADING.md.

Stdlib only. Run from the repo root:  python grading/grade.py
"""
from __future__ import annotations

import ast
import json
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REQUIRED = ["config.yaml", "src/run.py", "src/data.py", "src/features.py", "src/naive_bayes.py",
            "src/models.py", "src/evaluate.py", "README.md", "requirements.txt", "SUBMISSION.md"]
CLASSIFIERS = {"naive_bayes", "linear_svm", "knn_k1", "knn_k3", "knn_k5"}
UNIT = {"accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"}

_venv = ROOT / ".venv" / "bin" / "python"
PYEXE = str(_venv) if _venv.exists() else sys.executable


def run(cmd, timeout=900):
    try:
        p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=timeout)
        return p.returncode, (p.stdout + p.stderr)[-4000:]
    except subprocess.TimeoutExpired:
        return 124, f"TIMEOUT after {timeout}s"
    except FileNotFoundError as e:
        return 127, str(e)


def _read(path: Path) -> str:
    try:
        return path.read_text(errors="ignore")
    except OSError:
        return ""


def _metrics():
    p = ROOT / "results" / "metrics.json"
    if not p.exists():
        return None, "results/metrics.json not found"
    try:
        return json.loads(p.read_text()), None
    except Exception as e:
        return None, f"invalid JSON: {e}"


def check_structure():
    missing = [f for f in REQUIRED if not (ROOT / f).exists()]
    return {"id": "structure", "status": "pass" if not missing else "fail",
            "evidence": "all present" if not missing else f"missing: {missing}"}


def check_tests():
    rc, out = run([PYEXE, "-m", "pytest", "-q"], timeout=300)
    status = "pass" if rc == 0 else ("dep_missing" if "No module named" in out else "fail")
    return {"id": "tests", "status": status, "returncode": rc, "evidence": out[-800:]}


def check_reproduce():
    rc, out = run([PYEXE, "-m", "src.run", "--config", "config.yaml"])
    if rc == 0:
        status = "pass"
    elif "NotImplementedError" in out:
        status = "fail"          # stubs never filled in
    elif "No module named" in out:
        status = "dep_missing"
    else:
        status = "fail"
    return {"id": "reproduce_runs", "status": status, "returncode": rc, "evidence": out[-1200:]}


def check_metrics():
    m, err = _metrics()
    if m is None:
        return {"id": "metrics_present", "status": "fail", "evidence": err}
    problems = []
    protocols = m.get("protocols", {})
    if not protocols:
        problems.append("no protocols in metrics.json")
    for pname, p in protocols.items():
        missing = CLASSIFIERS - set(p.get("models", {}))
        if missing:
            problems.append(f"{pname}: missing models {sorted(missing)}")
        for name, mm in p.get("models", {}).items():
            for k in UNIT:
                v = mm.get(k)
                if isinstance(v, (int, float)) and not (0 <= v <= 1):
                    problems.append(f"{pname}/{name}/{k} out of range: {v}")
    return {"id": "metrics_present", "status": "pass" if not problems else "fail",
            "evidence": problems or {"protocols": sorted(protocols),
                                     "data_source": m.get("data_source")}}


def check_protocols():
    m, err = _metrics()
    if m is None:
        return {"id": "protocols_run", "status": "fail", "evidence": err}
    have = set(m.get("protocols", {}))
    ok = {"paper", "stratified"} <= have
    return {"id": "protocols_run", "status": "pass" if ok else "review",
            "evidence": f"protocols in metrics.json: {sorted(have)} (both 'paper' and 'stratified' expected)"}


def check_data_source():
    m, _ = _metrics()
    src = (m or {}).get("data_source")
    return {"id": "real_corpus_used", "status": "pass" if src == "file" else "review",
            "evidence": f"data.source = {src!r}. The bundled 'sample' is only a debugging aid; "
                        "the deliverable must report results on the SMS Spam Collection."}


def check_duplicates():
    m, _ = _metrics()
    if not m:
        return {"id": "duplicates", "status": "review", "evidence": "no metrics"}
    ev = {n: {"duplicates_removed": p.get("duplicates_removed"),
              "test_messages_also_in_train": p.get("test_messages_also_in_train")}
          for n, p in m.get("protocols", {}).items()}
    leaky = any((v["test_messages_also_in_train"] or 0) > 0 for v in ev.values())
    return {"id": "duplicates", "status": "review" if leaky else "pass", "evidence": ev}


def check_exploratory():
    """Bonus work: Enron-Spam cross-corpus run or the pipeline run on Enron."""
    m, err = _metrics()
    if m is None:
        return {"id": "exploratory_enron", "status": "review", "evidence": err}
    cross = m.get("cross_corpus")
    on_enron = m.get("data_source") == "enron"
    done = bool(cross) or on_enron
    return {"id": "exploratory_enron", "status": "pass" if done else "not_attempted",
            "evidence": {"cross_corpus_in_metrics": bool(cross), "pipeline_on_enron": on_enron,
                         "note": "bonus only (up to +10); not attempting it costs nothing"}}


def _scratch_class():
    try:
        tree = ast.parse(_read(ROOT / "src" / "naive_bayes.py"))
    except SyntaxError:
        return None
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "NaiveBayesScratch":
            return node
    return None


def _raises_not_implemented(fn):
    for sub in ast.walk(fn):
        if isinstance(sub, ast.Raise):
            exc = sub.exc
            name = getattr(exc, "id", None) or getattr(getattr(exc, "func", None), "id", None)
            if name == "NotImplementedError":
                return True
    return False


def check_scratch_implemented():
    cls = _scratch_class()
    if cls is None:
        return {"id": "scratch_implemented", "status": "fail",
                "evidence": "NaiveBayesScratch class not found in src/naive_bayes.py"}
    wanted = {"_class_log_prior", "_feature_log_prob", "fit", "predict_log_proba", "predict"}
    stubs, found = [], set()
    for node in cls.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in wanted:
            found.add(node.name)
            if _raises_not_implemented(node):
                stubs.append(node.name)
    missing = sorted(wanted - found)
    ok = not stubs and not missing
    return {"id": "scratch_implemented", "status": "pass" if ok else "fail",
            "evidence": {"still_stubs": sorted(stubs), "methods_missing": missing}}


def check_no_library_nb_in_scratch():
    cls = _scratch_class()
    if cls is None:
        return {"id": "no_library_nb_in_scratch", "status": "fail", "evidence": "class not found"}
    hits = []
    for node in ast.walk(cls):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            mod = getattr(node, "module", "") or ""
            names = ",".join(a.name for a in node.names)
            if "naive_bayes" in mod or "NB" in names or "NaiveBayes" in names:
                hits.append(f"import {mod}.{names}")
        if isinstance(node, ast.Call):
            fn = node.func
            name = getattr(fn, "id", None) or getattr(fn, "attr", None)
            if name in {"MultinomialNB", "BernoulliNB", "GaussianNB", "ComplementNB", "CategoricalNB",
                        "NaiveBayesReference"}:
                hits.append(f"call {name}()")
    return {"id": "no_library_nb_in_scratch", "status": "clean" if not hits else "violation",
            "evidence": hits or "no library Naive Bayes call inside NaiveBayesScratch"}


def check_config_uses_scratch():
    txt = _read(ROOT / "config.yaml")
    m = re.search(r"^\s*implementation:\s*([A-Za-z_0-9]+)", txt, re.M)   # first match = naive_bayes
    impl = m.group(1) if m else None
    return {"id": "config_uses_scratch", "status": "pass" if impl == "scratch" else "fail",
            "evidence": f"naive_bayes.implementation = {impl!r} (must be 'scratch' for submission)"}


def check_reference_agreement():
    m, err = _metrics()
    if m is None:
        return {"id": "reference_agreement", "status": "review", "evidence": err}
    res = {}
    for pname, p in m.get("protocols", {}).items():
        rc = p.get("reference_check")
        if rc:
            res[pname] = rc
    if not res:
        return {"id": "reference_agreement", "status": "review",
                "evidence": "no reference_check (needs implementation: scratch and compare_to_reference: true)"}
    ok = all(v.get("prediction_agreement", 0) >= 0.99 for v in res.values())
    return {"id": "reference_agreement", "status": "pass" if ok else "review", "evidence": res}


def check_leakage():
    """Heuristic: the vectorizer must not be fitted on test text or on all the data."""
    hits = []
    pats = [r"fit(_transform)?\s*\(\s*(X_?te|test|texts|all_texts)", r"\.fit\s*\(\s*X_te", r"\.fit\s*\([^)]*y_te",
            r"vec(torizer)?\.fit(_transform)?\s*\(\s*(texts|corpus|all)"]
    for pyf in sorted((ROOT / "src").glob("*.py")):
        text = _read(pyf)
        for pat in pats:
            for m in re.finditer(pat, text):
                hits.append(f"{pyf.name}: {m.group(0).strip()}")
    return {"id": "leakage_scan", "status": "review" if hits else "clean",
            "evidence": hits or "no sign of fitting on test or all data (heuristic only)"}


def check_report():
    rdir = ROOT / "report"
    files = ([p.name for p in rdir.glob("**/*") if p.suffix.lower() in {".tex", ".pdf", ".md"}]
             if rdir.exists() else [])
    has_real = any(f.lower().endswith((".tex", ".pdf")) for f in files)
    return {"id": "report_present", "status": "pass" if has_real else "review",
            "evidence": f"report files: {files}" if files else "no report/ .tex or .pdf found"}


def check_git_hygiene():
    txt = _read(ROOT / ".gitignore")
    committed = ([p.name for p in (ROOT / "data").glob("*") if p.name not in {"README.md", ".gitkeep"}]
                 if (ROOT / "data").exists() else [])
    ok = "data/" in txt and "results/" in txt
    return {"id": "git_hygiene", "status": "pass" if ok else "review",
            "evidence": {"gitignore_covers_data_results": ok, "possible_committed_data": committed}}


def main():
    checks = [check_structure(), check_tests(), check_reproduce(), check_metrics(), check_protocols(),
              check_data_source(), check_duplicates(), check_config_uses_scratch(),
              check_scratch_implemented(), check_no_library_nb_in_scratch(),
              check_reference_agreement(), check_leakage(), check_report(), check_git_hygiene(), check_exploratory()]
    report = {
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "repo": ROOT.name,
        "project": "Project 4: Detecting SMS Spam",
        "checks": checks,
        "note_for_agent": (
            "These are OBJECTIVE signals only. status 'dep_missing' means the grader environment lacked a "
            "dependency: set up the env and re-run before penalizing reproducibility. "
            "'scratch_implemented', 'config_uses_scratch' and 'no_library_nb_in_scratch' are the key "
            "integrity checks: the student must write multinomial Naive Bayes themselves. 'real_corpus_used' "
            "is 'review' when only the bundled sample was run. Confirm heuristic flags by reading the code, "
            "then score every rubric criterion per AGENT_GRADING.md."),
    }
    out = ROOT / "grading" / "auto_report.json"
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
