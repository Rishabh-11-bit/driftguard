"""Controlled real-text evaluation: removes the length / prefix / near-duplicate confounds.

The first real-text eval (scripts/run_real_text_eval.py) compared long injection strings with short
"User: ..." persona facts, so length alone could separate the classes. Here:

  * both classes get the same "User: " prefix and first-letter casing
  * texts are truncated at a word boundary to at most MAX_CHARS and length-matched by binning
    (equal numbers of attack and benign per length bin)
  * near-duplicates (token Jaccard >= 0.8) are removed before any split
  * we report length-only and style-only (TF-IDF logistic regression) baselines; if those match the
    detector, the detector has learned nothing about intent
  * zero-shot scoring of the trained detectors, with v2's training strings masked out

Usage:  python -m scripts.run_real_text_controlled          (after scripts.download_external)
"""

import copy
import json
import random
import re
from pathlib import Path

import numpy as np
import typer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from driftguard.benchmark.labelling import load_benchmark
from driftguard.detector.classifier import DriftGuardClassifier
from driftguard.detector.features import SemanticFeatureExtractor
from scripts.run_real_text_eval import load_real

app = typer.Typer()
MAX_CHARS = 200
BIN = 20
_TOK = re.compile(r"[a-z0-9]+")


def _core(s: str) -> str:
    return s[6:] if s.startswith("User: ") else s


def _cut(s: str) -> str:
    s = s.strip()
    if len(s) <= MAX_CHARS:
        return s
    s = s[:MAX_CHARS]
    return s[: s.rfind(" ")] if " " in s else s


def _fmt(s: str) -> str:
    s = _cut(_core(s))
    return "User: " + (s[0].upper() + s[1:] if s else s)


def dedupe(texts: list[str], thresh: float = 0.8) -> list[str]:
    kept, sets = [], []
    for t in texts:
        toks = set(_TOK.findall(t.lower()))
        if any(len(toks & k) / max(1, len(toks | k)) >= thresh for k in sets):
            continue
        kept.append(t)
        sets.append(toks)
    return kept


def length_match(atk: list[str], ben: list[str], rng: random.Random):
    bins_a, bins_b = {}, {}
    for t in atk:
        bins_a.setdefault(len(t) // BIN, []).append(t)
    for t in ben:
        bins_b.setdefault(len(t) // BIN, []).append(t)
    a_out, b_out = [], []
    for b in sorted(set(bins_a) & set(bins_b)):
        n = min(len(bins_a[b]), len(bins_b[b]))
        rng.shuffle(bins_a[b]), rng.shuffle(bins_b[b])
        a_out += bins_a[b][:n]
        b_out += bins_b[b][:n]
    return a_out, b_out


def scores(y, pred, proba):
    atk = y == 0
    return {"attack_tpr": float(np.mean(pred[atk] == 0)), "false_alarm": float(np.mean(pred[~atk] == 0)),
            "auroc": float(roc_auc_score(atk.astype(int), proba)) if proba is not None else None}


def cv_auroc(make, X, y, seed) -> tuple[float, float]:
    aucs = []
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(X, y):
        m = make().fit(X[tr], (y[tr] == 0).astype(int))
        aucs.append(roc_auc_score((y[te] == 0).astype(int), m.predict_proba(X[te])[:, 1]))
    return float(np.mean(aucs)), float(np.std(aucs))


@app.command()
def main(
    ext_dir: Path = Path("data/external"),
    benchmark: Path = Path("data/benchmark_hard.jsonl"),
    models: list[Path] = [Path("models/classifier_hard.pkl"), Path("models/classifier_robust.pkl"),
                          Path("models/classifier_v2.pkl")],
    out: Path = Path("results_research"),
    seed: int = 0,
) -> None:
    rng = random.Random(seed)
    atk_raw, ben_raw = load_real(ext_dir, 100000, random.Random(seed))
    atk = dedupe(sorted({_fmt(t) for t in atk_raw}))
    ben = dedupe(sorted({_fmt(t) for t in ben_raw}))
    atk = [t for t in atk if len(t) > 26]
    atk, ben = length_match(atk, ben, rng)
    texts = atk + ben
    y = np.array([0] * len(atk) + [2] * len(ben))
    print(f"controlled set: attacks={len(atk)} benign={len(ben)}  "
          f"mean len attack={np.mean([len(t) for t in atk]):.1f} benign={np.mean([len(t) for t in ben]):.1f}")

    entries = load_benchmark(benchmark)
    recs = []
    for t in texts:
        d = copy.deepcopy(rng.choice(entries))
        recs.append(dict(content=t, provenance=d.provenance, write_history=d.write_history,
                         session_id=d.session_id, last_query=d.last_query, memory_entries=d.memory_entries))
    X = SemanticFeatureExtractor().extract_batch(recs)

    result = {"n_attack": len(atk), "n_benign": len(ben), "mean_len_attack": float(np.mean([len(t) for t in atk])),
              "mean_len_benign": float(np.mean([len(t) for t in ben])), "zero_shot": {}, "baselines": {}, "cv": {}}

    # style-only baselines (these see no context features at all)
    L = np.array([[len(t)] for t in texts], dtype=float)
    result["baselines"]["length-only logistic (CV AUROC)"] = dict(zip(("mean", "std"), cv_auroc(lambda: LogisticRegression(), L, y, seed)))
    tf = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True).fit_transform(texts)
    result["baselines"]["tf-idf logistic (CV AUROC)"] = dict(zip(("mean", "std"), cv_auroc(lambda: LogisticRegression(max_iter=1000), tf, y, seed)))

    # zero-shot detectors; v2 trained on the first half of a seed-42 shuffle -> mask those strings
    a42, b42 = load_real(ext_dir, 600, random.Random(42))
    h = len(a42) // 2
    seen = [_cut(_core(s)).lower() for s in a42[:h] + b42[:h]]
    seen_set = set(seen)

    def v2_seen(t: str) -> bool:
        c = _core(t).lower()
        return c in seen_set or any(s.startswith(c) or c.startswith(s) for s in seen)

    unseen = np.array([not v2_seen(t) for t in texts])
    for m in models:
        if not m.exists():
            continue
        d = DriftGuardClassifier.load(m)
        k = len(d._feature_names or []) or 18
        pred = np.asarray(d.predict(X[:, :k]))
        proba = d.predict_proba(X[:, :k])[:, 0]
        mask = unseen if m.stem == "classifier_v2" else np.ones(len(y), bool)
        if mask.sum() > 20 and len(set(y[mask])) == 2:
            r = scores(y[mask], pred[mask], proba[mask])
            r["n_scored"] = int(mask.sum())
            result["zero_shot"][m.stem] = r
            print("zero-shot", m.stem, {a: round(b, 3) for a, b in r.items() if b is not None}, flush=True)

    # learnability with text + context features
    from scripts.train_detector import FEATURE_NAMES
    folds = []
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(X, y):
        mdl = DriftGuardClassifier(backend="random_forest", n_estimators=100, random_state=seed)
        mdl.fit(X[tr], y[tr], feature_names=FEATURE_NAMES)
        folds.append(scores(y[te], np.asarray(mdl.predict(X[te])), mdl.predict_proba(X[te])[:, 0]))
    result["cv"]["random_forest on matched real text"] = {a: float(np.mean([f[a] for f in folds])) for a in folds[0]}
    print("baselines", {a: round(v["mean"], 3) for a, v in result["baselines"].items()})
    print("cv", {a: round(v, 3) for a, v in result["cv"]["random_forest on matched real text"].items()})
    out.mkdir(exist_ok=True)
    (out / "real_text_controlled.json").write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    app()
