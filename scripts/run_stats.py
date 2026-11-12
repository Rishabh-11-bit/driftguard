"""Stronger statistics for the main comparison and the feature-family ablation.

Fixes the weaknesses a reviewer would raise about scripts/run_research_eval.py:
  * intervals use the Nadeau-Bengio corrected resampled t (variance inflated by 1/J + n_test/n_train),
    because folds of repeated CV overlap and a plain t-interval is far too narrow
  * 10 x 5-fold repeats (50 folds) instead of 3 x 5, for every ablation variant too
  * paired per-fold differences against the full model / against baselines, with Holm correction
  * a second scheme, grouped CV, where near-duplicate texts (same content once digits are masked)
    are never split across train and test (a proxy for template grouping, since the hardening step
    removes the template id)

Usage:  python -m scripts.run_stats          Writes results_research/stats.{json,md}
"""

import json
import re
from pathlib import Path

import numpy as np
import typer
from scipy import stats
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedGroupKFold

from driftguard.detector.classifier import DriftGuardClassifier
from eval.baselines.rule_based import RuleBasedDetector
from eval.harness import EvaluationHarness
from scripts.run_research_eval import FAMILIES
from scripts.train_detector import FEATURE_NAMES

app = typer.Typer()


def corrected_t(diffs: np.ndarray, n_train: int, n_test: int) -> dict:
    """Nadeau & Bengio (2003) corrected resampled t-test on per-fold paired differences."""
    j = len(diffs)
    mean, var = float(diffs.mean()), float(diffs.var(ddof=1))
    se = np.sqrt((1.0 / j + n_test / n_train) * var) if var > 0 else 0.0
    half = float(stats.t.ppf(0.975, j - 1) * se)
    p = 1.0 if se == 0 and mean == 0 else (0.0 if se == 0 else float(2 * stats.t.sf(abs(mean / se), j - 1)))
    return {"mean": mean, "ci95": [mean - half, mean + half], "p": p}


def holm(pvals: dict[str, float]) -> dict[str, float]:
    order = sorted(pvals, key=pvals.get)
    m, adj, running = len(order), {}, 0.0
    for i, k in enumerate(order):
        running = max(running, min(1.0, (m - i) * pvals[k]))
        adj[k] = running
    return adj


def folds(scheme: str, X, y, groups, repeats: int, seed: int):
    if scheme == "stratified":
        yield from RepeatedStratifiedKFold(n_splits=5, n_repeats=repeats, random_state=seed).split(X, y)
    else:
        for r in range(repeats):
            yield from StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed + r).split(X, y, groups)


def run_model(make, X, y, tr, te, cols):
    m = make()
    if isinstance(m, RuleBasedDetector):
        pred = m.predict(X[te])
    else:
        if isinstance(m, DriftGuardClassifier):
            m.fit(X[tr][:, cols], y[tr], feature_names=[FEATURE_NAMES[c] for c in cols])
        else:
            m.fit(X[tr][:, cols], y[tr])
        pred = m.predict(X[te][:, cols])
    pred = np.asarray(pred)
    atk, ben = y[te] == 0, y[te] != 0
    return {"accuracy": float(np.mean(pred == y[te])),
            "attack_tpr": float(np.mean(pred[atk] == 0)) if atk.any() else np.nan,
            "FAR": float(np.mean(pred[ben] == 0)) if ben.any() else np.nan}


@app.command()
def main(benchmark: Path = Path("data/benchmark_hard.jsonl"), out: Path = Path("results_research"),
         repeats: int = 10, seed: int = 0) -> None:
    h = EvaluationHarness(str(benchmark))
    h.load_and_extract()
    X, y = h._X, h._y
    groups = np.array([hash(re.sub(r"\d+", "#", e.content.lower())) for e in h._entries])
    n = len(y)
    allc = list(range(X.shape[1]))
    rf = lambda: DriftGuardClassifier(backend="random_forest", n_estimators=100, random_state=seed)
    models = {"DriftGuard (RF)": (rf, allc),
              "Logistic regression": (lambda: DriftGuardClassifier(backend="logistic", random_state=seed), allc),
              "Rule-based": (RuleBasedDetector, allc),
              "Majority class": (lambda: DummyClassifier(strategy="most_frequent"), allc)}
    for fam, cols in FAMILIES.items():
        models[f"RF without {fam}"] = (rf, [c for c in allc if c not in cols])
        models[f"RF only {fam}"] = (rf, cols)

    result = {"n": n, "repeats": repeats, "schemes": {}}
    for scheme in ("stratified", "grouped"):
        print(f"[{scheme}] {repeats} x 5-fold ...", flush=True)
        per = {k: [] for k in models}
        sizes = []
        for tr, te in folds(scheme, X, y, groups, repeats, seed):
            sizes.append((len(tr), len(te)))
            for k, (make, cols) in models.items():
                per[k].append(run_model(make, X, y, tr, te, cols))
        n_tr, n_te = int(np.mean([s[0] for s in sizes])), int(np.mean([s[1] for s in sizes]))
        res = {"n_folds": len(sizes), "models": {}, "paired_vs_full_rf": {}}
        for k, rows in per.items():
            res["models"][k] = {}
            for met in ("accuracy", "attack_tpr", "FAR"):
                v = np.array([r[met] for r in rows], float)
                ct = corrected_t(v - 0.0, n_tr, n_te)        # interval for the mean (not a difference)
                res["models"][k][met] = {"mean": float(np.nanmean(v)), "ci95": ct["ci95"]}
        base = np.array([r["accuracy"] for r in per["DriftGuard (RF)"]])
        pv, tests = {}, {}
        for k, rows in per.items():
            if k == "DriftGuard (RF)":
                continue
            d = base - np.array([r["accuracy"] for r in rows])   # positive = full RF better
            tests[k] = corrected_t(d, n_tr, n_te)
            pv[k] = tests[k]["p"]
        adj = holm(pv)
        for k in tests:
            tests[k]["p_holm"] = adj[k]
        res["paired_vs_full_rf"] = tests
        result["schemes"][scheme] = res

    out.mkdir(exist_ok=True)
    (out / "stats.json").write_text(json.dumps(result, indent=2))
    lines = []
    for scheme, res in result["schemes"].items():
        lines += [f"\n### {scheme} CV ({res['n_folds']} folds, Nadeau-Bengio corrected 95 % CI)\n",
                  "| Model | Accuracy | Attack TPR | FAR | acc. diff vs full RF (CI) | Holm p |", "|---|---|---|---|---|---|"]
        for k, m in res["models"].items():
            f = lambda met: f"{m[met]['mean']:.3f} [{m[met]['ci95'][0]:.3f}, {m[met]['ci95'][1]:.3f}]"
            t = res["paired_vs_full_rf"].get(k)
            d = "-" if t is None else f"{t['mean']:+.3f} [{t['ci95'][0]:+.3f}, {t['ci95'][1]:+.3f}]"
            p = "-" if t is None else f"{t['p_holm']:.3g}"
            lines.append(f"| {k} | {f('accuracy')} | {f('attack_tpr')} | {f('FAR')} | {d} | {p} |")
    (out / "stats.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    app()
