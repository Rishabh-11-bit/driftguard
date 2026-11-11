"""Research-grade evaluation: repeated CV with CIs, baselines, family ablation, McNemar test.

Usage:  python -m scripts.run_research_eval
Writes results_research/research_metrics.{json,md}.
"""

import json
from pathlib import Path

import numpy as np
import typer
from scipy import stats
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold

from driftguard.detector.classifier import DriftGuardClassifier
from eval.baselines.rule_based import RuleBasedDetector
from eval.extended_metrics import security_metrics
from eval.harness import EvaluationHarness
from eval.metrics import compute_all_metrics
from scripts.train_detector import FEATURE_NAMES

app = typer.Typer()

FAMILIES = {
    "provenance": list(range(0, 3)),
    "temporal": list(range(3, 7)),
    "consistency": list(range(7, 12)),
    "causal/content": list(range(12, 18)),
}


def _fold_metrics(y, pred, proba=None):
    m = compute_all_metrics(y, pred, proba)
    s = security_metrics(y, pred)
    from sklearn.metrics import matthews_corrcoef
    return {"accuracy": m["accuracy"], "f1_macro": m["f1_macro"], "FAR": m["FAR"], "MAR": m["MAR"],
            "attack_tpr": s["attack_tpr"], "mcc": float(matthews_corrcoef(y, pred))}


def _summ(rows):
    out = {}
    for k in rows[0]:
        v = np.array([r[k] for r in rows], float)
        half = stats.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v))
        out[k] = {"mean": float(v.mean()), "std": float(v.std(ddof=1)), "ci95": float(half)}
    return out


def cv(make, X, y, cols=None, n_splits=5, n_repeats=3, seed=0):
    cols = cols if cols is not None else list(range(X.shape[1]))
    rows = []
    for tr, te in RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed).split(X, y):
        m = make()
        if isinstance(m, RuleBasedDetector):
            pred, proba = m.predict(X[te]), None
        else:
            m.fit(X[tr][:, cols], y[tr], feature_names=[FEATURE_NAMES[c] for c in cols]) \
                if isinstance(m, DriftGuardClassifier) else m.fit(X[tr][:, cols], y[tr])
            pred = m.predict(X[te][:, cols])
            proba = m.predict_proba(X[te][:, cols]) if hasattr(m, "predict_proba") else None
        rows.append(_fold_metrics(y[te], np.asarray(pred), proba))
    return _summ(rows)


def oof(make, X, y, seed=0):
    pred = np.zeros_like(y)
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(X, y):
        m = make()
        if isinstance(m, RuleBasedDetector):
            pred[te] = m.predict(X[te])
        else:
            m.fit(X[tr], y[tr], feature_names=FEATURE_NAMES)
            pred[te] = m.predict(X[te])
    return pred


def mcnemar(y, a, b):
    """Exact McNemar test on paired correctness (two-sided binomial)."""
    ca, cb = a == y, b == y
    n01, n10 = int(np.sum(ca & ~cb)), int(np.sum(~ca & cb))
    n = n01 + n10
    p = 1.0 if n == 0 else float(stats.binomtest(min(n01, n10), n, 0.5).pvalue)
    return {"a_only_correct": n01, "b_only_correct": n10, "p_value": p}


@app.command()
def main(out: Path = Path("results_research"), seed: int = 0) -> None:
    out.mkdir(exist_ok=True)
    result = {}
    for tag, path in [("raw", "data/benchmark.jsonl"), ("hardened", "data/benchmark_hard.jsonl")]:
        h = EvaluationHarness(path)
        h.load_and_extract()
        X, y = h._X, h._y
        rf = lambda: DriftGuardClassifier(backend="random_forest", n_estimators=100, random_state=seed)
        res = {"n": int(len(y)), "class_counts": np.bincount(y).tolist()}

        print(f"[{tag}] repeated 5x3 CV …", flush=True)
        res["models"] = {
            "DriftGuard (RF)": cv(rf, X, y, seed=seed),
            "DriftGuard (GBM)": cv(lambda: DriftGuardClassifier(backend="gradient_boosting", random_state=seed), X, y, seed=seed),
            "Logistic regression": cv(lambda: DriftGuardClassifier(backend="logistic", random_state=seed), X, y, seed=seed),
            "Rule-based baseline": cv(RuleBasedDetector, X, y, seed=seed),
            "Majority class": cv(lambda: DummyClassifier(strategy="most_frequent"), X, y, seed=seed),
        }
        print(f"[{tag}] feature-family ablation …", flush=True)
        abl = {"all features": res["models"]["DriftGuard (RF)"]}
        for fam, cols in FAMILIES.items():
            rest = [c for c in range(X.shape[1]) if c not in cols]
            abl[f"without {fam}"] = cv(rf, X, y, cols=rest, n_repeats=1, seed=seed)
            abl[f"only {fam}"] = cv(rf, X, y, cols=cols, n_repeats=1, seed=seed)
        res["ablation"] = abl

        print(f"[{tag}] McNemar …", flush=True)
        p_rf, p_rule = oof(rf, X, y, seed), oof(RuleBasedDetector, X, y, seed)
        res["mcnemar_rf_vs_rule"] = mcnemar(y, p_rf, p_rule)
        result[tag] = res

    (out / "research_metrics.json").write_text(json.dumps(result, indent=2))
    lines = []
    for tag, res in result.items():
        lines += [f"\n### {tag} benchmark (n={res['n']}, classes={res['class_counts']})\n",
                  "| Model | Accuracy | Macro-F1 | MCC | Attack TPR | FAR | MAR |", "|---|---|---|---|---|---|---|"]
        for name, m in res["models"].items():
            f = lambda k: f"{m[k]['mean']:.3f} ± {m[k]['ci95']:.3f}"
            lines.append(f"| {name} | {f('accuracy')} | {f('f1_macro')} | {f('mcc')} | {f('attack_tpr')} | {f('FAR')} | {f('MAR')} |")
        lines += ["\n| Ablation (RF) | Accuracy | Attack TPR | FAR |", "|---|---|---|---|"]
        for name, m in res["ablation"].items():
            lines.append(f"| {name} | {m['accuracy']['mean']:.3f} | {m['attack_tpr']['mean']:.3f} | {m['FAR']['mean']:.3f} |")
        mc = res["mcnemar_rf_vs_rule"]
        lines.append(f"\nMcNemar RF vs rule-based: RF-only-correct={mc['a_only_correct']}, "
                     f"rule-only-correct={mc['b_only_correct']}, p={mc['p_value']:.2e}")
    (out / "research_metrics.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    app()
