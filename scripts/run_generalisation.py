"""Does semantic-intent featurisation and real-injection training data improve generalisation?

Settings compared (all random-forest, hardened benchmark, class-independent contexts for real text):
  base18            the original 18 features
  sem25             + 7 semantic attack-intent anchor features
  base18+real       base features, training augmented with REAL injection / persona text
  sem25+real        both

Measured:
  lofo_<family>     leave-one-attack-family-out detection (train without the family, test on it)
  real_tpr / real_far   on a held-out SPLIT of real text (train/test disjoint strings)
  cv_accuracy       5-fold CV accuracy on the synthetic hardened benchmark (regression check)

Real text: attacks = real injection strings (label 1 in the downloaded sets), benign = persona facts.
They are split 50/50 into a real-train pool (used only when "+real") and a real-test pool (always
used for evaluation, never trained on).

Usage:  python -m scripts.run_generalisation
"""

import copy
import json
import random
from pathlib import Path

import numpy as np
import typer
from sklearn.model_selection import StratifiedKFold

from driftguard.benchmark.labelling import load_benchmark
from driftguard.detector.classifier import DriftGuardClassifier
from driftguard.detector.features import (
    FeatureExtractor, SemanticFeatureExtractor, SEMANTIC_FEATURE_NAMES,
)
from scripts.run_real_text_eval import load_real
from scripts.train_detector import FEATURE_NAMES

app = typer.Typer()


def _recs(texts, entries, rng):
    out = []
    for t in texts:
        d = copy.deepcopy(rng.choice(entries))
        out.append(dict(content=t, provenance=d.provenance, write_history=d.write_history,
                        session_id=d.session_id, last_query=d.last_query, memory_entries=d.memory_entries))
    return out


@app.command()
def main(
    benchmark: Path = Path("data/benchmark_hard.jsonl"),
    ext_dir: Path = Path("data/external"),
    out: Path = Path("results_research"),
    n_real_each: int = 300,
    seed: int = 0,
) -> None:
    rng = random.Random(seed)
    entries = load_benchmark(benchmark)
    y = np.array([e.label_int for e in entries])
    fam = np.array([e.attack_family for e in entries])
    sem = SemanticFeatureExtractor()
    recs = [dict(content=e.content, provenance=e.provenance, write_history=e.write_history,
                 session_id=e.session_id, last_query=e.last_query, memory_entries=e.memory_entries) for e in entries]
    X25 = sem.extract_batch(recs)
    X18 = X25[:, :18]

    atk, ben = load_real(ext_dir, n_real_each * 2, rng)
    h = len(atk) // 2
    real = {
        "train": (atk[:h], ben[:h]), "test": (atk[h:], ben[h:]),
    }
    def real_X(texts):
        return sem.extract_batch(_recs(texts, entries, rng))
    R = {}
    for split, (a, b) in real.items():
        R[split] = (real_X(a + b), np.array([0] * len(a) + [2] * len(b)))
    print(f"real train={len(R['train'][1])} test={len(R['test'][1])}", flush=True)

    names18, names25 = FEATURE_NAMES, FEATURE_NAMES + SEMANTIC_FEATURE_NAMES
    settings = {"base18": (18, False), "sem25": (25, False), "base18+real": (18, True), "sem25+real": (25, True)}
    result = {}
    for sname, (dim, use_real) in settings.items():
        cols = list(range(dim))
        names = names25 if dim == 25 else names18

        def fit(Xtr, ytr):
            if use_real:
                Xtr = np.vstack([Xtr, R["train"][0][:, cols]])
                ytr = np.concatenate([ytr, R["train"][1]])
            m = DriftGuardClassifier(backend="random_forest", n_estimators=100, random_state=seed)
            m.fit(Xtr, ytr, feature_names=names)
            return m

        res = {"lofo": {}}
        for f in sorted(set(fam[y == 0])):
            te = fam == f
            m = fit(X25[~te][:, cols], y[~te])
            res["lofo"][f] = float(np.mean(m.predict(X25[te][:, cols]) == 0))
        res["lofo_mean"] = float(np.mean(list(res["lofo"].values())))
        m = fit(X25[:, cols], y)
        Xt, yt = R["test"]
        pred = np.asarray(m.predict(Xt[:, cols]))
        res["real_tpr"] = float(np.mean(pred[yt == 0] == 0))
        res["real_far"] = float(np.mean(pred[yt != 0] == 0))
        accs = []
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(X25, y):
            mm = fit(X25[tr][:, cols], y[tr])
            accs.append(float(np.mean(mm.predict(X25[te][:, cols]) == y[te])))
        res["cv_accuracy"] = float(np.mean(accs))
        result[sname] = res
        print(sname, json.dumps({k: (round(v, 3) if isinstance(v, float) else {a: round(b, 3) for a, b in v.items()})
                                 for k, v in res.items()}), flush=True)

    out.mkdir(exist_ok=True)
    (out / "generalisation.json").write_text(json.dumps(result, indent=2))
    fams = sorted(next(iter(result.values()))["lofo"])
    lines = ["| Setting | " + " | ".join(f"LOFO {f}" for f in fams) + " | LOFO mean | real TPR | real FAR | CV acc |",
             "|---|" + "---|" * (len(fams) + 4)]
    for s, r in result.items():
        lines.append(f"| {s} | " + " | ".join(f"{r['lofo'][f]:.2f}" for f in fams) +
                     f" | {r['lofo_mean']:.2f} | {r['real_tpr']:.2f} | {r['real_far']:.2f} | {r['cv_accuracy']:.3f} |")
    (out / "generalisation.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    app()
