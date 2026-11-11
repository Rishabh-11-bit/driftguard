"""Real-text evaluation: does the detector work on text that none of our generators wrote?

Attacks : real prompt-injection strings (deepset/prompt-injections, xTRam1/safe-guard-prompt-injection,
          label == 1), truncated to 300 characters.
Benign  : real crowd-written persona facts (Cynaptics/persona-chat persona_b lines), prefixed "User: ".
Context : non-text fields (history, trust, queries, memories) come from RANDOM hardened-benchmark
          entries (class-independent), so only the TEXT carries signal.

Caveats stated up front: injection strings are not memory writes (they are prompts), persona
facts are first-person chat facts, and there is no real "benign drift" class here, so this is a
binary attack-vs-benign test of text-feature transfer, not of the full three-class task.

Two parts:
  zero_shot  - detectors trained on synthetic data, scored on real text with no retraining
  cv         - 5-fold CV on real text (text features + random context), showing learnability

Usage:  python -m scripts.run_real_text_eval          (after scripts.download_external)
"""

import copy
import json
import random
from pathlib import Path

import numpy as np
import typer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from driftguard.benchmark.labelling import load_benchmark
from driftguard.detector.classifier import DriftGuardClassifier
from driftguard.detector.features import SemanticFeatureExtractor
from eval.baselines.rule_based import RuleBasedDetector
from scripts.train_detector import FEATURE_NAMES

app = typer.Typer()


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_real(ext_dir: Path, n_each: int, rng: random.Random):
    atk = []
    for name in ("deepset_prompt_injections_train", "deepset_prompt_injections_test", "safeguard_prompt_injection_test"):
        atk += [r["text"].strip()[:300] for r in _jsonl(ext_dir / f"{name}.jsonl") if r["label"] == 1]
    atk = sorted({a for a in atk if len(a) > 20})
    facts = set()
    for r in _jsonl(ext_dir / "persona_chat.jsonl"):
        for s in r["persona_b"]:
            s = s.strip()
            if 15 < len(s) < 200:
                facts.add("User: " + s[0].upper() + s[1:])
    facts = sorted(facts)
    rng.shuffle(atk), rng.shuffle(facts)
    n = min(n_each, len(atk), len(facts))
    return atk[:n], facts[:n]


def scores(y, pred, proba):
    atk = y == 0
    return {"attack_tpr": float(np.mean(pred[atk] == 0)), "false_alarm": float(np.mean(pred[~atk] == 0)),
            "auroc": float(roc_auc_score(atk.astype(int), proba)) if proba is not None else None}


@app.command()
def main(
    ext_dir: Path = Path("data/external"),
    benchmark: Path = Path("data/benchmark_hard.jsonl"),
    models: list[Path] = [Path("models/classifier.pkl"), Path("models/classifier_hard.pkl"),
                          Path("models/classifier_robust.pkl")],
    n_each: int = 400,
    out: Path = Path("results_research"),
    seed: int = 0,
) -> None:
    rng = random.Random(seed)
    atk, benign = load_real(ext_dir, n_each, rng)
    texts = atk + benign
    y = np.array([0] * len(atk) + [2] * len(benign))
    entries = load_benchmark(benchmark)
    recs = []
    for t in texts:
        d = copy.deepcopy(rng.choice(entries))
        recs.append(dict(content=t, provenance=d.provenance, write_history=d.write_history,
                         session_id=d.session_id, last_query=d.last_query, memory_entries=d.memory_entries))
    X = SemanticFeatureExtractor().extract_batch(recs)      # 25 columns; models take their own prefix
    print(f"real attacks={len(atk)}  real benign={len(benign)}")

    result = {"n_attack": len(atk), "n_benign": len(benign), "zero_shot": {}, "cv": {}}
    dets = {"rule-based": RuleBasedDetector()}
    for m in models:
        if m.exists():
            dets[m.stem] = DriftGuardClassifier.load(m)
    # v2 (scripts/train_v2.py) trains on the first half of a seed-42 shuffle of the same pools;
    # exclude exactly those strings when scoring v2 so its numbers stay held-out.
    a42, b42 = load_real(ext_dir, 600, random.Random(42))
    h42 = len(a42) // 2
    v2_seen = set(a42[:h42]) | set(b42[:h42])
    unseen = np.array([t not in v2_seen for t in texts])
    for name, d in dets.items():
        k = len(getattr(d, "_feature_names", None) or []) or 18
        pred = np.asarray(d.predict(X[:, :k]))
        proba = d.predict_proba(X[:, :k])[:, 0] if hasattr(d, "predict_proba") else None
        m = unseen if name == "classifier_v2" else np.ones(len(y), bool)
        result["zero_shot"][name] = scores(y[m], pred[m], proba[m] if proba is not None else None)
        result["zero_shot"][name]["n_scored"] = int(m.sum())
        print("zero-shot", name, {k: round(v, 3) for k, v in result["zero_shot"][name].items() if v is not None}, flush=True)

    # learnability on real text: text + random context, binary labels remapped to {0, 2}
    folds = []
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(X, y):
        m = DriftGuardClassifier(backend="random_forest", n_estimators=100, random_state=seed)
        m.fit(X[tr], y[tr], feature_names=FEATURE_NAMES)
        folds.append(scores(y[te], np.asarray(m.predict(X[te])), m.predict_proba(X[te])[:, 0]))
    result["cv"]["random_forest on real text"] = {k: float(np.mean([f[k] for f in folds])) for k in folds[0]}
    print("cv", {k: round(v, 3) for k, v in result["cv"]["random_forest on real text"].items()})
    out.mkdir(exist_ok=True)
    (out / "real_text.json").write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    app()
