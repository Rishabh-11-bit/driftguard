"""Train the v2 detector: semantic features + adversarial augmentation + real injection/persona text.

  features   18 base + 7 semantic attack-intent anchors (SemanticFeatureExtractor)
  data       hardened synthetic benchmark (80 % train split)
             + adversarial augmentation (evasion variants of attacks, matching perturbations on benign)
             + REAL text: injection strings as class 0, persona facts as class 2, with class-independent
               contexts (half of each pool for training; the other half is held out for evaluation)

Usage:  python -m scripts.train_v2        (after scripts.download_external)
"""

import copy
import json
import random
from pathlib import Path

import numpy as np
import typer
from sklearn.model_selection import train_test_split

from driftguard.benchmark.labelling import load_benchmark
from driftguard.detector.classifier import DriftGuardClassifier
from driftguard.detector.features import SEMANTIC_FEATURE_NAMES, SemanticFeatureExtractor
from eval import adversarial as adv
from eval.metrics import compute_all_metrics
from scripts.run_real_text_eval import load_real
from scripts.train_detector import FEATURE_NAMES

app = typer.Typer()


def _rec(e):
    return dict(content=e.content, provenance=e.provenance, write_history=e.write_history,
                session_id=e.session_id, last_query=e.last_query, memory_entries=e.memory_entries)


@app.command()
def main(
    benchmark: Path = Path("data/benchmark_hard.jsonl"),
    ext_dir: Path = Path("data/external"),
    out: Path = Path("models/classifier_v2.pkl"),
    variants: int = 3,
    n_real_each: int = 300,
    seed: int = 42,
) -> None:
    rng = random.Random(seed)
    entries = load_benchmark(benchmark)
    y = np.array([e.label_int for e in entries])
    tr, te = train_test_split(np.arange(len(entries)), test_size=0.2, random_state=seed, stratify=y)
    ext = SemanticFeatureExtractor()
    hist = [entries[i].write_history for i in tr if entries[i].label_int == 2]
    attack_tf = [adv.benign_padding, adv.synonym_swap, adv.origin_laundering, adv.make_history_camouflage(hist),
                 adv.combo(hist), adv.zero_width, adv.homoglyph]
    benign_tf = [adv.benign_padding, adv.synonym_swap]

    recs, labels = [], []
    arng = random.Random(seed + 1000)
    for i in tr:
        e = entries[i]; base = _rec(e)
        recs.append(base); labels.append(e.label_int)
        if e.label_int == 0:
            for _ in range(variants):
                recs.append(adv.apply(arng.choice(attack_tf), [base], seed=arng.randrange(10**6))[0]); labels.append(0)
        elif arng.random() < 0.6:
            recs.append(adv.apply(arng.choice(benign_tf), [base], seed=arng.randrange(10**6))[0]); labels.append(e.label_int)

    atk, ben = load_real(ext_dir, n_real_each * 2, rng)
    h = len(atk) // 2

    def real_recs(texts):
        out_ = []
        for t in texts:
            d = copy.deepcopy(rng.choice(entries))
            out_.append(dict(content=t, provenance=d.provenance, write_history=d.write_history,
                             session_id=d.session_id, last_query=d.last_query, memory_entries=d.memory_entries))
        return out_

    recs += real_recs(atk[:h]) + real_recs(ben[:h])
    labels += [0] * h + [2] * h
    X, yv = ext.extract_batch(recs), np.array(labels)
    print(f"training rows {len(yv)} class counts {np.bincount(yv)} (real {2 * h})", flush=True)
    clf = DriftGuardClassifier(backend="random_forest", random_state=seed)
    clf.fit(X, yv, feature_names=FEATURE_NAMES + SEMANTIC_FEATURE_NAMES)
    clf.save(out)

    Xt = ext.extract_batch([_rec(entries[i]) for i in te])
    m = compute_all_metrics(y[te], clf.predict(Xt), clf.predict_proba(Xt))
    Xr = ext.extract_batch(real_recs(atk[h:]) + real_recs(ben[h:]))
    yr = np.array([0] * (len(atk) - h) + [2] * (len(ben) - h))
    pr = np.asarray(clf.predict(Xr))
    result = {
        "clean_test": {"accuracy": m["accuracy"], "FAR": m["FAR"], "MAR": m["MAR"], "f1_macro": m["f1_macro"]},
        "real_test": {"tpr": float(np.mean(pr[yr == 0] == 0)), "far": float(np.mean(pr[yr != 0] == 0)),
                      "n_attack": int((yr == 0).sum()), "n_benign": int((yr != 0).sum())},
    }
    Path("results_research").mkdir(exist_ok=True)
    Path("results_research/v2.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    app()
