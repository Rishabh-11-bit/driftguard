"""Train an evasion-robust detector via adversarial augmentation.

Malicious training writes are augmented with evasion variants (padding, synonym
swap, origin laundering, history camouflage, the combined attacker, zero-width,
homoglyph). Benign writes get the *same* padding / synonym perturbations so the
perturbation itself does not become a label cue.

Usage:  python -m scripts.train_robust
Then:   python -m scripts.run_adversarial --model models/classifier_robust.pkl --out results_security_robust
"""

import random
from pathlib import Path

import numpy as np
import typer
from sklearn.model_selection import train_test_split

from driftguard.benchmark.labelling import load_benchmark
from driftguard.detector.classifier import DriftGuardClassifier
from driftguard.detector.features import FeatureExtractor
from eval import adversarial as adv
from eval.metrics import compute_all_metrics
from scripts.train_detector import FEATURE_NAMES

app = typer.Typer()


def _rec(e):
    return dict(content=e.content, provenance=e.provenance, write_history=e.write_history,
                session_id=e.session_id, last_query=e.last_query, memory_entries=e.memory_entries)


@app.command()
def main(
    benchmark: Path = Path("data/benchmark_hard.jsonl"),
    out: Path = Path("models/classifier_robust.pkl"),
    variants: int = 3,
    seed: int = 42,
) -> None:
    entries = load_benchmark(benchmark)
    y = np.array([e.label_int for e in entries])
    tr, te = train_test_split(np.arange(len(entries)), test_size=0.2, random_state=seed, stratify=y)
    ext = FeatureExtractor()
    benign_hist = [entries[i].write_history for i in tr if entries[i].label_int == 2]

    attack_tf = [adv.benign_padding, adv.synonym_swap, adv.origin_laundering,
                 adv.make_history_camouflage(benign_hist), adv.combo(benign_hist),
                 adv.zero_width, adv.homoglyph]
    benign_tf = [adv.benign_padding, adv.synonym_swap]

    rng = random.Random(seed + 1000)  # different stream from the evaluation transforms (seed 0)
    recs, labels = [], []
    for i in tr:
        e = entries[i]
        base = _rec(e)
        recs.append(base); labels.append(e.label_int)
        if e.label_int == 0:
            for _ in range(variants):
                recs.append(adv.apply(rng.choice(attack_tf), [base], seed=rng.randrange(10**6))[0])
                labels.append(0)
        elif rng.random() < 0.6:
            recs.append(adv.apply(rng.choice(benign_tf), [base], seed=rng.randrange(10**6))[0])
            labels.append(e.label_int)

    X = ext.extract_batch(recs)
    yv = np.array(labels)
    print(f"training rows: {len(yv)}  class counts: {np.bincount(yv)}")
    clf = DriftGuardClassifier(backend="random_forest", random_state=seed)
    clf.fit(X, yv, feature_names=FEATURE_NAMES)
    clf.save(out)

    Xt = ext.extract_batch([_rec(entries[i]) for i in te])
    yt = y[te]
    m = compute_all_metrics(yt, clf.predict(Xt), clf.predict_proba(Xt))
    print(f"clean test  accuracy={m['accuracy']:.4f}  FAR={m['FAR']:.4f}  MAR={m['MAR']:.4f}")


if __name__ == "__main__":
    app()
