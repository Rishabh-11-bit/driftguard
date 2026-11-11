"""Leave-one-evasion-out: train with adversarial augmentation EXCLUDING one evasion
technique, then test on that unseen technique. Measures whether robustness transfers
to evasions the detector never saw.

Usage:  python -m scripts.run_transform_holdout
"""

import json
import random
from pathlib import Path

import numpy as np
import typer
from sklearn.model_selection import train_test_split

from driftguard.benchmark.labelling import load_benchmark
from driftguard.detector.classifier import DriftGuardClassifier
from driftguard.detector.features import FeatureExtractor
from eval import adversarial as adv

app = typer.Typer()


def _rec(e):
    return dict(content=e.content, provenance=e.provenance, write_history=e.write_history,
                session_id=e.session_id, last_query=e.last_query, memory_entries=e.memory_entries)


@app.command()
def main(benchmark: Path = Path("data/benchmark_hard.jsonl"),
         out: Path = Path("results_security_robust"), variants: int = 3, seed: int = 42) -> None:
    out.mkdir(exist_ok=True)
    entries = load_benchmark(benchmark)
    y = np.array([e.label_int for e in entries])
    tr, te = train_test_split(np.arange(len(entries)), test_size=0.2, random_state=seed, stratify=y)
    ext = FeatureExtractor()
    hist = [entries[i].write_history for i in tr if entries[i].label_int == 2]
    all_tf = {
        "benign padding": adv.benign_padding, "synonym swap": adv.synonym_swap,
        "origin laundering": adv.origin_laundering, "history camouflage": adv.make_history_camouflage(hist),
        "COMBINED attacker": adv.combo(hist), "zero-width": adv.zero_width, "homoglyph": adv.homoglyph,
    }
    benign_tf = [adv.benign_padding, adv.synonym_swap]
    test_attacks = [_rec(entries[i]) for i in te if entries[i].label_int == 0]

    results = {}
    for held in all_tf:
        # excluding "benign padding" also excludes the combined attacker, which contains it
        banned = {held} | ({"COMBINED attacker"} if held in ("benign padding", "synonym swap",
                           "origin laundering", "history camouflage") else set())
        pool = [f for n, f in all_tf.items() if n not in banned]
        bpool = [f for f in benign_tf if f is not all_tf.get(held)]
        rng = random.Random(seed + 1000)
        recs, labels = [], []
        for i in tr:
            e = entries[i]; base = _rec(e)
            recs.append(base); labels.append(e.label_int)
            if e.label_int == 0:
                for _ in range(variants):
                    recs.append(adv.apply(rng.choice(pool), [base], seed=rng.randrange(10**6))[0]); labels.append(0)
            elif bpool and rng.random() < 0.6:
                recs.append(adv.apply(rng.choice(bpool), [base], seed=rng.randrange(10**6))[0]); labels.append(e.label_int)
        clf = DriftGuardClassifier(backend="random_forest", random_state=seed)
        clf.fit(ext.extract_batch(recs), np.array(labels))
        tpr = float(np.mean(clf.predict(ext.extract_batch(adv.apply(all_tf[held], test_attacks))) == 0))
        results[held] = tpr
        print(f"unseen evasion = {held:20s} attack TPR = {tpr:.2f}", flush=True)
    (out / "leave_one_evasion_out.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    app()
