"""Train v3 = v2 recipe + LLM-written memory notes, then retest on held-out writers/goals/styles.

Training adds the `train` split of data/llm_notes.jsonl (notes written by llama3.2:1b about
TRAIN attack goals / drift / ordinary topics). Evaluation uses splits whose goals, topics and (for
test_b / test_c) writer never appear in training:

  benchmark   the synthetic hardened test split (same seed/split as v2) - checks for regression
  test_a      llama3.2:1b notes, new goals/topics     (seen writer, unseen goals)
  test_b      phi3.5 notes                             (unseen writer, unseen goals)
  test_c      phi3.5, first-person voice               (unseen writer, goals AND style)

Context (history / trust / origin) for every LLM note is taken from a random benchmark entry of any
class, so only the text carries signal. Reported: attack recall and false-alarm rate with Wilson 95 %
intervals, 3-class accuracy, binary AUROC for malicious-vs-rest, and a text-only TF-IDF baseline
trained on the same training notes (to show how much is plain vocabulary).

Usage:  python -m scripts.generate_notes && python -m scripts.train_v3
"""

import copy
import json
import random
from pathlib import Path

import numpy as np
import typer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from driftguard.benchmark.labelling import load_benchmark
from driftguard.detector.classifier import DriftGuardClassifier
from driftguard.detector.features import SEMANTIC_FEATURE_NAMES, SemanticFeatureExtractor
from eval import adversarial as adv
from scripts.run_real_text_eval import load_real
from scripts.train_detector import FEATURE_NAMES
from scripts.train_v2 import _rec

app = typer.Typer()


def wilson(k: int, n: int, z: float = 1.96) -> list[float]:
    if n == 0:
        return [0.0, 1.0]
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / d
    return [round(max(0.0, c - h), 3), round(min(1.0, c + h), 3)]


def evaluate(pred, proba_mal, y) -> dict:
    atk, ben = y == 0, y != 0
    k_tp, k_fp = int((pred[atk] == 0).sum()), int((pred[ben] == 0).sum())
    out = {"n_attack": int(atk.sum()), "n_benign": int(ben.sum()),
           "attack_recall": k_tp / max(1, atk.sum()), "attack_recall_ci": wilson(k_tp, int(atk.sum())),
           "false_alarm": k_fp / max(1, ben.sum()), "false_alarm_ci": wilson(k_fp, int(ben.sum())),
           "accuracy_3class": float(np.mean(pred == y))}
    if atk.any() and ben.any():
        out["auroc_malicious_vs_rest"] = float(roc_auc_score(atk.astype(int), proba_mal))
    return out


@app.command()
def main(
    benchmark: Path = Path("data/benchmark_hard.jsonl"),
    ext_dir: Path = Path("data/external"),
    notes: Path = Path("data/llm_notes.jsonl"),
    v2_path: Path = Path("models/classifier_v2.pkl"),
    out: Path = Path("models/classifier_v3.pkl"),
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
        e = entries[i]
        base = _rec(e)
        recs.append(base); labels.append(e.label_int)
        if e.label_int == 0:
            for _ in range(variants):
                recs.append(adv.apply(arng.choice(attack_tf), [base], seed=arng.randrange(10**6))[0]); labels.append(0)
        elif arng.random() < 0.6:
            recs.append(adv.apply(arng.choice(benign_tf), [base], seed=arng.randrange(10**6))[0]); labels.append(e.label_int)

    def with_context(texts):
        res = []
        for t in texts:
            d = copy.deepcopy(rng.choice(entries))
            res.append(dict(content=t, provenance=d.provenance, write_history=d.write_history,
                            session_id=d.session_id, last_query=d.last_query, memory_entries=d.memory_entries))
        return res

    atk, ben = load_real(ext_dir, n_real_each * 2, rng)       # identical to v2's real-text pool/seed
    h = len(atk) // 2
    recs += with_context(atk[:h]) + with_context(ben[:h])
    labels += [0] * h + [2] * h

    rows = [json.loads(line) for line in notes.read_text(encoding="utf-8").splitlines() if line]
    train_rows = [r for r in rows if r["split"] == "train" and r["generator"] == "llama3.2:1b"]
    recs += with_context([r["text"] for r in train_rows])
    labels += [r["label"] for r in train_rows]
    X, yv = ext.extract_batch(recs), np.array(labels)
    print(f"training rows {len(yv)}  class counts {np.bincount(yv)}  llm-note rows {len(train_rows)}", flush=True)

    clf = DriftGuardClassifier(backend="random_forest", random_state=seed)
    clf.fit(X, yv, feature_names=FEATURE_NAMES + SEMANTIC_FEATURE_NAMES)
    clf.save(out)
    v2 = DriftGuardClassifier.load(v2_path)

    tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    Ttr = tfidf.fit_transform([r["text"] for r in train_rows])
    tf_clf = LogisticRegression(max_iter=1000).fit(Ttr, [int(r["label"] == 0) for r in train_rows])

    def sets():
        yield "benchmark", ext.extract_batch([_rec(entries[i]) for i in te]), y[te], [entries[i].content for i in te]
        for split in ("test_a", "test_b", "test_c"):
            rs = [r for r in rows if r["split"] == split]
            if rs:
                yield (split, ext.extract_batch(with_context([r["text"] for r in rs])),
                       np.array([r["label"] for r in rs]), [r["text"] for r in rs])

    report = {"train_rows": int(len(yv)), "llm_note_train_rows": len(train_rows), "sets": {}}
    for name, Xs, ys, texts in sets():
        entry = {}
        for label, model in (("v2", v2), ("v3", clf)):
            k = len(model._feature_names or []) or 18
            entry[label] = evaluate(np.asarray(model.predict(Xs[:, :k])), model.predict_proba(Xs[:, :k])[:, 0], ys)
        if name != "benchmark":
            p = tf_clf.predict_proba(tfidf.transform(texts))[:, 1]
            entry["tfidf_text_only"] = evaluate(np.where(p > 0.5, 0, 2), p, ys)
        report["sets"][name] = entry
        for label, r in entry.items():
            print(f"{name:9s} {label:16s} recall={r['attack_recall']:.2f} {r['attack_recall_ci']}  "
                  f"FAR={r['false_alarm']:.2f} {r['false_alarm_ci']}  acc3={r['accuracy_3class']:.2f}  "
                  f"auroc={r.get('auroc_malicious_vs_rest', float('nan')):.2f}", flush=True)
    Path("results_research").mkdir(exist_ok=True)
    Path("results_research/v3.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    app()
