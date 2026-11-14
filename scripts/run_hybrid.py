"""Does fusing the feature detector (v3) with plain text classifiers beat either alone?

Text-only baselines (TF-IDF logistic regression; sentence-embedding logistic regression) matched or beat v3 on
the LLM-written note sets (paper section 5.7b). Here we evaluate score fusion. To avoid selecting a winner on
the test sets, the combinations are fixed in advance and ALL are reported:

  v3                      feature detector probability of 'malicious'
  tfidf                   TF-IDF (1-2 grams) logistic regression, attack vs not
  embed                   all-MiniLM-L6-v2 sentence embedding logistic regression, attack vs not
  v3+tfidf, v3+embed, v3+tfidf+embed     unweighted mean of the probabilities

Text models are trained on exactly the text v3 saw: benchmark train split, the real-text training half, and the
llama3.2:1b training notes. Test sets: benchmark test split; notes sets A (seen writer, new goals), B (unseen writer),
C (unseen writer + style); the length-matched real-text set with strings seen in training masked out.
Reported: AUROC (malicious vs rest), recall and false-alarm at the 0.5 threshold, and the false-alarm rate needed to
reach 90 % recall (threshold-free).

Usage:  python -m scripts.run_hybrid          Writes results_research/hybrid.{json,md}
"""

import copy
import json
import random
from pathlib import Path

import numpy as np
import typer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split

from driftguard.benchmark.labelling import load_benchmark
from driftguard.detector.classifier import DriftGuardClassifier
from driftguard.detector.features import FeatureExtractor, SemanticFeatureExtractor
from scripts.run_real_text_controlled import _core, _cut, _fmt, dedupe, length_match
from scripts.run_real_text_eval import load_real

app = typer.Typer()
COMBOS = ["v3", "tfidf", "embed", "v3+tfidf", "v3+embed", "v3+tfidf+embed"]


def far_at_recall(y_attack, p, target=0.9) -> float:
    fpr, tpr, _ = roc_curve(y_attack, p)
    return float(fpr[np.argmax(tpr >= target)])


def metrics(y_attack, p) -> dict:
    pred = p >= 0.5
    return {"auroc": float(roc_auc_score(y_attack, p)),
            "recall": float(np.mean(pred[y_attack == 1])), "false_alarm": float(np.mean(pred[y_attack == 0])),
            "far_at_90_recall": far_at_recall(y_attack, p), "n_attack": int(y_attack.sum()),
            "n_benign": int((1 - y_attack).sum())}


@app.command()
def main(
    benchmark: Path = Path("data/benchmark_hard.jsonl"),
    ext_dir: Path = Path("data/external"),
    notes: Path = Path("data/llm_notes.jsonl"),
    model: Path = Path("models/classifier_v3.pkl"),
    out: Path = Path("results_research"),
    seed: int = 42,
) -> None:
    entries = load_benchmark(benchmark)
    y_all = np.array([e.label_int for e in entries])
    tr, te = train_test_split(np.arange(len(entries)), test_size=0.2, random_state=seed, stratify=y_all)
    rows = [json.loads(line) for line in notes.read_text(encoding="utf-8").splitlines() if line]
    train_notes = [r for r in rows if r["split"] == "train" and r["generator"] == "llama3.2:1b"]
    atk42, ben42 = load_real(ext_dir, 600, random.Random(seed))
    h = len(atk42) // 2

    tr_texts = ([entries[i].content for i in tr] + atk42[:h] + ben42[:h] + [r["text"] for r in train_notes])
    tr_y = np.array([int(y_all[i] == 0) for i in tr] + [1] * h + [0] * h + [int(r["label"] == 0) for r in train_notes])
    ext = SemanticFeatureExtractor()
    enc = FeatureExtractor().encoder
    embed = lambda texts: enc.encode(list(texts), normalize_embeddings=True, batch_size=64, show_progress_bar=False)

    tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    lr_tfidf = LogisticRegression(max_iter=2000, C=4.0, class_weight="balanced").fit(tfidf.fit_transform(tr_texts), tr_y)
    lr_embed = LogisticRegression(max_iter=2000, C=4.0, class_weight="balanced").fit(embed(tr_texts), tr_y)
    v3 = DriftGuardClassifier.load(model)

    rng = random.Random(0)

    def ctx(texts):
        res = []
        for t in texts:
            d = copy.deepcopy(rng.choice(entries))
            res.append(dict(content=t, provenance=d.provenance, write_history=d.write_history, session_id=d.session_id,
                            last_query=d.last_query, memory_entries=d.memory_entries))
        return res

    def scores(texts, X):
        p = {"v3": v3.predict_proba(X[:, :len(v3._feature_names)])[:, 0],
             "tfidf": lr_tfidf.predict_proba(tfidf.transform(texts))[:, 1],
             "embed": lr_embed.predict_proba(embed(texts))[:, 1]}
        p["v3+tfidf"] = (p["v3"] + p["tfidf"]) / 2
        p["v3+embed"] = (p["v3"] + p["embed"]) / 2
        p["v3+tfidf+embed"] = (p["v3"] + p["tfidf"] + p["embed"]) / 3
        return p

    sets = {}
    te_recs = [dict(content=entries[i].content, provenance=entries[i].provenance, write_history=entries[i].write_history,
                    session_id=entries[i].session_id, last_query=entries[i].last_query,
                    memory_entries=entries[i].memory_entries) for i in te]
    sets["benchmark test"] = ([r["content"] for r in te_recs], ext.extract_batch(te_recs), (y_all[te] == 0).astype(int))
    for split, label in (("test_a", "notes A (seen writer, new goals)"), ("test_b", "notes B (unseen writer)"),
                         ("test_c", "notes C (unseen writer + style)")):
        rs = [r for r in rows if r["split"] == split]
        texts = [r["text"] for r in rs]
        sets[label] = (texts, ext.extract_batch(ctx(texts)), np.array([int(r["label"] == 0) for r in rs]))

    # controlled real-text set; drop strings that were in the training half (v3 and the text models saw them)
    a_raw, b_raw = load_real(ext_dir, 100000, random.Random(0))
    atk = [t for t in dedupe(sorted({_fmt(t) for t in a_raw})) if len(t) > 26]
    ben = dedupe(sorted({_fmt(t) for t in b_raw}))
    atk, ben = length_match(atk, ben, random.Random(0))
    seen = [_cut(_core(s)).lower() for s in atk42[:h] + ben42[:h]]
    seen_set = set(seen)

    def is_seen(t):
        c = _core(t).lower()
        return c in seen_set or any(s.startswith(c) or c.startswith(s) for s in seen)

    keep = [t for t in atk + ben if not is_seen(t)]
    labels = np.array([int(t in set(atk)) for t in keep])
    sets["real text, length-matched, unseen"] = (keep, ext.extract_batch(ctx(keep)), labels)

    result = {}
    for name, (texts, X, y_attack) in sets.items():
        p = scores(texts, X)
        result[name] = {c: metrics(y_attack, p[c]) for c in COMBOS}
        print(name, {c: round(result[name][c]["auroc"], 3) for c in COMBOS}, flush=True)

    out.mkdir(exist_ok=True)
    (out / "hybrid.json").write_text(json.dumps(result, indent=2))
    lines = ["| Test set | Scorer | AUROC | Recall @0.5 | False alarm @0.5 | False alarm at 90 % recall |", "|---|---|---|---|---|---|"]
    for name, res in result.items():
        for c in COMBOS:
            m = res[c]
            lines.append(f"| {name} (n={m['n_attack']}+{m['n_benign']}) | {c} | {m['auroc']:.3f} | {m['recall']:.2f} | "
                         f"{m['false_alarm']:.2f} | {m['far_at_90_recall']:.2f} |")
    (out / "hybrid.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    app()
