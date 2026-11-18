"""Train and save the hybrid detector: v3 feature classifier + TF-IDF LR + sentence-embedding LR.

The text models are trained on exactly the text the v3 model saw (so the masking used in scripts.run_hybrid stays
valid): the benchmark train split, the first half of the seed-42 real-text pools, and the llama3.2:1b training notes.
The result is a signed pickle (models/classifier_hybrid.pkl + .sig) that WritePipeline and the Guard API load like any
other model.

Usage:  python -m scripts.train_v3 && python -m scripts.train_hybrid
"""

import json
import random
from pathlib import Path

import numpy as np
import typer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from driftguard.benchmark.labelling import load_benchmark
from driftguard.detector.classifier import DriftGuardClassifier
from driftguard.detector.features import FeatureExtractor
from driftguard.detector.hybrid import HybridDetector
from scripts.run_real_text_eval import load_real

app = typer.Typer()


@app.command()
def main(
    benchmark: Path = Path("data/benchmark_hard.jsonl"),
    ext_dir: Path = Path("data/external"),
    notes: Path = Path("data/llm_notes.jsonl"),
    base_path: Path = Path("models/classifier_v3.pkl"),
    out: Path = Path("models/classifier_hybrid.pkl"),
    seed: int = 42,
    extra: Path | None = typer.Option(None, help="optional label<TAB>text file of extra training notes (text models only)"),
) -> None:
    entries = load_benchmark(benchmark)
    y_all = np.array([e.label_int for e in entries])
    tr, _ = train_test_split(np.arange(len(entries)), test_size=0.2, random_state=seed, stratify=y_all)
    rows = [json.loads(line) for line in notes.read_text(encoding="utf-8").splitlines() if line]
    train_notes = [r for r in rows if r["split"] == "train" and r["generator"] == "llama3.2:1b"]
    atk, ben = load_real(ext_dir, 600, random.Random(seed))
    h = len(atk) // 2

    texts = [entries[i].content for i in tr] + atk[:h] + ben[:h] + [r["text"] for r in train_notes]
    y = np.array([int(y_all[i] == 0) for i in tr] + [1] * h + [0] * h + [int(r["label"] == 0) for r in train_notes])
    if extra is not None:   # label 0 = attack; 1/2 = not attack
        pairs = [ln.split("\t", 1) for ln in extra.read_text(encoding="utf-8").splitlines() if "\t" in ln]
        texts += [t for _, t in pairs]
        y = np.concatenate([y, np.array([int(lab == "0") for lab, _ in pairs])])
    print(f"text-model training rows: {len(texts)} ({int(y.sum())} attack)", flush=True)

    tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    lr_tfidf = LogisticRegression(max_iter=2000, C=4.0, class_weight="balanced").fit(tfidf.fit_transform(texts), y)
    enc = FeatureExtractor().encoder
    emb = enc.encode(texts, normalize_embeddings=True, batch_size=64, show_progress_bar=False)
    lr_embed = LogisticRegression(max_iter=2000, C=4.0, class_weight="balanced").fit(emb, y)

    HybridDetector(DriftGuardClassifier.load(base_path), tfidf, lr_tfidf, lr_embed).save(out)
    print(f"saved {out} (+ .sig)")


if __name__ == "__main__":
    app()
