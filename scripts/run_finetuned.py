"""Fine-tuned transformer baseline vs the shipped hybrid (the comparison reviewers ask for first).

A small pretrained encoder (all-MiniLM-L6-v2 weights + a linear head) is fine-tuned end to end, on CPU, for
attack-vs-not on EXACTLY the text the hybrid's text models saw (benchmark train split, real-text training half,
llama3.2:1b training notes). It is evaluated on the same four held-out sets as scripts.run_hybrid and fused
with the shipped hybrid by unweighted mean. Scorers are fixed in advance and all are reported.

Usage:  python -m scripts.run_finetuned          Writes results_research/finetuned.{json,md}
"""

import copy
import json
import random
from pathlib import Path

import numpy as np
import torch
import typer
from sklearn.model_selection import train_test_split
from transformers import AutoModel, AutoTokenizer

from driftguard.benchmark.labelling import load_benchmark
from driftguard.detector.features import SemanticFeatureExtractor
from driftguard.detector.hybrid import load_detector
from scripts.run_hybrid import metrics
from scripts.run_real_text_controlled import _core, _cut, _fmt, dedupe, length_match
from scripts.run_real_text_eval import load_real

app = typer.Typer()
NAME = "sentence-transformers/all-MiniLM-L6-v2"
COMBOS = ["hybrid", "finetuned", "hybrid+finetuned"]


class Clf(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = AutoModel.from_pretrained(NAME)
        self.head = torch.nn.Linear(self.enc.config.hidden_size, 1)

    def forward(self, ids, mask):
        h = self.enc(input_ids=ids, attention_mask=mask).last_hidden_state
        m = mask.unsqueeze(-1).float()
        return self.head((h * m).sum(1) / m.sum(1)).squeeze(-1)


def _batches(tok, texts, bs, maxlen=96):
    for i in range(0, len(texts), bs):
        b = tok(texts[i:i + bs], padding=True, truncation=True, max_length=maxlen, return_tensors="pt")
        yield i, b["input_ids"], b["attention_mask"]


def predict(model, tok, texts):
    model.eval()
    out = []
    with torch.no_grad():
        for _, ids, mask in _batches(tok, list(texts), 128):
            out.append(torch.sigmoid(model(ids, mask)).numpy())
    return np.concatenate(out)


@app.command()
def main(
    benchmark: Path = Path("data/benchmark_hard.jsonl"),
    ext_dir: Path = Path("data/external"),
    notes: Path = Path("data/llm_notes.jsonl"),
    hand: Path = Path("data/hand_notes.tsv"),
    hybrid: Path = Path("models/classifier_hybrid.pkl"),
    out: Path = Path("results_research"),
    epochs: int = 3,
    seed: int = 42,
) -> None:
    torch.manual_seed(seed)
    entries = load_benchmark(benchmark)
    y_all = np.array([e.label_int for e in entries])
    tr, te = train_test_split(np.arange(len(entries)), test_size=0.2, random_state=seed, stratify=y_all)
    rows = [json.loads(line) for line in notes.read_text(encoding="utf-8").splitlines() if line]
    train_notes = [r for r in rows if r["split"] == "train" and r["generator"] == "llama3.2:1b"]
    atk42, ben42 = load_real(ext_dir, 600, random.Random(seed))
    h = len(atk42) // 2
    tr_texts = [entries[i].content for i in tr] + atk42[:h] + ben42[:h] + [r["text"] for r in train_notes]
    tr_y = np.array([int(y_all[i] == 0) for i in tr] + [1] * h + [0] * h + [int(r["label"] == 0) for r in train_notes],
                    dtype=np.float32)
    print(f"fine-tuning on {len(tr_texts)} rows ({int(tr_y.sum())} attack), {epochs} epochs", flush=True)

    tok = AutoTokenizer.from_pretrained(NAME)
    model = Clf()
    opt = torch.optim.AdamW(model.parameters(), lr=3e-5, weight_decay=0.01)
    pos_w = torch.tensor((len(tr_y) - tr_y.sum()) / max(tr_y.sum(), 1))
    lossf = torch.nn.BCEWithLogitsLoss(pos_weight=pos_w)
    rng = np.random.default_rng(seed)
    for ep in range(epochs):
        model.train()
        order = rng.permutation(len(tr_texts))
        texts = [tr_texts[i] for i in order]
        ys = torch.tensor(tr_y[order])
        total = 0.0
        for i, ids, mask in _batches(tok, texts, 32):
            loss = lossf(model(ids, mask), ys[i:i + 32])
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss) * len(ids)
        print(f"epoch {ep + 1}: train loss {total / len(texts):.4f}", flush=True)

    Path("models").mkdir(exist_ok=True)
    torch.save(model.state_dict(), "models/finetuned_minilm.pt")   # git-ignored; avoids retraining next time
    ext = SemanticFeatureExtractor()
    det = load_detector(hybrid)
    n_feat = len(det._feature_names)
    r2 = random.Random(0)

    def ctx(texts):
        res = []
        for t in texts:
            d = copy.deepcopy(r2.choice(entries))
            res.append(dict(content=t, provenance=d.provenance, write_history=d.write_history, session_id=d.session_id,
                            last_query=d.last_query, memory_entries=d.memory_entries))
        return res

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

    if hand.exists():       # set D: 100 notes written by hand-style prompting of a different model (label<TAB>text)
        pairs = [ln.split("\t", 1) for ln in hand.read_text(encoding="utf-8").splitlines() if "\t" in ln]
        texts = [t for _, t in pairs]
        sets["notes D (hand-style, 100 notes, 18 benign instruction-like)"] = (
            texts, ext.extract_batch(ctx(texts)), np.array([int(lab == "0") for lab, _ in pairs]))

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
    atk_set = set(atk)
    sets["real text, length-matched, unseen"] = (keep, ext.extract_batch(ctx(keep)),
                                                 np.array([int(t in atk_set) for t in keep]))

    result = {}
    for name, (texts, X, y_attack) in sets.items():
        p_h = det.predict_proba_text(X[:, :n_feat], texts)[:, 0]
        p_f = predict(model, tok, texts)
        p = {"hybrid": p_h, "finetuned": p_f, "hybrid+finetuned": (p_h + p_f) / 2}
        result[name] = {c: metrics(y_attack, p[c]) for c in COMBOS}
        print(name, {c: round(result[name][c]["auroc"], 3) for c in COMBOS}, flush=True)

    out.mkdir(exist_ok=True)
    (out / "finetuned.json").write_text(json.dumps(result, indent=2))
    lines = ["| Test set | Scorer | AUROC | Recall @0.5 | False alarm @0.5 | False alarm at 90 % recall |",
             "|---|---|---|---|---|---|"]
    for name, res in result.items():
        for c in COMBOS:
            m = res[c]
            lines.append(f"| {name} (n={m['n_attack']}+{m['n_benign']}) | {c} | {m['auroc']:.3f} | {m['recall']:.2f} | "
                         f"{m['false_alarm']:.2f} | {m['far_at_90_recall']:.2f} |")
    (out / "finetuned.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    app()
