"""Leave-one-group-out stacking of the three text scorers (hybrid_instr, finetuned_instr, protectai_deberta).

Why: on memory-write text our detectors catch attacks but over-flag generic chat prompts, while a generic
prompt-injection detector (ProtectAI DeBERTa) ranks generic prompts well but misses memory-poisoning notes at its
default threshold. Their probabilities are calibrated differently, so an unweighted mean is a poor fusion
(results_research/multisource.md). A 3-input logistic regression on the logits learns the weighting.

Protocol (no tuning on the test group): for each test group g, fit the stacker on the saved scores of every OTHER
group (each group weighted equally, classes balanced), then score g. Reported next to the best single scorer and the
mean fusion. Scores come from results_research/multisource_scores.json (python -m scripts.run_multisource).
Caveats: hybrid_instr / finetuned_instr saw data.hand_train.tsv, written after the first look at notes_D, so
notes_D is not fully independent for them; deepset/safeguard slices were seen by the shipped models.

Usage:  python -m scripts.run_stacking      Writes results_research/stacking.{json,md}
"""

import json
from pathlib import Path

import numpy as np
import typer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

app = typer.Typer()
INPUTS = ["hybrid_instr", "finetuned_instr", "protectai_deberta"]


def logit(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def recall_at_far(y, p, far):
    """Recall when the threshold is set so at most `far` of the benign items are flagged (threshold-free comparison)."""
    thr = np.quantile(p[y == 0], 1 - far)
    return float((p[y == 1] > thr).mean())


def feats(d):
    return np.stack([logit(d[k]) for k in INPUTS], axis=1)


@app.command()
def main(scores: Path = Path("results_research/multisource_scores.json"), out: Path = Path("results_research")) -> None:
    S = json.loads(scores.read_text())
    groups = list(S)
    res = {}
    for g in groups:
        X, y, w = [], [], []
        for h in groups:
            if h == g:
                continue
            Xh, yh = feats(S[h]), np.array(S[h]["y"])
            n = len(yh)
            wh = np.where(yh == 1, 0.5 / max(yh.sum(), 1), 0.5 / max((1 - yh).sum(), 1)) * n  # balanced within group
            X.append(Xh), y.append(yh), w.append(wh / n)                                      # groups weigh equally
        X, y, w = np.vstack(X), np.concatenate(y), np.concatenate(w)
        lr = LogisticRegression(C=1.0, max_iter=1000).fit(X, y, sample_weight=w)
        yt = np.array(S[g]["y"])
        p = lr.predict_proba(feats(S[g]))[:, 1]
        pred = p >= 0.5
        single = {k: roc_auc_score(yt, S[g][k]) for k in INPUTS}
        rec5 = {k: recall_at_far(yt, np.asarray(S[g][k]), 0.05) for k in INPUTS}
        rec5["stack"] = recall_at_far(yt, p, 0.05)
        mean = roc_auc_score(yt, np.mean([S[g][k] for k in INPUTS], axis=0))
        res[g] = {"stack_auroc": float(roc_auc_score(yt, p)), "stack_recall": float(pred[yt == 1].mean()),
                  "stack_false_alarm": float(pred[yt == 0].mean()), "best_single": max(single, key=single.get),
                  "best_single_auroc": float(max(single.values())), "mean_fusion_auroc": float(mean),
                  "recall_at_5pct_far": {k: float(v) for k, v in rec5.items()},
                  "weights": dict(zip(INPUTS, [round(float(c), 3) for c in lr.coef_[0]]))}
        print(g, {k: round(v, 3) if isinstance(v, float) else v for k, v in res[g].items() if k != "weights"}, flush=True)

    out.mkdir(exist_ok=True)
    (out / "stacking.json").write_text(json.dumps(res, indent=2))
    lines = ["| Test group | Stacked AUROC | Recall @0.5 | False alarm @0.5 | Best single (AUROC) | Mean fusion AUROC |",
             "|---|---|---|---|---|---|"]
    for g, v in res.items():
        lines.append(f"| {g} | {v['stack_auroc']:.3f} | {v['stack_recall']:.2f} | {v['stack_false_alarm']:.2f} | "
                     f"{v['best_single']} ({v['best_single_auroc']:.3f}) | {v['mean_fusion_auroc']:.3f} |")
    lines += ["", "Recall when at most 5 % of benign items are flagged (threshold set per group on its own benign items, so this "
              "compares ranking quality at a fixed false-alarm budget, not a deployable threshold):", "",
              "| Test group | hybrid_instr | finetuned_instr | protectai_deberta | stack |", "|---|---|---|---|---|"]
    for g, v in res.items():
        r = v["recall_at_5pct_far"]
        lines.append(f"| {g} | {r['hybrid_instr']:.2f} | {r['finetuned_instr']:.2f} | {r['protectai_deberta']:.2f} | {r['stack']:.2f} |")
    (out / "stacking.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    app()
