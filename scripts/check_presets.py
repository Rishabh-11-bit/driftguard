"""Score every demo-UI preset with each detector and compare to the label implied by its name.

Preset names start with a marker: "Ordinary" -> class 2, "Benign drift" -> class 1, "Attack" -> class 0.
Uses the same seed memories, daily write history and feature extractor as the Streamlit demo, so the output
shows exactly what a presenter would see.

Usage:  python -m scripts.check_presets        Writes results_research/presets.json
"""

import json
from pathlib import Path

import typer

from app.demo_data import PRESETS, SEED_MEMORIES, daily_history
from driftguard.detector.classifier import DriftGuardClassifier
from driftguard.detector.features import SemanticFeatureExtractor
from driftguard.provenance.schema import make_provenance

app = typer.Typer()
NAMES = {0: "malicious", 1: "drift", 2: "ordinary"}


def expected(name: str) -> int | None:
    if "Attack" in name:
        return 0
    if "Benign drift" in name:
        return 1
    if "Ordinary" in name:
        return 2
    return None


@app.command()
def main(
    models: list[Path] = [Path("models/classifier_hard.pkl"), Path("models/classifier_robust.pkl"),
                          Path("models/classifier_v2.pkl"), Path("models/classifier_v3.pkl")],
    out: Path = Path("results_research/presets.json"),
) -> None:
    ext = SemanticFeatureExtractor()
    hist = daily_history()
    dets = {m.stem: DriftGuardClassifier.load(m) for m in models if m.exists()}
    rows, summary = [], {k: {"correct": 0, "n": 0, "attack_missed": 0, "benign_quarantined": 0} for k in dets}
    for name, vals in PRESETS.items():
        exp = expected(name)
        if vals is None or exp is None:
            continue
        origin, query, text = vals
        prov = make_provenance(origin=origin, session_id="demo", triggering_text=query)
        x = ext.extract(content=text, provenance=prov, write_history=hist, session_id="demo",
                        last_query=query, user_memory_entries=SEED_MEMORIES)
        row = {"preset": name, "expected": NAMES[exp]}
        for k, d in dets.items():
            n_feat = len(d._feature_names or []) or 18
            pred, proba = d.predict_one(x[:n_feat])
            row[k] = NAMES[int(pred)]
            s = summary[k]
            s["n"] += 1
            s["correct"] += int(pred) == exp
            s["attack_missed"] += exp == 0 and int(pred) != 0
            s["benign_quarantined"] += exp != 0 and int(pred) == 0
        rows.append(row)
    for k, s in summary.items():
        print(f"{k:20s} correct {s['correct']}/{s['n']}  attacks missed {s['attack_missed']}  "
              f"benign quarantined {s['benign_quarantined']}")
    for r in rows:
        bad = [k for k in dets if r[k] != r["expected"]]
        if bad:
            print(f"  {r['preset'][:58]:58s} expected {r['expected']:9s} wrong: " + ", ".join(f"{k}={r[k]}" for k in bad))
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({"summary": summary, "rows": rows}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    app()
