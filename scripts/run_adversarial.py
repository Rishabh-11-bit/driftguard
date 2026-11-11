"""Run the adaptive-attacker evaluation and write results_security/adversarial.{json,md}.

Usage:  python -m scripts.run_adversarial --model models/classifier_hard.pkl
"""

import json
from pathlib import Path

import numpy as np
import typer
from sklearn.model_selection import train_test_split

from driftguard.benchmark.labelling import load_benchmark
from driftguard.detector.classifier import DriftGuardClassifier
from driftguard.detector.features import FeatureExtractor
from driftguard.security import normalise_text
from eval import adversarial as adv
from eval.baselines.rule_based import RuleBasedDetector
from eval.extended_metrics import bootstrap_ci

app = typer.Typer()


def _records(entries):
    return [dict(content=e.content, provenance=e.provenance, write_history=e.write_history,
                 session_id=e.session_id, last_query=e.last_query, memory_entries=e.memory_entries)
            for e in entries]


@app.command()
def main(
    benchmark: Path = Path("data/benchmark_hard.jsonl"),
    held_out: Path = Path("data/held_out_hard.jsonl"),
    model: Path = Path("models/classifier_hard.pkl"),
    out: Path = Path("results_security"),
    seed: int = 42,
) -> None:
    out.mkdir(exist_ok=True)
    clf = DriftGuardClassifier.load(model)
    rule = RuleBasedDetector()
    ext = FeatureExtractor()

    entries = load_benchmark(benchmark)
    y = np.array([e.label_int for e in entries])
    _, te = train_test_split(np.arange(len(entries)), test_size=0.2, random_state=seed, stratify=y)
    test_attacks = [entries[i] for i in te if entries[i].label_int == 0]
    held = [e for e in load_benchmark(held_out) if e.label_int == 0]
    benign_hist = [e.write_history for e in entries if e.label_int == 2]

    transforms = {
        "none (clean)": None,
        "zero-width chars": adv.zero_width,
        "full-width forms": adv.fullwidth,
        "cross-script homoglyphs": adv.homoglyph,
        "benign padding": adv.benign_padding,
        "synonym swap": adv.synonym_swap,
        "origin laundering (claims user_turn)": adv.origin_laundering,
        "history camouflage": adv.make_history_camouflage(benign_hist),
        "COMBINED adaptive attacker": adv.combo(benign_hist),
    }
    sets = {"in-distribution attacks": test_attacks, "held-out attack families": held}

    def tpr(detector, recs, normalise):
        if normalise:
            recs = [dict(r, content=normalise_text(r["content"]), last_query=normalise_text(r["last_query"]))
                    for r in recs]
        pred = detector.predict(ext.extract_batch(recs))
        hit = (pred == 0).astype(int)
        # bootstrap the share of attacks caught (truth is all class 0)
        lo, hi = bootstrap_ci(np.zeros(len(hit), int), 2 * (1 - hit),
                              lambda t, p: float(np.mean(p == 0)), n_boot=500, seed=1)
        return float(hit.mean()), (float(lo), float(hi))

    rows = []
    for sname, ents in sets.items():
        base = _records(ents)
        for tname, fn in transforms.items():
            recs = base if fn is None else adv.apply(fn, base)
            row = {"set": sname, "transform": tname, "n": len(recs)}
            for det_name, det, norm in [("DriftGuard (no sanitiser)", clf, False),
                                        ("DriftGuard (+sanitiser)", clf, True),
                                        ("Rule-based", rule, False)]:
                v, ci = tpr(det, recs, norm)
                row[det_name] = {"attack_tpr": v, "ci95": ci}
            rows.append(row)
            print(f"{sname[:12]:12s} {tname:38s} "
                  f"raw={row['DriftGuard (no sanitiser)']['attack_tpr']:.2f} "
                  f"+san={row['DriftGuard (+sanitiser)']['attack_tpr']:.2f} "
                  f"rule={row['Rule-based']['attack_tpr']:.2f}", flush=True)

    (out / "adversarial.json").write_text(json.dumps(rows, indent=2))
    lines = ["| Attack set | Evasion technique | n | DriftGuard | DriftGuard + sanitiser | Rule-based |",
             "|---|---|---|---|---|---|"]
    for r in rows:
        def f(k):
            return f"{r[k]['attack_tpr']:.2f} [{r[k]['ci95'][0]:.2f}, {r[k]['ci95'][1]:.2f}]"
        lines.append(f"| {r['set']} | {r['transform']} | {r['n']} | {f('DriftGuard (no sanitiser)')} | "
                     f"{f('DriftGuard (+sanitiser)')} | {f('Rule-based')} |")
    (out / "adversarial.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    app()
