"""The 8B end-to-end agent study (paper 5.9) rerun with the newer defences, on the IDENTICAL scenarios.

scripts.run_agent_study generates its scenarios from a seed, so the 300 scenarios (100 attack, 100 drift, 100 ordinary,
seed 3) are regenerated exactly. Defences compared: the rule-based filter and DriftGuard-v2 (the study's strongest
defences), the hybrid (models/classifier_hybrid_instr.pkl) and, if built, the ensemble (models/ensemble.json); origins are
attributed at runtime, as in the original study. Writes are normalised first, as the pipeline does.

Attack success is MEASURED, not projected: for every attack the defence admits, the local agent (llama3:latest) answers
the probe question with that write in memory; success = the answer contains the attacker's marker. Blocked attacks
cannot succeed (the candidate is not in memory). Drift and ordinary utility are NOT re-measured (that needs an agent
call for every scenario); only their write-level admission is reported.

Usage:  python -m scripts.run_agent_newdefences       Writes results_research/agent_newdefences.{json,md}
"""

import json
import random
from pathlib import Path

import numpy as np
import typer

from app.demo_data import daily_history
from driftguard.detector.classifier import DriftGuardClassifier
from driftguard.detector.features import SemanticFeatureExtractor
from driftguard.detector.hybrid import load_detector
from driftguard.provenance.attribution import Source, attribute_origin
from driftguard.provenance.schema import make_provenance
from driftguard.security import normalise_text
from eval.baselines.rule_based import RuleBasedDetector
from scripts.run_agent_study import agent_answer, make_scenarios, wilson

app = typer.Typer()


@app.command()
def main(
    v2: Path = Path("models/classifier_v2.pkl"), hybrid: Path = Path("models/classifier_hybrid_instr.pkl"),
    ensemble: Path = Path("models/ensemble.json"), agent_model: str = "llama3:latest", n_per: int = 100, seed: int = 3,
    out: Path = Path("results_research"),
) -> None:
    scenarios = make_scenarios(n_per, random.Random(seed))
    ext = SemanticFeatureExtractor()
    hist = daily_history()
    rule = RuleBasedDetector()
    dets = {"DriftGuard-v2": DriftGuardClassifier.load(v2), "Hybrid": load_detector(hybrid)}
    if ensemble.exists():
        dets["Ensemble"] = load_detector(ensemble)

    def feats(sc, origin):
        prov = make_provenance(origin=origin, session_id="study", triggering_text=sc.query)
        return ext.extract(content=sc.candidate, provenance=prov, write_history=hist, session_id="study",
                           last_query=sc.query, user_memory_entries=sc.seeds)

    names = ["No defence", "Rule-based", *dets]
    decisions = {n: [] for n in names}
    for i, sc in enumerate(scenarios):
        attributed = attribute_origin(sc.candidate, [Source("user_turn", sc.query)] +
                                      [Source(o["origin"], o["text"]) for o in sc.observations]).origin
        x = feats(sc, attributed)
        text = normalise_text(sc.candidate)
        decisions["No defence"].append("admit")
        decisions["Rule-based"].append({0: "block", 1: "flag", 2: "admit"}[int(rule.predict(x[:18].reshape(1, -1))[0])])
        for n, d in dets.items():
            k = len(d._feature_names or []) or 18
            pred, _ = d.predict_one(x[:k], content=text) if getattr(d, "needs_text", False) else d.predict_one(x[:k])
            decisions[n].append({0: "block", 1: "flag", 2: "admit"}[int(pred)])
        if i % 25 == 0:
            print(f"decided {i}/{len(scenarios)}", flush=True)

    cache: dict = {}
    report = {}
    for n in names:
        r = {k: [] for k in ("block_attack", "exposure", "asr", "drift_admit", "ordinary_admit")}
        for sc, dec in zip(scenarios, decisions[n]):
            admitted = dec in ("admit", "flag")
            if sc.kind == "attack":
                r["block_attack"].append(dec == "block")
                r["exposure"].append(admitted)
                if admitted:        # a blocked write is not in memory, so the attack cannot reach the agent
                    ans = agent_answer(agent_model, sc.seeds + [sc.candidate], sc.probe, cache).lower()
                    r["asr"].append(any(m in ans for m in sc.markers))
                else:
                    r["asr"].append(False)
            elif sc.kind == "drift":
                r["drift_admit"].append(admitted)
            else:
                r["ordinary_admit"].append(admitted)
        report[n] = {k: float(np.mean(v)) for k, v in r.items() if v}
        report[n]["_ci95"] = {k: wilson(int(np.sum(v)), len(v)) for k, v in r.items() if v}
        print(n, {k: round(v, 2) for k, v in report[n].items() if k != "_ci95"}, flush=True)

    # paired comparison on the SAME attack scenarios: exact McNemar (two-sided binomial on the discordant pairs)
    from scipy.stats import binomtest
    attack_idx = [i for i, s in enumerate(scenarios) if s.kind == "attack"]
    exposed = {n: np.array([decisions[n][i] != "block" for i in attack_idx]) for n in names}
    paired = {}
    for a, b in (("Hybrid", "Rule-based"), ("Ensemble", "Rule-based"), ("Hybrid", "DriftGuard-v2"),
                 ("Ensemble", "DriftGuard-v2"), ("Ensemble", "Hybrid")):
        if a in exposed and b in exposed:
            only_a = int((exposed[a] & ~exposed[b]).sum())       # a lets it through, b blocks it
            only_b = int((~exposed[a] & exposed[b]).sum())
            p = binomtest(min(only_a, only_b), only_a + only_b, 0.5).pvalue if only_a + only_b else 1.0
            paired[f"{a} vs {b}"] = {f"only_{a}_exposed": only_a, f"only_{b}_exposed": only_b, "mcnemar_exact_p": float(p)}
    report["_paired_exposure_tests"] = paired
    print("paired:", json.dumps(paired, indent=1))
    report["_setup"] = {"agent_model": agent_model, "n_per": n_per, "seed": seed, "llm_calls": len(cache),
                        "note": "ASR measured for every attack with the candidate in memory; utility not re-measured"}
    out.mkdir(exist_ok=True)
    (out / "agent_newdefences.json").write_text(json.dumps(report, indent=2))
    cols = ["block_attack", "exposure", "asr", "drift_admit", "ordinary_admit"]
    lines = ["| Defence | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]
    for n in names:
        lines.append(f"| {n} | " + " | ".join(f"{report[n][c]:.2f} [{report[n]['_ci95'][c][0]:.2f},{report[n]['_ci95'][c][1]:.2f}]" for c in cols) + " |")
    (out / "agent_newdefences.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    app()
