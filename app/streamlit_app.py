"""
DriftGuard demo UI.

Run:  streamlit run app/streamlit_app.py

Tabs
  Live Guard  - type a memory write, watch it get classified and enforced
  Evaluation  - metrics, confusion matrices, ROC, generalisation results
  Benchmark   - explore the labelled dataset the detector is trained on
  How it works - architecture
"""

from __future__ import annotations

import json
import logging
import sys
import warnings
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")
logging.disable(logging.WARNING)

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from driftguard.detector.classifier import DriftGuardClassifier
from driftguard.provenance.enforcement import EnforcementPolicy, InMemoryReviewQueue
from driftguard.provenance.write_pipeline import CandidateWrite, WritePipeline
from scripts.train_detector import FEATURE_NAMES
from app.demo_data import PRESETS, SEED_MEMORIES, daily_history
from app import local_llm

st.set_page_config(page_title="DriftGuard", page_icon="🛡️", layout="wide")

CLASS_NAMES = ["Malicious", "Benign drift", "Ordinary learning"]
CLASS_COLORS = ["#ef4444", "#f59e0b", "#22c55e"]
MODELS = {
    "Hardened model (recommended)": ROOT / "models" / "classifier_hard.pkl",
    "Raw-benchmark model": ROOT / "models" / "classifier.pkl",
}
st.markdown(
    """
    <style>
    .block-container {padding-top: 2.2rem; max-width: 1280px;}
    .hero h1 {margin: 0; font-size: 2.3rem; letter-spacing: -0.5px;}
    .hero p {color: #94a3b8; margin: .2rem 0 1rem 0;}
    .verdict {border-radius: 14px; padding: 18px 22px; margin: 6px 0 14px 0; border: 1px solid;}
    .verdict h2 {margin: 0; font-size: 1.5rem;}
    .verdict p {margin: .3rem 0 0 0; color: #cbd5e1;}
    .v0 {background: #3b1219; border-color: #ef4444;}
    .v1 {background: #3a2a0c; border-color: #f59e0b;}
    .v2 {background: #0f2f1c; border-color: #22c55e;}
    .chip {display:inline-block; padding: 2px 10px; border-radius: 999px; font-size: .78rem;
           margin-right: 6px; background:#1e293b; color:#cbd5e1; border:1px solid #334155;}
    div[data-testid="stMetricValue"] {font-size: 1.6rem;}
    div[data-testid="stMetric"] {background:#131c2e; border:1px solid #1f2b44; border-radius:12px;
           padding: 10px 14px;}
    </style>
    """,
    unsafe_allow_html=True,
)


# ─────────────────────────────────────────────────────────────
# Session state
# ─────────────────────────────────────────────────────────────

class ListMemory:
    """Minimal in-memory store with the add/list_all interface WritePipeline needs."""

    def __init__(self) -> None:
        self.items: list[dict] = []

    def add(self, text: str, metadata: dict | None = None) -> str:
        self.items.append({"text": text, "metadata": metadata or {}})
        return str(len(self.items))

    def list_all(self) -> list[dict]:
        return self.items

    def __len__(self) -> int:
        return len(self.items)


@st.cache_resource(show_spinner="Loading detector…")
def load_model(path: str) -> DriftGuardClassifier:
    return DriftGuardClassifier.load(path)


@st.cache_resource(show_spinner="Warming up the embedding model (first run only)…")
def load_extractor():
    """One shared FeatureExtractor with the encoder already loaded."""
    from driftguard.detector.features import FeatureExtractor

    ex = FeatureExtractor()
    ex.encoder.encode(["warm-up"])
    return ex


def new_session(model_path: Path) -> None:
    mem = ListMemory()
    for t in SEED_MEMORIES:
        mem.add(t, {"seed": True})
    queue = InMemoryReviewQueue()
    st.session_state.mem = mem
    st.session_state.queue = queue
    st.session_state.pipeline = WritePipeline(
        mem, load_model(str(model_path)), EnforcementPolicy(quarantine_queue=queue)
    )
    st.session_state.pipeline._extractor = load_extractor()
    st.session_state.pipeline.inject_history(daily_history())
    st.session_state.naive = list(SEED_MEMORIES)  # unprotected agent: stores every write
    st.session_state.llm_out = None
    st.session_state.log = []
    st.session_state.last = None
    st.session_state.model_path = str(model_path)


def read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────

available = {k: v for k, v in MODELS.items() if v.exists()}

with st.sidebar:
    st.markdown("### 🛡️ DriftGuard")
    st.caption("Provenance-aware defence against memory poisoning in LLM agents")
    if not available:
        st.error("No trained model found in `models/`.")
        st.code(
            "python scripts/generate_benchmark.py --hard --out data/benchmark_hard.jsonl\n"
            "python scripts/train_detector.py --benchmark data/benchmark_hard.jsonl "
            "--model-out models/classifier_hard.pkl",
            language="bash",
        )
        st.stop()
    model_label = st.selectbox("Detector model", list(available))
    model_path = available[model_label]
    if st.button("Reset session", use_container_width=True):
        new_session(model_path)
    st.divider()
    st.caption("**Team** · Tanisha Talathi · Aryaan Sethi · Rishabh Gaikwad")
    st.caption("DCET, MIT-WPU · Capstone 2026")

if st.session_state.get("model_path") != str(model_path):
    new_session(model_path)

pipeline: WritePipeline = st.session_state.pipeline
queue: InMemoryReviewQueue = st.session_state.queue
mem: ListMemory = st.session_state.mem

st.markdown(
    '<div class="hero"><h1>🛡️ DriftGuard</h1>'
    "<p>Tells a genuine change in what the user wants apart from an attacker rewriting the "
    "agent's memory.</p></div>",
    unsafe_allow_html=True,
)

tab_live, tab_eval, tab_bench, tab_how = st.tabs(
    ["🔬 Live Guard", "📊 Evaluation", "🧪 Benchmark", "🧭 How it works"]
)


# ─────────────────────────────────────────────────────────────
# Live Guard
# ─────────────────────────────────────────────────────────────

VERDICTS = {
    0: ("v0", "🚫 Quarantined — likely memory poisoning",
        "Blocked before reaching memory. Queued for human review."),
    1: ("v1", "🟡 Admitted & flagged — benign preference drift",
        "A real change in the user's preferences. Stored, with a flag so it can be audited."),
    2: ("v2", "✅ Admitted — ordinary learning",
        "Routine fact. Stored silently with no friction."),
}


def run_write(text: str, origin: str, query: str, burst: int) -> None:
    if burst:
        now = datetime.now(timezone.utc)
        pipeline.inject_history([
            {
                "timestamp": (now - timedelta(seconds=20 * (burst - i))).isoformat(),
                "origin": origin, "trust_level": 0.5, "session_id": "demo-session",
                "interaction_hash": f"burst-{i}", "predicted_class": 2,
            }
            for i in range(burst)
        ])
    res = pipeline.process(CandidateWrite(
        content=text, origin=origin, session_id="demo-session", triggering_text=query,
    ))
    st.session_state.setdefault("naive", list(SEED_MEMORIES)).append(text)
    st.session_state.last = {
        "text": text, "origin": origin, "query": query, "res": res,
        "features": None if pipeline.last_features is None else list(map(float, pipeline.last_features)),
    }
    st.session_state.log.append({
        "time": datetime.now().strftime("%H:%M:%S"), "origin": origin,
        "verdict": CLASS_NAMES[res.predicted_class], "action": res.action,
        "write": text[:90],
    })


with tab_live:
    left, right = st.columns([5, 6], gap="large")

    with left:
        st.subheader("Submit a memory write")
        choice = st.selectbox("Example scenarios", list(PRESETS))
        preset = PRESETS[choice]
        if preset and st.session_state.get("_preset") != choice:
            st.session_state._preset = choice
            st.session_state.in_origin, st.session_state.in_query, st.session_state.in_text = preset
        origins = ["user_turn", "tool_output", "retrieved_doc", "agent_reflection", "unknown"]
        st.session_state.setdefault("in_origin", "user_turn")
        origin = st.selectbox("Origin of the write (provenance)", origins, key="in_origin")
        query = st.text_input("What the user just asked", key="in_query")
        text = st.text_area("Candidate memory write", key="in_text", height=120,
                            placeholder="e.g. User now prefers vegetarian food.")
        burst = st.slider("Simulate a burst: prior writes in the last few minutes", 0, 12, 0)
        go_btn = st.button("Run through DriftGuard", type="primary", use_container_width=True,
                           disabled=not text.strip())
        if go_btn:
            with st.spinner("Classifying…"):
                run_write(text.strip(), origin, query.strip(), burst)

        auto_btn = st.button("▶ Auto-run attack demo", use_container_width=True,
                             help="Feeds every attack scenario through DriftGuard automatically, "
                                  "then asks the local Llama agent with and without protection.")
        if auto_btn:
            rows = []
            with st.spinner("Running attack scenarios…"):
                for name, preset_vals in PRESETS.items():
                    if not preset_vals or "Attack" not in name:
                        continue
                    o, q, t = preset_vals
                    run_write(t, o, q, 0)
                    r = st.session_state.last["res"]
                    rows.append({
                        "attack": name.replace("🔴 Attack: ", ""),
                        "verdict": CLASS_NAMES[r.predicted_class],
                        "action": r.action,
                        "P(malicious)": f"{r.probabilities[0]:.0%}",
                    })
            st.session_state.auto_rows = rows
            if local_llm.list_models():
                m = ("llama3.2:1b" if "llama3.2:1b" in local_llm.list_models()
                     else local_llm.list_models()[0])
                qn = "Can you recommend a laptop for my studies?"
                with st.spinner(f"Asking {m} locally…"):
                    try:
                        st.session_state.llm_out = {
                            "q": qn, "model": m,
                            "naive": local_llm.answer(m, st.session_state.naive, qn),
                            "guard": local_llm.answer(m, [e["text"] for e in mem.items], qn),
                        }
                    except Exception as exc:
                        st.session_state.llm_out = {"error": str(exc)}
        if st.session_state.get("auto_rows"):
            rws = st.session_state.auto_rows
            blocked = sum(r["action"] == "quarantined" for r in rws)
            st.success(f"Blocked {blocked} of {len(rws)} poisoning attempts.")
            st.dataframe(pd.DataFrame(rws), hide_index=True, use_container_width=True)

    with right:
        st.subheader("Verdict")
        last = st.session_state.last
        if not last:
            st.info("Pick an example scenario or type your own write, then run it.")
        else:
            res = last["res"]
            cls = res.predicted_class
            css, title, blurb = VERDICTS[cls]
            st.markdown(
                f'<div class="verdict {css}"><h2>{title}</h2><p>{blurb}</p></div>',
                unsafe_allow_html=True,
            )
            c1, c2, c3 = st.columns(3)
            c1.metric("Action", {"quarantined": "Blocked", "admitted_flagged": "Flagged",
                                 "admitted": "Admitted"}.get(res.action, res.action))
            c2.metric("Confidence", f"{max(res.probabilities):.0%}")
            c3.metric("Latency", f"{res.latency_ms:.0f} ms")

            fig = go.Figure(go.Bar(
                x=list(res.probabilities), y=CLASS_NAMES, orientation="h",
                marker_color=CLASS_COLORS, text=[f"{p:.1%}" for p in res.probabilities],
                textposition="outside",
            ))
            fig.update_layout(
                height=190, margin=dict(l=0, r=30, t=10, b=0), xaxis=dict(range=[0, 1.1]),
                yaxis=dict(autorange="reversed"), paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#e6edf7"),
            )
            st.plotly_chart(fig, use_container_width=True)

            if last["features"]:
                with st.expander("Signals the detector saw (18 features)"):
                    fdf = pd.DataFrame({"feature": FEATURE_NAMES, "value": last["features"]})
                    st.dataframe(fdf, hide_index=True, use_container_width=True, height=300)

    st.divider()
    m1, m2 = st.columns(2, gap="large")

    with m1:
        st.subheader(f"🧠 Agent memory ({len(mem)})")
        rows = []
        for e in mem.items:
            meta = e["metadata"]
            rows.append({
                "memory": e["text"],
                "status": "seed" if meta.get("seed") else (
                    "🟡 flagged drift" if meta.get("flagged") else "admitted"),
            })
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

    with m2:
        pending = queue.list_pending()
        st.subheader(f"🚨 Quarantine inbox ({len(pending)})")
        if not pending:
            st.caption("Nothing waiting for review.")
        for q in pending:
            with st.container(border=True):
                st.write(q.candidate_text)
                st.caption(f"P(malicious) = {q.probabilities[0]:.0%}")
                a, r = st.columns(2)
                if a.button("Approve (false alarm)", key=f"ap{q.quarantine_id}"):
                    queue.mark_reviewed(q.quarantine_id, "approve")
                    mem.add(q.candidate_text, {"reviewed": True})
                    st.rerun()
                if r.button("Reject", key=f"rj{q.quarantine_id}"):
                    queue.mark_reviewed(q.quarantine_id, "reject")
                    st.rerun()

    st.divider()
    st.subheader("🦙 Ask the local Llama agent")
    models = local_llm.list_models()
    if not models:
        st.info("Ollama is not running. Start it (`ollama serve`) and pull a model, "
                "e.g. `ollama pull llama3.2:1b`, then reload.")
    else:
        st.caption("Same local model, same question — one agent remembers every write "
                   "(unprotected), the other only what DriftGuard admitted.")
        cm, cq = st.columns([2, 5])
        llm_model = cm.selectbox("Local model", models,
                                 index=models.index("llama3.2:1b") if "llama3.2:1b" in models else 0)
        question = cq.text_input("Question for the agent",
                                 value="Can you recommend a laptop for my studies?")
        if st.button("Ask both agents", use_container_width=True):
            protected = [e["text"] for e in mem.items]
            with st.spinner(f"Running {llm_model} locally…"):
                try:
                    st.session_state.llm_out = {
                        "q": question, "model": llm_model,
                        "naive": local_llm.answer(llm_model, st.session_state.setdefault("naive", list(SEED_MEMORIES)), question),
                        "guard": local_llm.answer(llm_model, protected, question),
                    }
                except Exception as exc:
                    st.session_state.llm_out = {"error": str(exc)}
        out = st.session_state.get("llm_out")
        if out and "error" in out:
            st.error(f"Ollama call failed: {out['error']}")
        elif out:
            a, b = st.columns(2, gap="large")
            with a:
                with st.container(border=True):
                    st.markdown("**⚠️ Unprotected agent**")
                    st.caption(f"memory: {len(st.session_state.get('naive', SEED_MEMORIES))} entries (all writes)")
                    st.write(out["naive"])
            with b:
                with st.container(border=True):
                    st.markdown("**🛡️ DriftGuard-protected agent**")
                    st.caption(f"memory: {len(mem)} entries (quarantined writes excluded)")
                    st.write(out["guard"])

    if st.session_state.log:
        st.subheader("Session log")
        st.dataframe(pd.DataFrame(st.session_state.log[::-1]), hide_index=True,
                     use_container_width=True)


# ─────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────

def pct(x):
    return "n/a" if x is None else f"{x:.1%}"


def render_eval(folder: Path, title: str, blurb: str) -> None:
    dg = read_json(folder / "DriftGuard_metrics.json")
    if not dg:
        st.warning(f"No results in `{folder.name}/`. Run `scripts/run_eval.py`.")
        return
    ho = read_json(folder / "DriftGuard_held-out_metrics.json") or {}
    rb = read_json(folder / "Rule-Based_metrics.json") or {}
    lofo = read_json(folder / "lofo.json") or {}
    ext = dg.get("extended") or {}

    st.markdown(f"#### {title}")
    st.caption(blurb)
    c = st.columns(6)
    c[0].metric("Accuracy", f"{dg['accuracy']:.1%}")
    c[1].metric("Macro F1", f"{dg['f1_macro']:.3f}")
    c[2].metric("False alarms", pct(dg["FAR"]), help="Benign drift wrongly quarantined")
    c[3].metric("Missed attacks", pct(dg["MAR"]), help="Attacks that got into memory")
    c[4].metric("MCC", f"{ext.get('mcc', 0):.3f}")
    c[5].metric("Unseen attacks caught",
                pct((ho.get("extended") or {}).get("attack_tpr")),
                help="Attack families never seen in training")

    t1, t2, t3 = st.tabs(["Comparison", "Extended metrics", "Plots"])
    with t1:
        rows = [{"Detector": "DriftGuard", "Accuracy": dg["accuracy"], "F1": dg["f1_macro"],
                 "FAR": dg["FAR"], "MAR": dg["MAR"]}]
        if rb:
            rows.append({"Detector": "Rule-based baseline", "Accuracy": rb["accuracy"],
                         "F1": rb["f1_macro"], "FAR": rb["FAR"], "MAR": rb["MAR"]})
        if ho:
            rows.append({"Detector": "DriftGuard on unseen attack families",
                         "Accuracy": ho["accuracy"], "F1": None,
                         "FAR": None, "MAR": ho["MAR"]})
        st.dataframe(pd.DataFrame(rows).round(4), hide_index=True, use_container_width=True)
        if lofo:
            st.markdown("**Leave-one-attack-family-out** (train without the family, then test on it)")
            lrows = [{"family": k, "n": v["n"], "detection": v["detection_rate"]}
                     for k, v in lofo.items() if not k.startswith("_")]
            st.dataframe(pd.DataFrame(lrows).round(3), hide_index=True, use_container_width=True)
    with t2:
        if not ext:
            st.info("Re-run `scripts/run_eval.py` to generate extended metrics.")
        else:
            keys = [
                ("Balanced accuracy", "balanced_accuracy"), ("Cohen's kappa", "cohen_kappa"),
                ("Attack recall", "attack_tpr"), ("Attack precision", "attack_precision"),
                ("Attack F2", "attack_f2"), ("Drift preservation rate", "drift_preservation_rate"),
                ("Drift recognition rate", "drift_recognition_rate"),
                ("Legit-write quarantine rate", "legit_quarantine_rate"),
                ("Malicious PR-AUC", "malicious_pr_auc"), ("Brier score ↓", "brier_score"),
                ("Calibration error (ECE) ↓", "ece"), ("Cost reduction vs admit-all", "cost_reduction"),
                ("Latency ms/write", "latency_ms_per_write"),
            ]
            st.dataframe(
                pd.DataFrame([{"Metric": n, "Value": ext.get(k)} for n, k in keys if k in ext]).round(4),
                hide_index=True, use_container_width=True)
            ci = ext.get("confidence_intervals", {})
            if ci:
                st.caption("95% bootstrap CI: " + " · ".join(
                    f"{k} [{lo:.3f}, {hi:.3f}]" for k, (lo, hi) in ci.items()))
    with t3:
        imgs = [("Confusion matrix", "driftguard_confusion_matrix.png"),
                ("ROC curves", "driftguard_roc_curves.png"),
                ("Detection by attack family", "driftguard_family_breakdown.png"),
                ("Held-out attacks", "held_out_confusion_matrix.png")]
        cols = st.columns(2)
        for i, (cap, fn) in enumerate(imgs):
            if (folder / fn).exists():
                cols[i % 2].image(str(folder / fn), caption=cap, use_container_width=True)


with tab_eval:
    which = st.radio("Benchmark", ["Hardened (realistic)", "Raw (generator artefacts present)"],
                     horizontal=True)
    if which.startswith("Hardened"):
        render_eval(ROOT / "results_hard", "Hardened benchmark",
                    "Timing, trust, length and metadata fingerprints removed, so the detector "
                    "has to use content and provenance.")
    else:
        render_eval(ROOT / "results", "Raw benchmark",
                    "Near-perfect scores here partly reflect generator artefacts, not "
                    "real separability. Compare with the hardened run.")


# ─────────────────────────────────────────────────────────────
# Benchmark explorer
# ─────────────────────────────────────────────────────────────

@st.cache_data
def load_bench(path: str) -> pd.DataFrame:
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line]
    return pd.DataFrame([{
        "label": r["label"], "family": r["attack_family"], "origin": r["provenance"]["origin"],
        "trust": round(r["provenance"]["trust_level"], 2), "content": r["content"],
        "query": r["last_query"],
    } for r in rows])


with tab_bench:
    files = {"Hardened training set": ROOT / "data" / "benchmark_hard.jsonl",
             "Raw training set": ROOT / "data" / "benchmark.jsonl",
             "Hardened held-out attacks": ROOT / "data" / "held_out_hard.jsonl",
             "Raw held-out attacks": ROOT / "data" / "held_out.jsonl"}
    files = {k: v for k, v in files.items() if v.exists()}
    if not files:
        st.warning("No benchmark files in `data/`. Run `scripts/generate_benchmark.py`.")
    else:
        name = st.selectbox("Dataset", list(files))
        df = load_bench(str(files[name]))
        a, b = st.columns([3, 4])
        counts = df.groupby(["label", "family"]).size().reset_index(name="entries")
        a.metric("Entries", len(df))
        a.dataframe(counts, hide_index=True, use_container_width=True)
        fig = go.Figure(go.Bar(
            x=counts["family"], y=counts["entries"],
            marker_color=[{"malicious": "#ef4444", "benign_drift": "#f59e0b"}.get(l, "#22c55e")
                          for l in counts["label"]]))
        fig.update_layout(height=260, margin=dict(l=0, r=0, t=10, b=0), paper_bgcolor="rgba(0,0,0,0)",
                          plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#e6edf7"))
        b.plotly_chart(fig, use_container_width=True)
        f = st.multiselect("Filter by class", sorted(df["label"].unique()),
                           default=sorted(df["label"].unique()))
        st.dataframe(df[df["label"].isin(f)].sample(frac=1, random_state=1).head(60),
                     hide_index=True, use_container_width=True, height=380)
        st.caption("All data is synthetic, generated from templates that mirror MINJA and "
                   "AgentPoison attack structure.")


# ─────────────────────────────────────────────────────────────
# How it works
# ─────────────────────────────────────────────────────────────

with tab_how:
    st.markdown(
        "Every candidate memory write is intercepted, tagged with **provenance** (origin, trust, "
        "time, interaction hash), turned into an **18-feature vector** in four families "
        "(provenance, temporal, profile-consistency, causal attribution), and classified into "
        "three classes by a calibrated random forest. The enforcement policy then quarantines "
        "malicious writes, admits benign drift with a flag, and admits ordinary learning silently."
    )
    st.image(str(ROOT / "app" / "assets" / "architecture.png"), use_container_width=True)
    st.image(str(ROOT / "app" / "assets" / "modules.png"), use_container_width=True)
    st.markdown(
        "**Why three classes?** Binary attack detectors quarantine every preference change and "
        "wreck personalisation. Separating *benign drift* from *malicious* is the point of the project."
    )
