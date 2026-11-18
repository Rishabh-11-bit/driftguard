"""Build the signed ensemble model: hybrid + fine-tuned MiniLM + ProtectAI DeBERTa, stacked.

Steps (all local, nothing is executed from downloaded files):
  1. fit the 3-input stacker on the saved per-sample scores of EVERY evaluation group (python -m scripts.run_multisource),
     each group weighted equally and classes balanced (the leave-one-group-out version of this fit is what
     scripts.run_stacking evaluates; the shipped stacker uses all groups);
  2. convert the fine-tuned weights (models/finetuned_minilm_instr.pt, torch state dict) to safetensors and sign them;
  3. pin the external detector to its current Hugging Face commit and write models/ensemble.json, then sign it.

Usage:  python -m scripts.run_multisource && python -m scripts.train_ensemble
Then:   DRIFTGUARD_MODEL_PATH=models/ensemble.json uvicorn --factory service.api:create_app
"""

import json
from pathlib import Path

import numpy as np
import torch
import typer
from sklearn.linear_model import LogisticRegression

from driftguard.security import sign_file

app = typer.Typer()
INPUTS = ["hybrid_instr", "finetuned_instr", "protectai_deberta"]
EXTERNAL = "protectai/deberta-v3-base-prompt-injection-v2"


def logit(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


@app.command()
def main(scores: Path = Path("results_research/multisource_scores.json"),
         weights: Path = Path("models/finetuned_minilm_instr.pt"), hybrid: str = "classifier_hybrid_instr.pkl",
         out_dir: Path = Path("models"), threshold: float = 0.5) -> None:
    from huggingface_hub import HfApi
    from safetensors.torch import save_file

    S = json.loads(scores.read_text())
    X, y, w = [], [], []
    for g, d in S.items():
        yg = np.array(d["y"])
        wg = np.where(yg == 1, 0.5 / max(yg.sum(), 1), 0.5 / max((1 - yg).sum(), 1))
        X.append(np.stack([logit(d[k]) for k in INPUTS], axis=1)), y.append(yg), w.append(wg)
    lr = LogisticRegression(C=1.0, max_iter=1000).fit(np.vstack(X), np.concatenate(y), sample_weight=np.concatenate(w))
    print("stacker weights", dict(zip(INPUTS, np.round(lr.coef_[0], 3))), "intercept", round(float(lr.intercept_[0]), 3))

    out_dir.mkdir(exist_ok=True)
    sd = {k: v.contiguous() for k, v in torch.load(weights, map_location="cpu").items()}
    st = out_dir / "finetuned_minilm_instr.safetensors"
    save_file(sd, str(st))
    sign_file(st)
    revision = HfApi().model_info(EXTERNAL).sha
    cfg = {"hybrid": hybrid, "finetuned": st.name, "external": {"name": EXTERNAL, "revision": revision},
           "coef": [round(float(c), 6) for c in lr.coef_[0]], "intercept": round(float(lr.intercept_[0]), 6),
           "threshold": threshold, "inputs": INPUTS, "fitted_on_groups": sorted(S)}
    path = out_dir / "ensemble.json"
    path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    sign_file(path)
    print(f"wrote {path} (+ .sig), {st} (+ .sig); external pinned at {revision}")


if __name__ == "__main__":
    app()
