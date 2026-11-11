"""Download small slices of public datasets used for real-text evaluation.

Fetches through the Hugging Face datasets-server JSON API (no archives, no code execution)
into data/external/ (git-ignored). Licences differ per dataset - check each dataset card
before redistributing; we only use them for local evaluation.

  deepset/prompt-injections             real prompt-injection strings + benign prompts (all rows)
  xTRam1/safe-guard-prompt-injection    injection / benign prompts (first N test rows)
  Cynaptics/persona-chat                crowd-written persona facts (first N rows)

Usage:  python -m scripts.download_external
"""

import json
from pathlib import Path

import requests
import typer

API = "https://datasets-server.huggingface.co/rows"
app = typer.Typer()


def fetch(dataset: str, split: str, total: int, config: str = "default") -> list[dict]:
    rows, offset = [], 0
    while offset < total:
        n = min(100, total - offset)
        r = requests.get(API, timeout=60, params=dict(dataset=dataset, config=config, split=split,
                                                       offset=offset, length=n))
        r.raise_for_status()
        batch = [x["row"] for x in r.json()["rows"]]
        if not batch:
            break
        rows += batch
        offset += len(batch)
    return rows


@app.command()
def main(out: Path = Path("data/external"), safeguard_n: int = 1000, persona_n: int = 1000) -> None:
    out.mkdir(parents=True, exist_ok=True)
    jobs = [
        ("deepset_prompt_injections_train", "deepset/prompt-injections", "train", 600),
        ("deepset_prompt_injections_test", "deepset/prompt-injections", "test", 200),
        ("safeguard_prompt_injection_test", "xTRam1/safe-guard-prompt-injection", "test", safeguard_n),
        ("persona_chat", "Cynaptics/persona-chat", "train", persona_n),
    ]
    for name, ds, split, n in jobs:
        rows = fetch(ds, split, n)
        path = out / f"{name}.jsonl"
        path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")
        print(f"{name}: {len(rows)} rows, {path.stat().st_size / 1024:.0f} KB, columns={list(rows[0])}")


if __name__ == "__main__":
    app()
