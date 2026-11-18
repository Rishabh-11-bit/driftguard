"""Download more independent public text sources for the multi-source validation (scripts.run_multisource).

Only JSON rows are fetched (Hugging Face datasets-server API; raw JSON files from GitHub). Nothing is executed
and no archives are unpacked. Output goes to data/external2/<source>.jsonl (git-ignored) as
{"text": str, "label": 1 = attack / 0 = benign, "source": str}. Licences differ per dataset: they are used here
only for local evaluation; check each dataset card before redistributing anything.

Attack-style sources (label 1 and/or 0 as published):
  jackhhao/jailbreak-classification      jailbreak prompts vs benign prompts (published labels)
  reshabhs/SPML_Chatbot_Prompt_Injection  chatbot user prompts, published injection flag
  geekyrakshit/prompt-injection-dataset   mixed injection / benign prompts, published label
  Lakera/gandalf_ignore_instructions      real 'ignore previous instructions' attempts (attack only)
  uiuc-kang-lab/InjecAgent (GitHub)       attacker instructions for tool-using agents (attack only)
Benign-only sources (label 0):
  databricks/databricks-dolly-15k         human-written instructions (benign instruction-like text)
  yahma/alpaca-cleaned                    instructions
  OpenAssistant/oasst1                    first user messages, English
  google/Synthetic-Persona-Chat           persona facts (memory-note-like)

Usage:  python -m scripts.download_external2
"""

import json
import random
import time
from pathlib import Path

import requests
import typer

API = "https://datasets-server.huggingface.co/rows"
INJECAGENT = "https://raw.githubusercontent.com/uiuc-kang-lab/InjecAgent/main/data/attacker_cases_{}.jsonl"
app = typer.Typer()


def _get(params: dict) -> dict:
    """GET one page with a polite delay and exponential back-off on 429 / transient errors."""
    for attempt in range(8):
        time.sleep(0.7)
        r = requests.get(API, timeout=60, params=params)
        if r.status_code == 429 or r.status_code >= 500:
            time.sleep(min(60, 4 * 2 ** attempt))
            continue
        r.raise_for_status()
        return r.json()
    raise RuntimeError(f"giving up on {params.get('dataset')} offset {params.get('offset')}")


def rows(dataset: str, split: str, config: str, total: int, n: int, rng: random.Random) -> list[dict]:
    """n rows spread across the split (random 100-row windows), so a sorted dataset is not read from its head."""
    if total <= n:
        starts = list(range(0, total, 100))
    else:
        starts = sorted(rng.sample(range(0, max(total - 100, 1), 100), min(max(n // 100, 1), max(total // 100, 1))))
    out = []
    for s in starts:
        j = _get(dict(dataset=dataset, config=config, split=split, offset=s, length=100))
        out += [x["row"] for x in j["rows"]]
    return out[:n] if total > n else out


def total_rows(dataset: str, split: str, config: str) -> int:
    return int(_get(dict(dataset=dataset, config=config, split=split, offset=0, length=1))["num_rows_total"])


def one(s) -> str:
    return " ".join(str(s).split())


@app.command()
def main(out: Path = Path("data/external2"), n: int = 1000, seed: int = 7) -> None:
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    sets: dict[str, list[dict]] = {}

    def hf(name, dataset, split, config="default", total=None):
        t = total or total_rows(dataset, split, config)
        return rows(dataset, split, config, t, n, rng)

    r = hf("jailbreak", "jackhhao/jailbreak-classification", "train")
    sets["jackhhao_jailbreak"] = [dict(text=one(x["prompt"]), label=int(x["type"] == "jailbreak")) for x in r]
    r = hf("spml", "reshabhs/SPML_Chatbot_Prompt_Injection", "train")
    sets["spml_chatbot"] = [dict(text=one(x["User Prompt"]), label=int(str(x["Prompt injection"]) == "1")) for x in r]
    r = hf("geeky", "geekyrakshit/prompt-injection-dataset", "train")
    sets["geekyrakshit_injection"] = [dict(text=one(x["prompt"]), label=int(str(x["label"]) == "1")) for x in r]
    r = hf("gandalf", "Lakera/gandalf_ignore_instructions", "train")
    sets["gandalf_attack"] = [dict(text=one(x["text"]), label=1) for x in r]
    r = hf("dolly", "databricks/databricks-dolly-15k", "train")
    sets["dolly_benign"] = [dict(text=one(x["instruction"]), label=0) for x in r]
    r = hf("alpaca", "yahma/alpaca-cleaned", "train")
    sets["alpaca_benign"] = [dict(text=one(x["instruction"]), label=0) for x in r]
    r = hf("oasst", "OpenAssistant/oasst1", "train")
    sets["oasst_benign"] = [dict(text=one(x["text"]), label=0) for x in r
                            if x["role"] == "prompter" and x["lang"] == "en" and not x["parent_id"]]
    r = hf("persona", "google/Synthetic-Persona-Chat", "train")
    facts = [one(f) for x in r for f in str(x["user 1 personas"]).split("\n") if f.strip()]
    sets["synthetic_persona_benign"] = [dict(text=f, label=0) for f in facts]

    for kind in ("dh", "ds"):   # direct-harm and data-stealing attacker instructions for agents
        try:
            resp = requests.get(INJECAGENT.format(kind), timeout=60)
            resp.raise_for_status()
            cases = [json.loads(line) for line in resp.text.splitlines() if line.strip()]
            sets[f"injecagent_{kind}"] = [dict(text=one(c["Attacker Instruction"]), label=1) for c in cases]
        except Exception as e:   # network or schema drift: report and continue
            print(f"injecagent_{kind}: skipped ({type(e).__name__}: {str(e)[:80]})")

    for name, recs in sets.items():
        recs = [dict(r, source=name) for r in recs if 8 <= len(r["text"]) <= 1500]
        (out / f"{name}.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in recs), encoding="utf-8")
        print(f"{name}: {len(recs)} rows, attack={sum(r['label'] for r in recs)}")


if __name__ == "__main__":
    app()
