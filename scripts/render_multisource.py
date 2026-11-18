"""Render results_research/multisource.md from results_research/multisource.json (so the tables always match the data,
including after a partial rerun with `run_multisource --only ...`).

Usage:  python -m scripts.render_multisource
"""

import json
from pathlib import Path

import typer

app = typer.Typer()
ORDER = ("hybrid_shipped", "hybrid_instr", "finetuned_instr", "protectai_deberta", "tfidf_loso", "embed_loso",
         "fuse_hybrid+deb", "fuse_ft+deb", "fuse_all3")


@app.command()
def main(src: Path = Path("results_research/multisource.json"), out: Path = Path("results_research/multisource.md")) -> None:
    result = json.loads(src.read_text())
    names = [n for n in ORDER if any(n in v for v in result.values())]
    head = lambda first: [f"| {first} | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
    lines = head("Test group (n, attacks) / note")
    for g, v in result.items():
        cells = [f"{v[n]['auroc']:.3f} [{v[n]['auroc_ci95'][0]:.2f},{v[n]['auroc_ci95'][1]:.2f}]" if n in v else "n/a" for n in names]
        lines.append(f"| {g} ({v['n']}, {v['n_attack']}); {v['note']} | " + " | ".join(cells) + " |")
    for title, key in (("False-alarm rate at threshold 0.5 (benign flagged):", "false_alarm"),
                       ("Recall at threshold 0.5:", "recall")):
        lines += ["", title, ""] + head("Test group")
        for g, v in result.items():
            lines.append(f"| {g} | " + " | ".join(f"{v[n][key]:.2f}" if n in v else "n/a" for n in names) + " |")
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {out} ({len(result)} groups)")


if __name__ == "__main__":
    app()
