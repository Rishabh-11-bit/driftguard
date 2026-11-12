"""Write results_research/manifest.json: what exactly produced the reported numbers.

Records the git commit (and whether the tree was dirty), Python and key library versions, SHA-256 of every
data file and model under data/ and models/, the row counts of the JSONL datasets, and the digests of the
local Ollama models used as generators / agents / judges.

Usage:  python -m scripts.make_manifest
"""

import hashlib
import importlib.metadata as md
import json
import platform
import subprocess
from pathlib import Path

import requests
import typer

app = typer.Typer()
LIBS = ["numpy", "scikit-learn", "scipy", "sentence-transformers", "torch", "transformers", "fastapi",
        "starlette", "pydantic", "langgraph", "langchain-core"]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git(*args: str) -> str:
    try:
        return subprocess.run(["git", *args], capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


@app.command()
def main(out: Path = Path("results_research/manifest.json"), ollama: str = "http://localhost:11434") -> None:
    files = {}
    for root in ("data", "models", "results_research"):
        for p in sorted(Path(root).rglob("*")):
            if p.is_file() and p.suffix in {".jsonl", ".json", ".pkl", ".sig", ".md"} and p != out:
                entry = {"sha256": sha256(p), "bytes": p.stat().st_size}
                if p.suffix == ".jsonl":
                    entry["rows"] = sum(1 for line in p.open(encoding="utf-8") if line.strip())
                files[p.as_posix()] = entry
    versions = {}
    for lib in LIBS:
        try:
            versions[lib] = md.version(lib)
        except md.PackageNotFoundError:
            versions[lib] = None
    try:
        tags = requests.get(f"{ollama}/api/tags", timeout=5).json().get("models", [])
        models = {m["name"]: m["digest"] for m in tags}
    except (requests.RequestException, ValueError):
        models = "ollama not reachable"
    manifest = {"git_sha": git("rev-parse", "HEAD"), "git_dirty": bool(git("status", "--porcelain")),
                "python": platform.python_version(), "platform": platform.platform(),
                "libraries": versions, "ollama_models": models, "files": files}
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"wrote {out}: {len(files)} files, git {manifest['git_sha'][:8]}, dirty={manifest['git_dirty']}")


if __name__ == "__main__":
    app()
