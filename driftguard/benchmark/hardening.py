"""
Benchmark hardening — removes generator artefacts from labelled entries.

The raw generators leave class-specific fingerprints in fields that have
nothing to do with whether a write is an attack: write timing, source trust,
text length, and label-bearing metadata. A detector can reach ~100% by reading
those instead of the content/provenance semantics, and then fails on any
attack family whose generator leaves different fingerprints.

`harden_entries` breaks that shortcut with three label-independent
transformations, applied to every class with probability `camouflage`:

  1. Donor swap     — write_history and trust_level are replaced by those of a
                      random entry from any class, so timing and trust carry no
                      class information.
  2. Length padding — neutral filler clauses (drawn from one shared pool) are
                      appended, so text length overlaps across classes.
  3. Metadata scrub — generator metadata that names the class is removed.

Labels and the semantic content of each write are never changed.
"""

from __future__ import annotations

import copy
import random

from driftguard.benchmark.labelling import BenchmarkEntry

NEUTRAL_FILLER = [
    "Noted during our conversation today.",
    "This came up while we were chatting.",
    "Mentioned in passing earlier on.",
    "Please keep this in mind going forward.",
    "That is what was said just now.",
    "This was stated in the current session.",
    "Recorded as part of the ongoing discussion.",
    "Brought up again after a short break.",
]

_LABEL_BEARING_META = ("type", "attack_type", "stage", "source", "goal", "template_idx", "drift_magnitude")


def harden_entries(
    entries: list[BenchmarkEntry],
    camouflage: float = 0.7,
    seed: int = 42,
) -> list[BenchmarkEntry]:
    """Return hardened deep copies of `entries`; the inputs are left untouched."""
    rng = random.Random(seed)
    out = [copy.deepcopy(e) for e in entries]
    donors = [(e.write_history, e.provenance.get("trust_level")) for e in entries]

    for e in out:
        if rng.random() < camouflage:
            hist, trust = copy.deepcopy(rng.choice(donors))
            e.write_history = hist
            if trust is not None:
                e.provenance["trust_level"] = trust
        if rng.random() < camouflage:
            extra = " ".join(rng.choices(NEUTRAL_FILLER, k=rng.randint(1, 3)))
            e.content = f"{e.content} {extra}"
        e.provenance["metadata"] = {
            k: v for k, v in e.provenance.get("metadata", {}).items() if k not in _LABEL_BEARING_META
        }
        e.metadata = {k: v for k, v in e.metadata.items() if k not in _LABEL_BEARING_META}
    return out
