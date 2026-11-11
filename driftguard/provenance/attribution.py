"""Runtime source attribution (taint tracking) for memory writes.

Problem (SECURITY.md F-03): the reference agent used to label every write
`agent_reflection`, which carries the *highest* default trust (0.9). A write that was
really copied out of a poisoned web page therefore looked like the agent's own trusted
reasoning, and the detector's provenance features carried no information.

Fix: the runtime knows which content entered the context window and from where
(user message, tool result, retrieved document). A candidate write is attributed to
the sources it is *grounded in* (token containment). When it overlaps several sources
we take the **least-trusted** one - conservative taint propagation: if untrusted
content contributed, the write is untrusted. Writes grounded in nothing are labelled
`agent_reflection` (the model's own inference).

The attribution is computed by the trusted runtime, never by the model or by the
content itself, so an attacker cannot assert their own origin.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable

from .schema import ORIGIN_TRUST_DEFAULTS, Origin

_TOKEN = re.compile(r"[a-z0-9]+(?:[.@/_-][a-z0-9]+)*")
_STOP = frozenset("""
the a an and or but if then than that this these those to of in on at by for with from as is are was
were be been being it its into onto over under about after before also not no yes can could should would
will shall may might must do does did done has have had having you your yours i me my mine we us our
he she they them their his her user users prefers prefer likes like wants want always never
""".split())


@dataclass(frozen=True)
class Source:
    origin: Origin
    text: str


@dataclass(frozen=True)
class Attribution:
    origin: Origin
    score: float                       # containment of the winning source (0 if ungrounded)
    tainted_by: tuple[Origin, ...] = field(default_factory=tuple)  # every source above threshold


def content_tokens(text: str) -> set[str]:
    return {t for t in _TOKEN.findall(text.lower()) if len(t) > 2 and t not in _STOP}


def containment(candidate: set[str], source: set[str]) -> float:
    return len(candidate & source) / len(candidate) if candidate else 0.0


def attribute_origin(
    candidate_text: str,
    sources: Iterable[Source],
    min_score: float = 0.5,
    min_tokens: int = 2,
) -> Attribution:
    """Attribute `candidate_text` to the least-trusted source it is grounded in."""
    cand = content_tokens(candidate_text)
    if len(cand) < min_tokens:           # too little content to ground reliably
        return Attribution("agent_reflection", 0.0)
    hits: list[tuple[Origin, float]] = []
    for s in sources:
        score = containment(cand, content_tokens(s.text))
        if score >= min_score:
            hits.append((s.origin, score))
    if not hits:
        return Attribution("agent_reflection", 0.0)
    # least-trusted origin wins; ties broken by the stronger grounding score
    origin, score = min(hits, key=lambda h: (ORIGIN_TRUST_DEFAULTS[h[0]], -h[1]))
    return Attribution(origin, score, tuple(sorted({o for o, _ in hits})))
