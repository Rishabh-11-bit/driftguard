"""Tests for driftguard/benchmark/hardening.py."""

import numpy as np

from driftguard.benchmark.attack_generators.minja_style import generate_minja_entries
from driftguard.benchmark.benign_generators.ordinary_learning import generate_ordinary_learning_entries
from driftguard.benchmark.hardening import harden_entries


def _entries():
    return generate_minja_entries(n=20, seed=1) + generate_ordinary_learning_entries(n=20, seed=2)


def test_labels_and_count_preserved():
    src = _entries()
    out = harden_entries(src, camouflage=1.0, seed=0)
    assert len(out) == len(src)
    assert [e.label_int for e in out] == [e.label_int for e in src]


def test_inputs_not_mutated():
    src = _entries()
    before = [(e.content, e.provenance["trust_level"]) for e in src]
    harden_entries(src, camouflage=1.0, seed=0)
    assert before == [(e.content, e.provenance["trust_level"]) for e in src]


def test_deterministic_for_seed():
    a = harden_entries(_entries(), 0.7, seed=5)
    b = harden_entries(_entries(), 0.7, seed=5)
    assert [e.content for e in a] == [e.content for e in b]


def test_zero_camouflage_keeps_content_but_scrubs_metadata():
    src = _entries()
    out = harden_entries(src, camouflage=0.0, seed=0)
    assert [e.content for e in out] == [e.content for e in src]
    assert all("attack_type" not in e.provenance["metadata"] for e in out)


def test_full_camouflage_lengthens_content_and_keeps_original_text():
    src = _entries()
    out = harden_entries(src, camouflage=1.0, seed=0)
    assert all(o.content.startswith(s.content) and len(o.content) > len(s.content) for s, o in zip(src, out))


def test_trust_no_longer_separates_classes():
    out = harden_entries(_entries(), camouflage=1.0, seed=0)
    mal = np.mean([e.provenance["trust_level"] for e in out if e.label_int == 0])
    ord_ = np.mean([e.provenance["trust_level"] for e in out if e.label_int == 2])
    assert abs(mal - ord_) < 0.2
