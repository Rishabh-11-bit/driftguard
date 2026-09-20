"""Tests for the benchmark generators and labelling I/O."""
import pytest

from driftguard.benchmark.labelling import (
    BenchmarkEntry,
    LABEL_TO_INT,
    save_benchmark,
    load_benchmark,
    benchmark_stats,
)
from driftguard.benchmark.attack_generators.minja_style import generate_minja_entries
from driftguard.benchmark.attack_generators.agentpoison_style import generate_agentpoison_entries
from driftguard.benchmark.benign_generators.preference_drift import generate_preference_drift_entries
from driftguard.benchmark.benign_generators.ordinary_learning import generate_ordinary_learning_entries
from driftguard.benchmark.held_out_attacks import (
    generate_slow_drip_entries,
    generate_indirect_tool_injection_entries,
)

GENERATORS = [
    (generate_minja_entries, "malicious", "minja"),
    (generate_agentpoison_entries, "malicious", "agentpoison"),
    (generate_preference_drift_entries, "benign_drift", "preference_change"),
    (generate_ordinary_learning_entries, "ordinary_learning", "fact_learning"),
    (generate_slow_drip_entries, "malicious", "slow_drip"),
    (generate_indirect_tool_injection_entries, "malicious", "indirect_tool_injection"),
]


@pytest.mark.parametrize("generator,expected_label,expected_family", GENERATORS)
def test_generator_produces_correct_count(generator, expected_label, expected_family):
    entries = generator(n=5, seed=1)
    assert len(entries) == 5


@pytest.mark.parametrize("generator,expected_label,expected_family", GENERATORS)
def test_generator_labels_are_correct(generator, expected_label, expected_family):
    entries = generator(n=10, seed=1)
    for entry in entries:
        assert isinstance(entry, BenchmarkEntry)
        assert entry.label == expected_label
        assert entry.label_int == LABEL_TO_INT[expected_label]
        assert entry.attack_family == expected_family


@pytest.mark.parametrize("generator,expected_label,expected_family", GENERATORS)
def test_generator_entries_have_valid_structure(generator, expected_label, expected_family):
    entries = generator(n=5, seed=2)
    for entry in entries:
        assert entry.content and isinstance(entry.content, str)
        assert entry.session_id
        assert entry.entry_id
        assert isinstance(entry.provenance, dict)
        assert "origin" in entry.provenance
        assert "interaction_hash" in entry.provenance
        assert isinstance(entry.memory_entries, list)
        assert isinstance(entry.write_history, list)


@pytest.mark.parametrize("generator,expected_label,expected_family", GENERATORS)
def test_generator_is_deterministic_given_seed(generator, expected_label, expected_family):
    entries_a = generator(n=5, seed=7)
    entries_b = generator(n=5, seed=7)
    assert [e.content for e in entries_a] == [e.content for e in entries_b]


def test_save_and_load_benchmark_roundtrip(tmp_path):
    entries = generate_minja_entries(n=3, seed=1) + generate_ordinary_learning_entries(n=3, seed=1)
    path = tmp_path / "bench.jsonl"
    save_benchmark(entries, path)

    loaded = load_benchmark(path)
    assert len(loaded) == len(entries)
    assert [e.entry_id for e in loaded] == [e.entry_id for e in entries]
    assert [e.label for e in loaded] == [e.label for e in entries]


def test_benchmark_stats_counts_classes_and_families():
    entries = (
        generate_minja_entries(n=4, seed=1)
        + generate_agentpoison_entries(n=2, seed=1)
        + generate_ordinary_learning_entries(n=3, seed=1)
    )
    stats = benchmark_stats(entries)
    assert stats["total"] == 9
    assert stats["class_counts"]["malicious"] == 6
    assert stats["class_counts"]["ordinary_learning"] == 3
    assert stats["attack_families"]["minja"] == 4
    assert stats["attack_families"]["agentpoison"] == 2
