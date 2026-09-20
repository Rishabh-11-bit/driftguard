"""Tests for the provenance schema."""
import pytest
from datetime import datetime, timezone
from driftguard.provenance.schema import ProvenanceRecord, make_provenance, ORIGIN_TRUST_DEFAULTS


def test_make_provenance_defaults():
    prov = make_provenance(
        origin="user_turn",
        session_id="sess-001",
        triggering_text="Hello, world!",
    )
    assert prov.origin == "user_turn"
    assert prov.trust_level == ORIGIN_TRUST_DEFAULTS["user_turn"]
    assert prov.session_id == "sess-001"
    assert len(prov.interaction_hash) == 64  # SHA-256 hex


def test_make_provenance_trust_override():
    prov = make_provenance("tool_output", "s", "text", trust_override=0.99)
    assert prov.trust_level == pytest.approx(0.99)


def test_provenance_invalid_trust():
    with pytest.raises(ValueError, match="trust_level"):
        ProvenanceRecord(
            origin="user_turn",
            trust_level=1.5,
            timestamp=datetime.now(timezone.utc),
            session_id="s",
            interaction_hash="abc",
        )


def test_provenance_roundtrip_dict():
    prov = make_provenance("agent_reflection", "s2", "some text")
    d = prov.to_dict()
    prov2 = ProvenanceRecord.from_dict(d)
    assert prov2.origin == prov.origin
    assert prov2.trust_level == pytest.approx(prov.trust_level)
    assert prov2.session_id == prov.session_id
    assert prov2.interaction_hash == prov.interaction_hash


def test_provenance_roundtrip_json():
    prov = make_provenance("retrieved_doc", "s3", "doc text")
    prov2 = ProvenanceRecord.from_json(prov.to_json())
    assert prov2.record_id == prov.record_id


def test_origin_index():
    prov = make_provenance("user_turn", "s", "t")
    assert isinstance(prov.origin_index, int)
    assert 0 <= prov.origin_index < len(ORIGIN_TRUST_DEFAULTS)


def test_all_origins_valid():
    for origin in ORIGIN_TRUST_DEFAULTS:
        prov = make_provenance(origin, "s", "text")
        assert prov.origin == origin
