"""
tests/test_audit.py

Audit-store tests: schema, reconstructability, redaction, resilience.

Run from the project root:
    python -m pytest tests/test_audit.py -v
"""

import json
import os
import sqlite3

import pytest

from src.audit import AUDIT_COLUMNS, REDACTED, AuditStore, redact, scrub_text


@pytest.fixture
def store(tmp_path):
    return AuditStore(str(tmp_path / "audit.db"))


def sample_kwargs(**overrides):
    payload = dict(
        transaction_id="TXN_TEST_0001",
        customer_id="C_TEST_001",
        input_features={"cart_value": 1000.0, "margin_percentage": 30.0, "category": "fashion"},
        model_output={"pred_p0": 0.30, "pred_p5": 0.42, "pred_optimal_discount": 5},
        policy_output={"decision": "APPROVED", "policy_selected_discount": 5,
                       "reason_code": "APPROVED",
                       "policy_checks": [{"check": "profit_floor", "discount": 5,
                                          "passed": True, "detail": "margin 25.00%"}]},
        final_decision="APPROVED",
        final_discount=5,
        reason_code="APPROVED",
        agent_explanation="A 5% discount clears the margin floor with enough lift.",
        model_version="t_learner_gbc_v1",
        policy_version="1.0.0",
        agent_model="scripted-model",
        success=True,
    )
    payload.update(overrides)
    return payload


# --------------------------------------------------------------------------------------
# SCHEMA / STORAGE
# --------------------------------------------------------------------------------------
def test_db_file_and_table_are_created(tmp_path):
    path = tmp_path / "nested" / "audit.db"
    AuditStore(str(path))
    assert os.path.exists(path)

    conn = sqlite3.connect(str(path))
    try:
        cols = [r[1] for r in conn.execute("PRAGMA table_info(decisions)")]
    finally:
        conn.close()
    for expected in AUDIT_COLUMNS:
        assert expected in cols


def test_init_db_is_idempotent(tmp_path):
    path = str(tmp_path / "audit.db")
    first = AuditStore(path)
    first.record_decision(**sample_kwargs())
    second = AuditStore(path)          # re-opening must not wipe or fail
    assert second.count() == 1


def test_record_and_get_decision(store):
    audit_id = store.record_decision(**sample_kwargs())
    assert isinstance(audit_id, int) and audit_id > 0

    row = store.get_decision(audit_id)
    assert row["final_decision"] == "APPROVED"
    assert row["final_discount"] == 5
    assert row["reason_code"] == "APPROVED"
    assert row["success"] == 1
    assert row["error_type"] is None
    assert row["timestamp"]


def test_reconstruct_returns_nested_structures(store):
    audit_id = store.record_decision(**sample_kwargs())
    record = store.reconstruct(audit_id)

    assert record["input_features"]["cart_value"] == 1000.0
    assert record["model_output"]["pred_p5"] == 0.42
    assert record["policy_output"]["policy_checks"][0]["check"] == "profit_floor"
    assert record["success"] is True


def test_fallback_record_stores_null_model_output(store):
    audit_id = store.record_decision(**sample_kwargs(
        model_output=None, final_decision="FALLBACK", final_discount=0,
        reason_code="MODEL_UNAVAILABLE", success=False, error_type="MODEL_TIMEOUT"))
    record = store.reconstruct(audit_id)

    assert record["model_output"] is None
    assert record["final_decision"] == "FALLBACK"
    assert record["final_discount"] == 0
    assert record["error_type"] == "MODEL_TIMEOUT"
    assert record["success"] is False


def test_list_decisions_orders_newest_first_and_filters_by_customer(store):
    store.record_decision(**sample_kwargs(transaction_id="T1", customer_id="C1"))
    store.record_decision(**sample_kwargs(transaction_id="T2", customer_id="C2"))
    store.record_decision(**sample_kwargs(transaction_id="T3", customer_id="C1"))

    assert [r["transaction_id"] for r in store.list_decisions()] == ["T3", "T2", "T1"]
    assert [r["transaction_id"] for r in store.list_decisions(customer_id="C1")] == ["T3", "T1"]
    assert len(store.list_decisions(limit=2)) == 2


def test_in_memory_store_works_for_fast_tests():
    store = AuditStore(":memory:")
    audit_id = store.record_decision(**sample_kwargs())
    assert store.get_decision(audit_id)["final_discount"] == 5


def test_get_missing_decision_returns_none(store):
    assert store.get_decision(999999) is None
    assert store.reconstruct(999999) is None


def test_unserializable_payload_does_not_break_the_write(store):
    class Opaque:
        def __repr__(self):
            return "<opaque>"

    audit_id = store.record_decision(**sample_kwargs(
        input_features={"cart_value": 1000.0, "weird": Opaque()}))
    record = store.reconstruct(audit_id)
    assert record is not None
    assert record["input_features"]["cart_value"] == 1000.0


# --------------------------------------------------------------------------------------
# REDACTION
# --------------------------------------------------------------------------------------
def test_redact_removes_values_under_sensitive_keys():
    dirty = {"cart_value": 100, "api_key": "abc123", "nested": {"AUTH_TOKEN": "xyz",
                                                                 "password": "hunter2"}}
    clean = redact(dirty)
    assert clean["cart_value"] == 100
    assert clean["api_key"] == REDACTED
    assert clean["nested"]["AUTH_TOKEN"] == REDACTED
    assert clean["nested"]["password"] == REDACTED


def test_scrub_text_removes_credential_shaped_strings():
    text = "debug sk-ant-api03-ABCDEFGHIJKLMN and Authorization: Bearer abcdefgh12345678"
    cleaned = scrub_text(text)
    assert "sk-ant-api03" not in cleaned
    assert "abcdefgh12345678" not in cleaned


def test_live_env_credentials_are_scrubbed(monkeypatch, store):
    secret = "totally-not-a-key-but-still-secret-value"
    monkeypatch.setenv("ANTHROPIC_API_KEY", secret)

    audit_id = store.record_decision(**sample_kwargs(
        agent_explanation=f"Approved. (leaked {secret})",
        input_features={"cart_value": 1000.0, "note": f"key {secret}"}))

    row = json.dumps(store.get_decision(audit_id))
    assert secret not in row
    assert REDACTED in row


def test_no_secret_columns_exist_in_the_schema(store):
    banned = ("api_key", "secret", "token", "password", "credential")
    for column in AUDIT_COLUMNS:
        assert not any(word in column.lower() for word in banned)


def test_redaction_preserves_decision_evidence(monkeypatch, store):
    """Redaction must not damage the fields a reviewer needs."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-api03-REDACTME12345678")
    audit_id = store.record_decision(**sample_kwargs())
    record = store.reconstruct(audit_id)

    assert record["model_output"]["pred_p5"] == 0.42
    assert record["policy_output"]["decision"] == "APPROVED"
    assert record["input_features"]["margin_percentage"] == 30.0
