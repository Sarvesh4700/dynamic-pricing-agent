"""
src/audit.py

SQLite audit trail for the Dynamic Pricing Agent.

The LLM never writes here. src/agent_tools.py::ToolSession.write_audit assembles
the row from deterministic session state and calls record_decision(), so every
stored figure is traceable to a model or policy output.

Each row is sufficient to reconstruct: what the checkout looked like, what the ML
predicted, which policy rules fired, what was decided, and what the merchant was
told.

Secrets are structurally excluded: values are redacted by key name, by shape
(sk-... style tokens) and by exact match against known credential environment
variables before anything touches disk.
"""

import json
import os
import re
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Optional

DEFAULT_DB_PATH = os.path.join("data", "audit.db")

SCHEMA = """
CREATE TABLE IF NOT EXISTS decisions (
    id                   INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp            TEXT    NOT NULL,
    transaction_id       TEXT,
    customer_id          TEXT,
    input_features_json  TEXT,
    model_output_json    TEXT,
    policy_output_json   TEXT,
    final_decision       TEXT    NOT NULL,
    final_discount       INTEGER NOT NULL,
    reason_code          TEXT,
    agent_explanation    TEXT,
    explanation_source   TEXT,
    model_version        TEXT,
    policy_version       TEXT,
    agent_model          TEXT,
    success              INTEGER NOT NULL,
    error_type           TEXT,
    latency_ms           REAL
);
CREATE INDEX IF NOT EXISTS idx_decisions_customer ON decisions (customer_id);
CREATE INDEX IF NOT EXISTS idx_decisions_txn      ON decisions (transaction_id);
CREATE INDEX IF NOT EXISTS idx_decisions_time     ON decisions (timestamp);
"""

AUDIT_COLUMNS = (
    "id", "timestamp", "transaction_id", "customer_id",
    "input_features_json", "model_output_json", "policy_output_json",
    "final_decision", "final_discount", "reason_code", "agent_explanation",
    "explanation_source", "model_version", "policy_version", "agent_model",
    "success", "error_type", "latency_ms",
)

REDACTED = "[REDACTED]"

# Keys whose values are never stored, whatever they contain.
_SENSITIVE_KEY_RE = re.compile(
    r"(api[_\-]?key|secret|token|password|passwd|credential|authorization|bearer|private[_\-]?key)",
    re.IGNORECASE,
)
# Credential-shaped values, stored nowhere even under an innocent key name.
_SENSITIVE_VALUE_RES = (
    re.compile(r"sk-ant-[A-Za-z0-9\-_]{4,}"),
    re.compile(r"sk-[A-Za-z0-9]{16,}"),
    re.compile(r"\bBearer\s+[A-Za-z0-9\-._~+/]{8,}", re.IGNORECASE),
)
# Env vars whose literal values are scrubbed on exact substring match.
_SECRET_ENV_VARS = ("ANTHROPIC_API_KEY", "OPENAI_API_KEY",
                    "RAZORPAY_KEY_SECRET", "RAZORPAY_KEY_ID")


def scrub_text(value: str) -> str:
    """Remove credential-shaped substrings and known live credentials."""
    if not isinstance(value, str) or not value:
        return value
    out = value
    for env_var in _SECRET_ENV_VARS:
        secret = os.getenv(env_var)
        if secret and len(secret) >= 8 and secret in out:
            out = out.replace(secret, REDACTED)
    for pattern in _SENSITIVE_VALUE_RES:
        out = pattern.sub(REDACTED, out)
    return out


def redact(obj: Any) -> Any:
    """Recursively redact a JSON-ish structure by key name and value shape."""
    if isinstance(obj, dict):
        clean = {}
        for key, value in obj.items():
            if isinstance(key, str) and _SENSITIVE_KEY_RE.search(key):
                clean[key] = REDACTED
            else:
                clean[key] = redact(value)
        return clean
    if isinstance(obj, (list, tuple)):
        return [redact(item) for item in obj]
    if isinstance(obj, str):
        return scrub_text(obj)
    return obj


def _dump_json(obj: Any) -> Optional[str]:
    if obj is None:
        return None
    try:
        return json.dumps(redact(obj), default=str, sort_keys=True)
    except Exception as exc:
        return json.dumps({"_serialization_error": type(exc).__name__})


class AuditStore:
    """Append-mostly SQLite store. Safe for a file path or ':memory:'."""

    def __init__(self, db_path: str = DEFAULT_DB_PATH):
        self.db_path = db_path
        self._lock = threading.Lock()
        self._memory_conn = None
        if db_path != ":memory:":
            parent = os.path.dirname(os.path.abspath(db_path))
            if parent:
                os.makedirs(parent, exist_ok=True)
        self.init_db()

    @contextmanager
    def _connect(self):
        if self.db_path == ":memory:":
            if self._memory_conn is None:
                self._memory_conn = sqlite3.connect(":memory:", check_same_thread=False)
                self._memory_conn.row_factory = sqlite3.Row
            yield self._memory_conn
            self._memory_conn.commit()
            return
        conn = sqlite3.connect(self.db_path, timeout=10.0)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def init_db(self) -> None:
        with self._lock, self._connect() as conn:
            conn.executescript(SCHEMA)

    # ----------------------------------------------------------------------
    def record_decision(self, *, transaction_id: Optional[str], customer_id: Optional[str],
                        input_features: Optional[dict], model_output: Optional[dict],
                        policy_output: Optional[dict], final_decision: str,
                        final_discount: int, reason_code: Optional[str],
                        agent_explanation: Optional[str], model_version: Optional[str],
                        policy_version: Optional[str], agent_model: Optional[str],
                        success: bool, error_type: Optional[str] = None,
                        explanation_source: Optional[str] = None,
                        latency_ms: Optional[float] = None,
                        timestamp: Optional[str] = None) -> Optional[int]:
        """Insert one decision. Returns the audit id, or None if the write failed
        - a failed audit write must not break a checkout, but it is never silent
        for the caller, which sees None."""
        row = (
            timestamp or datetime.now(timezone.utc).isoformat(),
            transaction_id,
            customer_id,
            _dump_json(input_features),
            _dump_json(model_output),
            _dump_json(policy_output),
            str(final_decision),
            int(final_discount or 0),
            reason_code,
            scrub_text(agent_explanation or ""),
            explanation_source,
            model_version,
            policy_version,
            agent_model,
            1 if success else 0,
            error_type,
            float(latency_ms) if latency_ms is not None else None,
        )
        try:
            with self._lock, self._connect() as conn:
                cursor = conn.execute(
                    """
                    INSERT INTO decisions (
                        timestamp, transaction_id, customer_id,
                        input_features_json, model_output_json, policy_output_json,
                        final_decision, final_discount, reason_code, agent_explanation,
                        explanation_source, model_version, policy_version, agent_model,
                        success, error_type, latency_ms
                    ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                    """,
                    row,
                )
                return int(cursor.lastrowid)
        except sqlite3.Error:
            return None

    # ----------------------------------------------------------------------
    def get_decision(self, audit_id: int) -> Optional[dict]:
        with self._lock, self._connect() as conn:
            cur = conn.execute("SELECT * FROM decisions WHERE id = ?", (audit_id,))
            row = cur.fetchone()
        return dict(row) if row else None

    def list_decisions(self, limit: int = 20, customer_id: Optional[str] = None) -> list:
        sql = "SELECT * FROM decisions"
        params: list = []
        if customer_id:
            sql += " WHERE customer_id = ?"
            params.append(customer_id)
        sql += " ORDER BY id DESC LIMIT ?"
        params.append(int(limit))
        with self._lock, self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [dict(r) for r in rows]

    def count(self) -> int:
        with self._lock, self._connect() as conn:
            return int(conn.execute("SELECT COUNT(*) FROM decisions").fetchone()[0])

    def reconstruct(self, audit_id: int) -> Optional[dict]:
        """Rehydrate a row into nested dicts - the reviewability guarantee."""
        row = self.get_decision(audit_id)
        if row is None:
            return None

        def _load(raw):
            if not raw:
                return None
            try:
                return json.loads(raw)
            except json.JSONDecodeError:
                return {"_unparseable": True}

        row["input_features"] = _load(row.pop("input_features_json"))
        row["model_output"] = _load(row.pop("model_output_json"))
        row["policy_output"] = _load(row.pop("policy_output_json"))
        row["success"] = bool(row["success"])
        return row
