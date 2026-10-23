"""SQLite-backed span store (stdlib only).

Kept behind a small interface so a Postgres/ClickHouse implementation can be
dropped in later without touching the API layer. Spans are upserted by span_id,
so retried batches are idempotent.
"""
from __future__ import annotations

import json
import sqlite3
import threading
from typing import Any, Dict, List, Optional

SCHEMA = """
CREATE TABLE IF NOT EXISTS spans (
    span_id     TEXT PRIMARY KEY,
    trace_id    TEXT NOT NULL,
    parent_id   TEXT,
    name        TEXT NOT NULL,
    kind        TEXT NOT NULL DEFAULT 'chain',
    service     TEXT,
    start_ns    INTEGER NOT NULL,
    end_ns      INTEGER,
    status      TEXT NOT NULL DEFAULT 'ok',
    error       TEXT,
    input       TEXT,   -- JSON
    output      TEXT,   -- JSON
    attributes  TEXT    -- JSON
);
CREATE INDEX IF NOT EXISTS idx_spans_trace ON spans(trace_id);
CREATE INDEX IF NOT EXISTS idx_spans_start ON spans(start_ns);
"""

_J = json.dumps


class SpanStore:
    def __init__(self, path: str = ":memory:") -> None:
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        with self._lock:
            self._conn.executescript(SCHEMA)

    def ingest(self, spans: List[Dict[str, Any]]) -> int:
        rows = [
            (
                s["span_id"], s["trace_id"], s.get("parent_id"), s["name"],
                s.get("kind") or "chain", s.get("service"), s["start_ns"], s.get("end_ns"),
                s.get("status") or "ok", s.get("error"),
                _J(s.get("input")), _J(s.get("output")), _J(s.get("attributes") or {}),
            )
            for s in spans
        ]
        with self._lock:
            self._conn.executemany(
                "INSERT OR REPLACE INTO spans VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)", rows
            )
            self._conn.commit()
        return len(rows)

    def get_trace(self, trace_id: str) -> List[Dict[str, Any]]:
        with self._lock:
            cur = self._conn.execute(
                "SELECT * FROM spans WHERE trace_id = ? ORDER BY start_ns", (trace_id,)
            )
            return [self._row_to_span(r) for r in cur.fetchall()]

    def list_traces(self, limit: int = 50, offset: int = 0,
                    service: Optional[str] = None) -> List[Dict[str, Any]]:
        where, params = "", []
        if service:
            where, params = "WHERE service = ?", [service]
        sql = f"""
            SELECT trace_id,
                   MIN(start_ns)                                   AS start_ns,
                   MAX(COALESCE(end_ns, start_ns))                 AS end_ns,
                   COUNT(*)                                        AS span_count,
                   SUM(status = 'error')                           AS error_count,
                   SUM(COALESCE(json_extract(attributes, '$."gen_ai.usage.input_tokens"'), 0))  AS input_tokens,
                   SUM(COALESCE(json_extract(attributes, '$."gen_ai.usage.output_tokens"'), 0)) AS output_tokens,
                   MAX(CASE WHEN parent_id IS NULL THEN name END)  AS root_name,
                   MAX(service)                                    AS service
            FROM spans {where}
            GROUP BY trace_id
            ORDER BY start_ns DESC
            LIMIT ? OFFSET ?
        """
        with self._lock:
            cur = self._conn.execute(sql, params + [limit, offset])
            out = []
            for r in cur.fetchall():
                d = dict(r)
                d["duration_ms"] = (d["end_ns"] - d["start_ns"]) / 1e6
                out.append(d)
            return out

    @staticmethod
    def _row_to_span(r: sqlite3.Row) -> Dict[str, Any]:
        d = dict(r)
        for k in ("input", "output", "attributes"):
            d[k] = json.loads(d[k]) if d[k] is not None else None
        return d
