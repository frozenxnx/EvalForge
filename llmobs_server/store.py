"""Span store on SQLite (a file path) or Postgres (a postgres:// URL), see llmobs_eval.db.

Spans are upserted by span_id, so retried batches are idempotent.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from llmobs_eval.db import Database

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

# Same table on Postgres: nanosecond timestamps need BIGINT; COLLATE "C" sorts text like SQLite does.
PG_SCHEMA = """
CREATE TABLE IF NOT EXISTS spans (
    span_id     TEXT COLLATE "C" PRIMARY KEY,
    trace_id    TEXT COLLATE "C" NOT NULL,
    parent_id   TEXT COLLATE "C",
    name        TEXT COLLATE "C" NOT NULL,
    kind        TEXT COLLATE "C" NOT NULL DEFAULT 'chain',
    service     TEXT COLLATE "C",
    start_ns    BIGINT NOT NULL,
    end_ns      BIGINT,
    status      TEXT COLLATE "C" NOT NULL DEFAULT 'ok',
    error       TEXT,
    input       TEXT,
    output      TEXT,
    attributes  TEXT
);
CREATE INDEX IF NOT EXISTS idx_spans_trace ON spans(trace_id);
CREATE INDEX IF NOT EXISTS idx_spans_start ON spans(start_ns);
"""

_J = json.dumps
_COLS = ("span_id", "trace_id", "parent_id", "name", "kind", "service", "start_ns", "end_ns", "status", "error",
         "input", "output", "attributes")
_UPSERT = (f"INSERT INTO spans ({','.join(_COLS)}) VALUES ({','.join('?' * len(_COLS))}) "
           f"ON CONFLICT (span_id) DO UPDATE SET " + ", ".join(f"{c} = excluded.{c}" for c in _COLS[1:]))


def _token_sum(attr: str, dialect: str) -> str:
    if dialect == "postgres":
        return f"SUM(COALESCE((attributes::jsonb ->> '{attr}')::bigint, 0))::bigint"
    return f"SUM(COALESCE(json_extract(attributes, '$.\"{attr}\"'), 0))"


class SpanStore:
    def __init__(self, path: str = ":memory:") -> None:
        self.db = Database(path, SCHEMA, PG_SCHEMA)

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
        with self.db.transaction() as tx:
            tx.executemany(_UPSERT, rows)
        return len(rows)

    def get_trace(self, trace_id: str) -> List[Dict[str, Any]]:
        rows = self.db.query("SELECT * FROM spans WHERE trace_id = ? ORDER BY start_ns", (trace_id,))
        return [self._row_to_span(r) for r in rows]

    def list_traces(self, limit: int = 50, offset: int = 0,
                    service: Optional[str] = None) -> List[Dict[str, Any]]:
        where, params = "", []
        if service:
            where, params = "WHERE service = ?", [service]
        d = self.db.dialect
        sql = f"""
            SELECT trace_id,
                   MIN(start_ns)                                            AS start_ns,
                   MAX(COALESCE(end_ns, start_ns))                          AS end_ns,
                   COUNT(*)                                                 AS span_count,
                   SUM(CASE WHEN status = 'error' THEN 1 ELSE 0 END)        AS error_count,
                   {_token_sum('gen_ai.usage.input_tokens', d)}             AS input_tokens,
                   {_token_sum('gen_ai.usage.output_tokens', d)}            AS output_tokens,
                   MAX(CASE WHEN parent_id IS NULL THEN name END)           AS root_name,
                   MAX(service)                                             AS service
            FROM spans {where}
            GROUP BY trace_id
            ORDER BY start_ns DESC
            LIMIT ? OFFSET ?
        """
        out = []
        for r in self.db.query(sql, params + [limit, offset]):
            r = {k: int(v) if k.endswith("_count") and v is not None else v for k, v in r.items()}
            r["duration_ms"] = (r["end_ns"] - r["start_ns"]) / 1e6
            out.append(r)
        return out

    @staticmethod
    def _row_to_span(r: Dict[str, Any]) -> Dict[str, Any]:
        d = dict(r)
        for k in ("input", "output", "attributes"):
            d[k] = json.loads(d[k]) if d[k] is not None else None
        return d
