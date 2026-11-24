"""One small database layer for both stores: SQLite (a file, or ":memory:") or Postgres (a URL).

SQL is written once, SQLite-style with `?` placeholders and standard `ON CONFLICT` clauses (SQLite >= 3.24),
and translated for Postgres. Only the schema differs per dialect (types, and triggers that keep the
append-only tables append-only).

    db = Database("llmobs.db", sqlite_schema, postgres_schema)       # or "postgresql://user:pw@host/db"
    rows = db.query("SELECT * FROM t WHERE id = ?", (1,))           # list of dicts
    with db.transaction() as tx:                                     # all or nothing
        tx.execute("INSERT INTO t VALUES (?, ?)", (1, "a"))

A Postgres connection is opened lazily, in autocommit mode with explicit transactions, without prepared
statements (so it works behind PgBouncer / Neon's pooled endpoint), and is reopened once if a serverless
pause dropped it between requests.
"""
from __future__ import annotations

import sqlite3
import threading
from contextlib import contextmanager
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

POSTGRES_PREFIXES = ("postgres://", "postgresql://")


def is_postgres(target: str) -> bool:
    return target.startswith(POSTGRES_PREFIXES)


def database_target(default: str = "llmobs.db") -> str:
    """Where the server keeps its data: $LLMOBS_DB, else $DATABASE_URL / $POSTGRES_URL (set by Vercel's Neon
    integration), else a local SQLite file."""
    import os
    for var in ("LLMOBS_DB", "DATABASE_URL", "POSTGRES_URL"):
        if os.environ.get(var):
            return os.environ[var]
    return default


def resolve_target(target: str) -> Tuple[str, Optional[str]]:
    """(connection target, Postgres schema or None). Tests replace this to run the suite on Postgres."""
    return target, None


class _Tx:
    def __init__(self, db: "Database", cur: Any) -> None:
        self._db, self._cur = db, cur

    def execute(self, sql: str, params: Sequence[Any] = ()) -> int:
        self._cur.execute(self._db.sql(sql), tuple(params))
        return self._cur.rowcount

    def executemany(self, sql: str, rows: Sequence[Sequence[Any]]) -> None:
        rows = [tuple(r) for r in rows]
        if rows:
            self._cur.executemany(self._db.sql(sql), rows)

    def query(self, sql: str, params: Sequence[Any] = ()) -> List[Dict[str, Any]]:
        self._cur.execute(self._db.sql(sql), tuple(params))
        return [dict(r) for r in self._cur.fetchall()]

    def one(self, sql: str, params: Sequence[Any] = ()) -> Optional[Dict[str, Any]]:
        rows = self.query(sql, params)
        return rows[0] if rows else None


class Database:
    def __init__(self, target: str, sqlite_schema: str, postgres_schema: str) -> None:
        self.target, self.schema = resolve_target(target)
        self.dialect = "postgres" if is_postgres(self.target) else "sqlite"
        self._lock = threading.RLock()
        self._conn: Any = None
        if self.dialect == "postgres":
            try:
                import psycopg  # noqa: F401
            except ImportError:
                raise RuntimeError("a postgres:// database needs psycopg: pip install 'psycopg[binary]'") from None
            from psycopg import errors
            # A trigger refusing a change raises RaiseException; a duplicate key raises UniqueViolation.
            self.integrity_errors: Tuple[type, ...] = (errors.IntegrityError, errors.RaiseException)
            self._connect()
            self._ensure_pg_schema(postgres_schema)
        else:
            self.integrity_errors = (sqlite3.IntegrityError,)
            self._connect()
            with self._lock:
                if self.target != ":memory:":
                    self._conn.execute("PRAGMA journal_mode=WAL")
                self._conn.executescript(sqlite_schema)

    def _ensure_pg_schema(self, ddl: str) -> None:
        """Apply the DDL once per version. A serverless cold start then costs one query, not a round of DDL
        (CREATE OR REPLACE TRIGGER takes table locks); concurrent cold starts serialize on an advisory lock."""
        import hashlib
        version = hashlib.sha256(ddl.encode()).hexdigest()[:16]
        if self._conn.execute("SELECT to_regclass('llmobs_schema') IS NOT NULL AS ok").fetchone()["ok"] and \
                self.one("SELECT 1 AS ok FROM llmobs_schema WHERE version = ?", (version,)):
            return
        with self.transaction() as tx:
            tx.execute("SELECT pg_advisory_xact_lock(4210331)")
            tx._cur.execute("CREATE TABLE IF NOT EXISTS llmobs_schema (version TEXT PRIMARY KEY, applied_at TIMESTAMPTZ "
                            "NOT NULL DEFAULT now())")
            if tx.one("SELECT 1 AS ok FROM llmobs_schema WHERE version = ?", (version,)):
                return
            tx._cur.execute(ddl)
            tx.execute("INSERT INTO llmobs_schema (version) VALUES (?)", (version,))

    # -- connection ------------------------------------------------------------------------------------------
    def _connect(self) -> None:
        if self.dialect == "sqlite":
            self._conn = sqlite3.connect(self.target, check_same_thread=False, isolation_level=None)
            self._conn.row_factory = sqlite3.Row
            return
        import psycopg
        from psycopg.rows import dict_row
        kwargs: Dict[str, Any] = {"autocommit": True, "row_factory": dict_row, "prepare_threshold": None,
                                  "connect_timeout": 10}
        if self.schema:
            kwargs["options"] = f"-c search_path={self.schema}"
        self._conn = psycopg.connect(self.target, **kwargs)
        if self.schema:
            self._conn.execute(f'CREATE SCHEMA IF NOT EXISTS "{self.schema}"')

    def _alive(self) -> None:
        """Reopen a Postgres connection that was closed while idle (a paused serverless function, a pooler)."""
        if self.dialect == "postgres" and self._conn.closed:
            self._connect()

    def close(self) -> None:
        with self._lock:
            if self._conn is not None:
                self._conn.close()

    def sql(self, sql: str) -> str:
        return sql.replace("%", "%%").replace("?", "%s") if self.dialect == "postgres" else sql

    # -- statements --------------------------------------------------------------------------------------------
    @contextmanager
    def transaction(self) -> Iterator[_Tx]:
        with self._lock:
            self._alive()
            if self.dialect == "sqlite":
                cur = self._conn.cursor()
                cur.execute("BEGIN")
                try:
                    yield _Tx(self, cur)
                    cur.execute("COMMIT")
                except BaseException:
                    cur.execute("ROLLBACK")
                    raise
            else:
                with self._conn.transaction(), self._conn.cursor() as cur:
                    yield _Tx(self, cur)

    def _run(self, fn):
        """Run a single-statement call, retrying once on a dropped Postgres connection (never mid-transaction)."""
        with self._lock:
            self._alive()
            try:
                return fn()
            except Exception as exc:
                if self.dialect == "postgres" and self._conn.closed:
                    import psycopg
                    if isinstance(exc, psycopg.OperationalError):
                        self._connect()
                        return fn()
                raise

    def query(self, sql: str, params: Sequence[Any] = ()) -> List[Dict[str, Any]]:
        def go():
            cur = self._conn.execute(self.sql(sql), tuple(params))
            return [dict(r) for r in cur.fetchall()]
        return self._run(go)

    def one(self, sql: str, params: Sequence[Any] = ()) -> Optional[Dict[str, Any]]:
        rows = self.query(sql, params)
        return rows[0] if rows else None

    def execute(self, sql: str, params: Sequence[Any] = ()) -> int:
        """One write statement in its own transaction; returns the affected row count."""
        with self.transaction() as tx:
            return tx.execute(sql, params)

    def columns(self, table: str) -> List[str]:
        if self.dialect == "sqlite":
            return [r["name"] for r in self.query(f"PRAGMA table_info({table})")]
        return [r["column_name"] for r in self.query(
            "SELECT column_name FROM information_schema.columns WHERE table_schema = current_schema() "
            "AND table_name = ? ORDER BY ordinal_position", (table,))]
