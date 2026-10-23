"""Experiment storage. Definitions and results are immutable, enforced by the database.

An experiment freezes: the config, a snapshot of every dataset example, and a content
hash of both, so two experiments can always be compared on exactly what was run.
Results are append-only. Only `experiments.status` may change after creation.
"""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import threading
import time
from typing import Any, Dict, List, Optional, Set, Tuple

from .datasets import Corpus, Dataset, Doc, Example

SCHEMA = """
CREATE TABLE IF NOT EXISTS experiments (
    id           TEXT PRIMARY KEY,
    name         TEXT NOT NULL,
    created_at   REAL NOT NULL,
    status       TEXT NOT NULL,          -- created | running | partial | complete
    dataset_name TEXT NOT NULL,
    dataset_hash TEXT NOT NULL,
    config       TEXT NOT NULL,          -- JSON
    config_hash  TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS experiment_examples (
    experiment_id    TEXT NOT NULL,
    position         INTEGER NOT NULL,
    example_id       TEXT NOT NULL,
    input            TEXT NOT NULL,
    reference        TEXT,
    reference_docs   TEXT,               -- JSON
    relevant_doc_ids TEXT,               -- JSON
    metadata         TEXT,               -- JSON
    PRIMARY KEY (experiment_id, example_id)
);
CREATE TABLE IF NOT EXISTS results (
    experiment_id TEXT NOT NULL,
    variant       TEXT NOT NULL,
    example_id    TEXT NOT NULL,
    status        TEXT NOT NULL,         -- ok | error
    output        TEXT,
    error         TEXT,
    attempts      INTEGER NOT NULL,
    latency_ms    REAL,                  -- duration of the final provider call
    wall_ms       REAL NOT NULL,         -- including retries and backoff waits
    input_tokens  INTEGER,
    output_tokens INTEGER,
    cost_usd      REAL,
    trace_id      TEXT,
    created_at    REAL NOT NULL,
    PRIMARY KEY (experiment_id, variant, example_id)
);
CREATE INDEX IF NOT EXISTS idx_results_exp ON results(experiment_id);

CREATE TABLE IF NOT EXISTS scores (
    experiment_id TEXT NOT NULL,
    variant       TEXT NOT NULL,
    example_id    TEXT NOT NULL,
    metric        TEXT NOT NULL,         -- e.g. exact_match, token_f1, judge:<model>, embed_sim:<model>
    value         REAL,                  -- NULL = evaluator ran but could not score (see detail)
    detail        TEXT,                  -- JSON
    created_at    REAL NOT NULL,
    PRIMARY KEY (experiment_id, variant, example_id, metric)
);
CREATE INDEX IF NOT EXISTS idx_scores_exp ON scores(experiment_id);
CREATE TRIGGER IF NOT EXISTS scores_no_update BEFORE UPDATE ON scores
BEGIN SELECT RAISE(ABORT, 'scores are immutable'); END;
CREATE TRIGGER IF NOT EXISTS scores_no_delete BEFORE DELETE ON scores
BEGIN SELECT RAISE(ABORT, 'scores are immutable'); END;

CREATE TABLE IF NOT EXISTS experiment_corpus (
    experiment_id TEXT PRIMARY KEY,
    name          TEXT NOT NULL,
    content_hash  TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS experiment_docs (
    experiment_id TEXT NOT NULL,
    position      INTEGER NOT NULL,
    doc_id        TEXT NOT NULL,
    title         TEXT,
    text          TEXT NOT NULL,
    metadata      TEXT,                  -- JSON
    PRIMARY KEY (experiment_id, doc_id)
);
CREATE TABLE IF NOT EXISTS retrievals (
    experiment_id TEXT NOT NULL,
    variant       TEXT NOT NULL,
    example_id    TEXT NOT NULL,
    rank          INTEGER NOT NULL,      -- 1 = best
    doc_id        TEXT NOT NULL,
    score         REAL,
    PRIMARY KEY (experiment_id, variant, example_id, rank)
);
CREATE INDEX IF NOT EXISTS idx_retrievals_exp ON retrievals(experiment_id);
CREATE TRIGGER IF NOT EXISTS corpus_no_update BEFORE UPDATE ON experiment_corpus
BEGIN SELECT RAISE(ABORT, 'experiment corpus is immutable'); END;
CREATE TRIGGER IF NOT EXISTS corpus_no_delete BEFORE DELETE ON experiment_corpus
BEGIN SELECT RAISE(ABORT, 'experiment corpus is immutable'); END;
CREATE TRIGGER IF NOT EXISTS docs_no_update BEFORE UPDATE ON experiment_docs
BEGIN SELECT RAISE(ABORT, 'experiment documents are immutable'); END;
CREATE TRIGGER IF NOT EXISTS docs_no_delete BEFORE DELETE ON experiment_docs
BEGIN SELECT RAISE(ABORT, 'experiment documents are immutable'); END;
CREATE TRIGGER IF NOT EXISTS retrievals_no_update BEFORE UPDATE ON retrievals
BEGIN SELECT RAISE(ABORT, 'retrievals are immutable'); END;
CREATE TRIGGER IF NOT EXISTS retrievals_no_delete BEFORE DELETE ON retrievals
BEGIN SELECT RAISE(ABORT, 'retrievals are immutable'); END;

CREATE TABLE IF NOT EXISTS labels (
    experiment_id TEXT NOT NULL,
    variant       TEXT NOT NULL,
    example_id    TEXT NOT NULL,
    labeler       TEXT NOT NULL,
    label         REAL NOT NULL,         -- 0 incorrect, 0.5 partially correct, 1 correct (same scale as the judge)
    source        TEXT NOT NULL,         -- random | disagreement: how the item was chosen (matters for bias)
    note          TEXT,
    created_at    REAL NOT NULL,
    PRIMARY KEY (experiment_id, variant, example_id, labeler)
);
CREATE TRIGGER IF NOT EXISTS results_no_update BEFORE UPDATE ON results
BEGIN SELECT RAISE(ABORT, 'results are immutable'); END;
CREATE TRIGGER IF NOT EXISTS results_no_delete BEFORE DELETE ON results
BEGIN SELECT RAISE(ABORT, 'results are immutable'); END;
CREATE TRIGGER IF NOT EXISTS examples_no_update BEFORE UPDATE ON experiment_examples
BEGIN SELECT RAISE(ABORT, 'experiment examples are immutable'); END;
CREATE TRIGGER IF NOT EXISTS examples_no_delete BEFORE DELETE ON experiment_examples
BEGIN SELECT RAISE(ABORT, 'experiment examples are immutable'); END;
CREATE TRIGGER IF NOT EXISTS experiments_frozen
BEFORE UPDATE OF id, name, created_at, dataset_name, dataset_hash, config, config_hash ON experiments
BEGIN SELECT RAISE(ABORT, 'experiment definition is immutable'); END;
CREATE TRIGGER IF NOT EXISTS experiments_no_delete BEFORE DELETE ON experiments
BEGIN SELECT RAISE(ABORT, 'experiments are immutable'); END;
"""


def canonical_hash(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


class ExperimentStore:
    def __init__(self, path: str = ":memory:") -> None:
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        with self._lock:
            if path != ":memory:":
                self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.executescript(SCHEMA)

    # -- experiments --------------------------------------------------------
    def create_experiment(self, name: str, dataset: Dataset, config: Dict[str, Any],
                          corpus: Optional[Corpus] = None) -> str:
        exp_id = "exp_" + time.strftime("%Y%m%d_%H%M%S") + "_" + os.urandom(3).hex()
        with self._lock:
            try:
                self._conn.execute(
                    "INSERT INTO experiments VALUES (?,?,?,?,?,?,?,?)",
                    (exp_id, name, time.time(), "created", dataset.name, dataset.content_hash,
                     json.dumps(config), canonical_hash(config)),
                )
                self._conn.executemany(
                    "INSERT INTO experiment_examples VALUES (?,?,?,?,?,?,?,?)",
                    [(exp_id, i, e.id, e.input, e.reference, json.dumps(list(e.reference_docs)),
                      json.dumps(list(e.relevant_doc_ids)), json.dumps(e.metadata))
                     for i, e in enumerate(dataset.examples)],
                )
                if corpus is not None:
                    self._conn.execute("INSERT INTO experiment_corpus VALUES (?,?,?)",
                                       (exp_id, corpus.name, corpus.content_hash))
                    self._conn.executemany(
                        "INSERT INTO experiment_docs VALUES (?,?,?,?,?,?)",
                        [(exp_id, i, d.id, d.title, d.text, json.dumps(d.metadata))
                         for i, d in enumerate(corpus.docs)])
                self._conn.commit()
            except Exception:
                self._conn.rollback()
                raise
        return exp_id

    def get_experiment(self, exp_id: str) -> Dict[str, Any]:
        with self._lock:
            row = self._conn.execute("SELECT * FROM experiments WHERE id = ?", (exp_id,)).fetchone()
        if row is None:
            raise KeyError(f"no such experiment: {exp_id}")
        d = dict(row)
        d["config"] = json.loads(d["config"])
        return d

    def list_experiments(self, limit: int = 50) -> List[Dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT id, name, created_at, status, dataset_name FROM experiments "
                "ORDER BY created_at DESC LIMIT ?", (limit,)).fetchall()
        return [dict(r) for r in rows]

    def set_status(self, exp_id: str, status: str) -> None:
        with self._lock:
            self._conn.execute("UPDATE experiments SET status = ? WHERE id = ?", (status, exp_id))
            self._conn.commit()

    def get_examples(self, exp_id: str) -> List[Example]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM experiment_examples WHERE experiment_id = ? ORDER BY position",
                (exp_id,)).fetchall()
        return [Example(
            id=r["example_id"], input=r["input"], reference=r["reference"],
            reference_docs=tuple(json.loads(r["reference_docs"] or "[]")),
            relevant_doc_ids=tuple(json.loads(r["relevant_doc_ids"] or "[]")),
            metadata=json.loads(r["metadata"] or "{}")) for r in rows]

    # -- results ------------------------------------------------------------
    def add_result(self, r: Dict[str, Any]) -> bool:
        """Insert a result (and its retrieved documents, atomically).
        Returns False (and changes nothing) if a result for this pair already exists."""
        cols = ("experiment_id", "variant", "example_id", "status", "output", "error", "attempts",
                "latency_ms", "wall_ms", "input_tokens", "output_tokens", "cost_usd", "trace_id",
                "created_at")
        retrieved = r.get("retrieved") or []          # [(doc_id, score), ...] in rank order
        with self._lock:
            try:
                self._conn.execute(
                    f"INSERT INTO results ({','.join(cols)}) VALUES ({','.join('?' * len(cols))})",
                    [r.get(c) for c in cols])
                self._conn.executemany(
                    "INSERT INTO retrievals VALUES (?,?,?,?,?,?)",
                    [(r["experiment_id"], r["variant"], r["example_id"], i, doc_id, score)
                     for i, (doc_id, score) in enumerate(retrieved, 1)])
                self._conn.commit()
                return True
            except sqlite3.IntegrityError:
                self._conn.rollback()
                return False

    def done_keys(self, exp_id: str) -> Set[Tuple[str, str]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT variant, example_id FROM results WHERE experiment_id = ?", (exp_id,)).fetchall()
        return {(r["variant"], r["example_id"]) for r in rows}

    def get_results(self, exp_id: str, variant: Optional[str] = None) -> List[Dict[str, Any]]:
        sql, params = "SELECT * FROM results WHERE experiment_id = ?", [exp_id]
        if variant:
            sql, params = sql + " AND variant = ?", params + [variant]
        with self._lock:
            return [dict(r) for r in self._conn.execute(sql + " ORDER BY created_at", params)]

    # -- scores -------------------------------------------------------------
    def add_scores(self, rows: List[Dict[str, Any]]) -> int:
        """Append scores, skipping any that already exist. Returns how many were inserted."""
        inserted = 0
        with self._lock:
            for r in rows:
                try:
                    self._conn.execute(
                        "INSERT INTO scores VALUES (?,?,?,?,?,?,?)",
                        (r["experiment_id"], r["variant"], r["example_id"], r["metric"], r.get("value"),
                         json.dumps(r["detail"]) if r.get("detail") is not None else None,
                         r.get("created_at", time.time())))
                    inserted += 1
                except sqlite3.IntegrityError:
                    continue
            self._conn.commit()
        return inserted

    def scored_keys(self, exp_id: str) -> Set[Tuple[str, str, str]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT variant, example_id, metric FROM scores WHERE experiment_id = ?", (exp_id,)).fetchall()
        return {(r["variant"], r["example_id"], r["metric"]) for r in rows}

    def get_scores(self, exp_id: str) -> List[Dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM scores WHERE experiment_id = ?", (exp_id,)).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            d["detail"] = json.loads(d["detail"]) if d["detail"] else None
            out.append(d)
        return out

    # -- corpus and retrievals ---------------------------------------------
    def get_docs(self, exp_id: str) -> List[Doc]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM experiment_docs WHERE experiment_id = ? ORDER BY position", (exp_id,)).fetchall()
        return [Doc(id=r["doc_id"], text=r["text"], title=r["title"],
                    metadata=json.loads(r["metadata"] or "{}")) for r in rows]

    def get_corpus_info(self, exp_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            row = self._conn.execute("SELECT name, content_hash FROM experiment_corpus WHERE experiment_id = ?",
                                     (exp_id,)).fetchone()
        return dict(row) if row else None

    def get_retrievals(self, exp_id: str) -> Dict[Tuple[str, str], List[Tuple[str, float]]]:
        """(variant, example_id) -> [(doc_id, score), ...] in rank order."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT variant, example_id, doc_id, score FROM retrievals WHERE experiment_id = ? "
                "ORDER BY variant, example_id, rank", (exp_id,)).fetchall()
        out: Dict[Tuple[str, str], List[Tuple[str, float]]] = {}
        for r in rows:
            out.setdefault((r["variant"], r["example_id"]), []).append((r["doc_id"], r["score"]))
        return out

    # -- human labels ------------------------------------------------------
    def add_label(self, exp_id: str, variant: str, example_id: str, label: float, labeler: str = "human",
                  source: str = "random", note: Optional[str] = None) -> None:
        """Record (or correct) a human label. Unlike results and scores, labels may be replaced."""
        if label not in (0.0, 0.5, 1.0):
            raise ValueError("label must be 0, 0.5 or 1")
        if source not in ("random", "disagreement"):
            raise ValueError("source must be 'random' or 'disagreement'")
        with self._lock:
            self._conn.execute("INSERT OR REPLACE INTO labels VALUES (?,?,?,?,?,?,?,?)",
                               (exp_id, variant, example_id, labeler, float(label), source, note, time.time()))
            self._conn.commit()

    def get_labels(self, exp_id: str, labeler: Optional[str] = None) -> List[Dict[str, Any]]:
        sql, params = "SELECT * FROM labels WHERE experiment_id = ?", [exp_id]
        if labeler:
            sql, params = sql + " AND labeler = ?", params + [labeler]
        with self._lock:
            return [dict(r) for r in self._conn.execute(sql + " ORDER BY created_at", params)]
