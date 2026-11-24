"""Experiment storage. Definitions and results are immutable, enforced by the database.

An experiment freezes: the config, a snapshot of every dataset example, and a content
hash of both, so two experiments can always be compared on exactly what was run.
Results are append-only. Only `experiments.status` may change after creation.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from typing import Any, Dict, List, Optional, Set, Tuple

from .datasets import Corpus, Dataset, Doc, Example
from .db import Database

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
CREATE TABLE IF NOT EXISTS hidden_metrics (
    experiment_id TEXT NOT NULL,
    metric        TEXT NOT NULL,
    PRIMARY KEY (experiment_id, metric)
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


# The same tables on Postgres. Types: BIGINT for integers, DOUBLE PRECISION for reals, COLLATE "C" so text
# sorts byte-wise as in SQLite. One trigger function enforces append-only tables, as the SQLite triggers do.
PG_SCHEMA = """
CREATE TABLE IF NOT EXISTS experiments (
    id TEXT COLLATE "C" PRIMARY KEY, name TEXT NOT NULL, created_at DOUBLE PRECISION NOT NULL,
    status TEXT NOT NULL, dataset_name TEXT NOT NULL, dataset_hash TEXT NOT NULL, config TEXT NOT NULL,
    config_hash TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS experiment_examples (
    experiment_id TEXT COLLATE "C" NOT NULL, position BIGINT NOT NULL, example_id TEXT COLLATE "C" NOT NULL,
    input TEXT NOT NULL, reference TEXT, reference_docs TEXT, relevant_doc_ids TEXT, metadata TEXT,
    PRIMARY KEY (experiment_id, example_id));
CREATE TABLE IF NOT EXISTS results (
    experiment_id TEXT COLLATE "C" NOT NULL, variant TEXT COLLATE "C" NOT NULL, example_id TEXT COLLATE "C" NOT NULL,
    status TEXT NOT NULL, output TEXT, error TEXT, attempts BIGINT NOT NULL, latency_ms DOUBLE PRECISION,
    wall_ms DOUBLE PRECISION NOT NULL, input_tokens BIGINT, output_tokens BIGINT, cost_usd DOUBLE PRECISION,
    trace_id TEXT, created_at DOUBLE PRECISION NOT NULL,
    PRIMARY KEY (experiment_id, variant, example_id));
CREATE INDEX IF NOT EXISTS idx_results_exp ON results(experiment_id);
CREATE TABLE IF NOT EXISTS scores (
    experiment_id TEXT COLLATE "C" NOT NULL, variant TEXT COLLATE "C" NOT NULL, example_id TEXT COLLATE "C" NOT NULL,
    metric TEXT COLLATE "C" NOT NULL, value DOUBLE PRECISION, detail TEXT, created_at DOUBLE PRECISION NOT NULL,
    PRIMARY KEY (experiment_id, variant, example_id, metric));
CREATE INDEX IF NOT EXISTS idx_scores_exp ON scores(experiment_id);
CREATE TABLE IF NOT EXISTS experiment_corpus (
    experiment_id TEXT COLLATE "C" PRIMARY KEY, name TEXT NOT NULL, content_hash TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS experiment_docs (
    experiment_id TEXT COLLATE "C" NOT NULL, position BIGINT NOT NULL, doc_id TEXT COLLATE "C" NOT NULL,
    title TEXT, text TEXT NOT NULL, metadata TEXT, PRIMARY KEY (experiment_id, doc_id));
CREATE TABLE IF NOT EXISTS retrievals (
    experiment_id TEXT COLLATE "C" NOT NULL, variant TEXT COLLATE "C" NOT NULL, example_id TEXT COLLATE "C" NOT NULL,
    rank BIGINT NOT NULL, doc_id TEXT COLLATE "C" NOT NULL, score DOUBLE PRECISION,
    PRIMARY KEY (experiment_id, variant, example_id, rank));
CREATE INDEX IF NOT EXISTS idx_retrievals_exp ON retrievals(experiment_id);
CREATE TABLE IF NOT EXISTS labels (
    experiment_id TEXT COLLATE "C" NOT NULL, variant TEXT COLLATE "C" NOT NULL, example_id TEXT COLLATE "C" NOT NULL,
    labeler TEXT COLLATE "C" NOT NULL, label DOUBLE PRECISION NOT NULL, source TEXT NOT NULL, note TEXT,
    created_at DOUBLE PRECISION NOT NULL, PRIMARY KEY (experiment_id, variant, example_id, labeler));
CREATE TABLE IF NOT EXISTS hidden_metrics (
    experiment_id TEXT COLLATE "C" NOT NULL, metric TEXT COLLATE "C" NOT NULL, PRIMARY KEY (experiment_id, metric));

CREATE OR REPLACE FUNCTION llmobs_refuse() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN RAISE EXCEPTION '%', TG_ARGV[0]; END $$;
CREATE OR REPLACE TRIGGER scores_no_update BEFORE UPDATE ON scores FOR EACH ROW EXECUTE FUNCTION llmobs_refuse('scores are immutable');
CREATE OR REPLACE TRIGGER scores_no_delete BEFORE DELETE ON scores FOR EACH ROW EXECUTE FUNCTION llmobs_refuse('scores are immutable');
CREATE OR REPLACE TRIGGER corpus_no_update BEFORE UPDATE ON experiment_corpus FOR EACH ROW EXECUTE FUNCTION llmobs_refuse('experiment corpus is immutable');
CREATE OR REPLACE TRIGGER corpus_no_delete BEFORE DELETE ON experiment_corpus FOR EACH ROW EXECUTE FUNCTION llmobs_refuse('experiment corpus is immutable');
CREATE OR REPLACE TRIGGER docs_no_update BEFORE UPDATE ON experiment_docs FOR EACH ROW EXECUTE FUNCTION llmobs_refuse('experiment documents are immutable');
CREATE OR REPLACE TRIGGER docs_no_delete BEFORE DELETE ON experiment_docs FOR EACH ROW EXECUTE FUNCTION llmobs_refuse('experiment documents are immutable');
CREATE OR REPLACE TRIGGER retrievals_no_update BEFORE UPDATE ON retrievals FOR EACH ROW EXECUTE FUNCTION llmobs_refuse('retrievals are immutable');
CREATE OR REPLACE TRIGGER retrievals_no_delete BEFORE DELETE ON retrievals FOR EACH ROW EXECUTE FUNCTION llmobs_refuse('retrievals are immutable');
CREATE OR REPLACE TRIGGER results_no_update BEFORE UPDATE ON results FOR EACH ROW EXECUTE FUNCTION llmobs_refuse('results are immutable');
CREATE OR REPLACE TRIGGER results_no_delete BEFORE DELETE ON results FOR EACH ROW EXECUTE FUNCTION llmobs_refuse('results are immutable');
CREATE OR REPLACE TRIGGER examples_no_update BEFORE UPDATE ON experiment_examples FOR EACH ROW EXECUTE FUNCTION llmobs_refuse('experiment examples are immutable');
CREATE OR REPLACE TRIGGER examples_no_delete BEFORE DELETE ON experiment_examples FOR EACH ROW EXECUTE FUNCTION llmobs_refuse('experiment examples are immutable');
CREATE OR REPLACE TRIGGER experiments_frozen BEFORE UPDATE OF id, name, created_at, dataset_name, dataset_hash, config, config_hash
    ON experiments FOR EACH ROW EXECUTE FUNCTION llmobs_refuse('experiment definition is immutable');
CREATE OR REPLACE TRIGGER experiments_no_delete BEFORE DELETE ON experiments FOR EACH ROW EXECUTE FUNCTION llmobs_refuse('experiments are immutable');
"""


def canonical_hash(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


EXPORT_FORMAT = "llmobs-experiment/1"
# Append-only tables and their keys. On import a row the target lacks is added; a row it already has must be
# identical (same key, different content = two histories of one experiment, which is refused, never merged).
_IMMUTABLE_TABLES = {
    "experiment_examples": ("experiment_id", "example_id"),
    "experiment_corpus": ("experiment_id",),
    "experiment_docs": ("experiment_id", "doc_id"),
    "results": ("experiment_id", "variant", "example_id"),
    "retrievals": ("experiment_id", "variant", "example_id", "rank"),
    "scores": ("experiment_id", "variant", "example_id", "metric"),
}
_FROZEN_EXPERIMENT_COLUMNS = ("name", "created_at", "dataset_name", "dataset_hash", "config", "config_hash")
_STATUSES = ("created", "running", "partial", "complete")


class ImportConflict(ValueError):
    """The target already holds a different version of this experiment's immutable data."""


class ExperimentStore:
    """Experiments on SQLite (a file path, or ":memory:") or Postgres (a postgres:// URL)."""

    def __init__(self, path: str = ":memory:") -> None:
        self.db = Database(path, SCHEMA, PG_SCHEMA)

    # -- experiments --------------------------------------------------------
    def create_experiment(self, name: str, dataset: Dataset, config: Dict[str, Any],
                          corpus: Optional[Corpus] = None) -> str:
        exp_id = "exp_" + time.strftime("%Y%m%d_%H%M%S") + "_" + os.urandom(3).hex()
        with self.db.transaction() as tx:
            tx.execute("INSERT INTO experiments VALUES (?,?,?,?,?,?,?,?)",
                       (exp_id, name, time.time(), "created", dataset.name, dataset.content_hash,
                        json.dumps(config), canonical_hash(config)))
            tx.executemany("INSERT INTO experiment_examples VALUES (?,?,?,?,?,?,?,?)",
                           [(exp_id, i, e.id, e.input, e.reference, json.dumps(list(e.reference_docs)),
                             json.dumps(list(e.relevant_doc_ids)), json.dumps(e.metadata))
                            for i, e in enumerate(dataset.examples)])
            if corpus is not None:
                tx.execute("INSERT INTO experiment_corpus VALUES (?,?,?)", (exp_id, corpus.name, corpus.content_hash))
                tx.executemany("INSERT INTO experiment_docs VALUES (?,?,?,?,?,?)",
                               [(exp_id, i, d.id, d.title, d.text, json.dumps(d.metadata))
                                for i, d in enumerate(corpus.docs)])
        return exp_id

    def get_experiment(self, exp_id: str) -> Dict[str, Any]:
        d = self.db.one("SELECT * FROM experiments WHERE id = ?", (exp_id,))
        if d is None:
            raise KeyError(f"no such experiment: {exp_id}")
        d["config"] = json.loads(d["config"])
        return d

    def list_experiments(self, limit: int = 50) -> List[Dict[str, Any]]:
        return self.db.query("SELECT id, name, created_at, status, dataset_name FROM experiments "
                             "ORDER BY created_at DESC LIMIT ?", (limit,))

    def set_status(self, exp_id: str, status: str) -> None:
        self.db.execute("UPDATE experiments SET status = ? WHERE id = ?", (status, exp_id))

    def get_examples(self, exp_id: str) -> List[Example]:
        rows = self.db.query("SELECT * FROM experiment_examples WHERE experiment_id = ? ORDER BY position", (exp_id,))
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
        with self.db.transaction() as tx:
            # ON CONFLICT DO NOTHING rather than catching the error: Postgres aborts a transaction on any error.
            if not tx.execute(f"INSERT INTO results ({','.join(cols)}) VALUES ({','.join('?' * len(cols))}) "
                              "ON CONFLICT DO NOTHING", [r.get(c) for c in cols]):
                return False
            tx.executemany("INSERT INTO retrievals VALUES (?,?,?,?,?,?)",
                           [(r["experiment_id"], r["variant"], r["example_id"], i, doc_id, score)
                            for i, (doc_id, score) in enumerate(retrieved, 1)])
        return True

    def done_keys(self, exp_id: str) -> Set[Tuple[str, str]]:
        rows = self.db.query("SELECT variant, example_id FROM results WHERE experiment_id = ?", (exp_id,))
        return {(r["variant"], r["example_id"]) for r in rows}

    def get_results(self, exp_id: str, variant: Optional[str] = None) -> List[Dict[str, Any]]:
        sql, params = "SELECT * FROM results WHERE experiment_id = ?", [exp_id]
        if variant:
            sql, params = sql + " AND variant = ?", params + [variant]
        return self.db.query(sql + " ORDER BY created_at", params)

    # -- scores -------------------------------------------------------------
    def add_scores(self, rows: List[Dict[str, Any]]) -> int:
        """Append scores, skipping any that already exist. Returns how many were inserted."""
        inserted = 0
        with self.db.transaction() as tx:
            for r in rows:
                inserted += tx.execute(
                    "INSERT INTO scores VALUES (?,?,?,?,?,?,?) ON CONFLICT DO NOTHING",
                    (r["experiment_id"], r["variant"], r["example_id"], r["metric"], r.get("value"),
                     json.dumps(r["detail"]) if r.get("detail") is not None else None,
                     r.get("created_at", time.time())))
        return inserted

    def scored_keys(self, exp_id: str) -> Set[Tuple[str, str, str]]:
        rows = self.db.query("SELECT variant, example_id, metric FROM scores WHERE experiment_id = ?", (exp_id,))
        return {(r["variant"], r["example_id"], r["metric"]) for r in rows}

    def get_scores(self, exp_id: str) -> List[Dict[str, Any]]:
        out = self.db.query("SELECT * FROM scores WHERE experiment_id = ?", (exp_id,))
        for d in out:
            d["detail"] = json.loads(d["detail"]) if d["detail"] else None
        return out

    # -- corpus and retrievals ---------------------------------------------
    def get_docs(self, exp_id: str) -> List[Doc]:
        rows = self.db.query("SELECT * FROM experiment_docs WHERE experiment_id = ? ORDER BY position", (exp_id,))
        return [Doc(id=r["doc_id"], text=r["text"], title=r["title"],
                    metadata=json.loads(r["metadata"] or "{}")) for r in rows]

    def get_corpus_info(self, exp_id: str) -> Optional[Dict[str, Any]]:
        return self.db.one("SELECT name, content_hash FROM experiment_corpus WHERE experiment_id = ?", (exp_id,))

    def get_retrievals(self, exp_id: str) -> Dict[Tuple[str, str], List[Tuple[str, float]]]:
        """(variant, example_id) -> [(doc_id, score), ...] in rank order."""
        rows = self.db.query("SELECT variant, example_id, doc_id, score FROM retrievals WHERE experiment_id = ? "
                             "ORDER BY variant, example_id, rank", (exp_id,))
        out: Dict[Tuple[str, str], List[Tuple[str, float]]] = {}
        for r in rows:
            out.setdefault((r["variant"], r["example_id"]), []).append((r["doc_id"], r["score"]))
        return out

    # -- human labels ------------------------------------------------------
    _LABEL_COLS = ("experiment_id", "variant", "example_id", "labeler", "label", "source", "note", "created_at")
    _LABEL_UPSERT = (f"INSERT INTO labels ({', '.join(_LABEL_COLS)}) VALUES (?,?,?,?,?,?,?,?) "
                     "ON CONFLICT (experiment_id, variant, example_id, labeler) DO UPDATE SET "
                     "label = excluded.label, source = excluded.source, note = excluded.note, "
                     "created_at = excluded.created_at")

    def add_label(self, exp_id: str, variant: str, example_id: str, label: float, labeler: str = "human",
                  source: str = "random", note: Optional[str] = None) -> None:
        """Record (or correct) a human label. Unlike results and scores, labels may be replaced."""
        if label not in (0.0, 0.5, 1.0):
            raise ValueError("label must be 0, 0.5 or 1")
        if source not in ("random", "disagreement"):
            raise ValueError("source must be 'random' or 'disagreement'")
        self.db.execute(self._LABEL_UPSERT, (exp_id, variant, example_id, labeler, float(label), source, note, time.time()))

    def get_labels(self, exp_id: str, labeler: Optional[str] = None) -> List[Dict[str, Any]]:
        sql, params = "SELECT * FROM labels WHERE experiment_id = ?", [exp_id]
        if labeler:
            sql, params = sql + " AND labeler = ?", params + [labeler]
        return self.db.query(sql + " ORDER BY created_at", params)

    # -- hidden metrics: a view preference, so a retired judge can disappear without deleting its scores -----
    def hide_metric(self, exp_id: str, metric: str) -> None:
        self.db.execute("INSERT INTO hidden_metrics VALUES (?, ?) ON CONFLICT DO NOTHING", (exp_id, metric))

    def unhide_metric(self, exp_id: str, metric: str) -> None:
        self.db.execute("DELETE FROM hidden_metrics WHERE experiment_id = ? AND metric = ?", (exp_id, metric))

    def hidden_metrics(self, exp_id: str) -> Set[str]:
        return {r["metric"] for r in self.db.query("SELECT metric FROM hidden_metrics WHERE experiment_id = ?", (exp_id,))}

    # -- moving an experiment between databases (e.g. a laptop's and the server's) -----------------------
    def export_experiment(self, exp_id: str) -> Dict[str, Any]:
        """Every stored row of one experiment, as plain JSON-able data (see import_experiment)."""
        with self.db.transaction() as tx:                     # one consistent snapshot
            row = tx.one("SELECT * FROM experiments WHERE id = ?", (exp_id,))
            if row is None:
                raise KeyError(f"no such experiment: {exp_id}")
            tables = {t: tx.query(f"SELECT * FROM {t} WHERE experiment_id = ?", (exp_id,))
                      for t in (*_IMMUTABLE_TABLES, "labels", "hidden_metrics")}
        return {"format": EXPORT_FORMAT, "experiment": row, "tables": tables}

    def import_experiment(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Add an exported experiment, or bring an existing copy up to date, in one transaction.

        Immutable rows the target lacks are added and identical ones skipped; a differing one raises
        ImportConflict and nothing is written. Labels are replaced by key, the hidden-metric set and the
        status are taken from the payload. Re-importing the same export is a no-op.
        """
        if not isinstance(payload, dict) or payload.get("format") != EXPORT_FORMAT:
            raise ValueError(f"not an experiment export (expected format {EXPORT_FORMAT!r})")
        exp, tables = payload.get("experiment"), payload.get("tables")
        if not isinstance(exp, dict) or not isinstance(tables, dict) or not isinstance(exp.get("id"), str):
            raise ValueError("an export needs an 'experiment' object with an 'id' and a 'tables' object")
        exp_id = exp["id"]
        if exp.get("status") not in _STATUSES:
            raise ValueError(f"experiment status must be one of {_STATUSES}")
        unknown = set(tables) - set(_IMMUTABLE_TABLES) - {"labels", "hidden_metrics"}
        if unknown:
            raise ValueError(f"unknown tables in export: {sorted(unknown)}")
        columns = {t: self.db.columns(t) for t in ("experiments", *_IMMUTABLE_TABLES, "labels", "hidden_metrics")}
        if set(exp) != set(columns["experiments"]):
            raise ValueError(f"experiment must have exactly the columns {columns['experiments']}")

        def checked(table: str, row: Any) -> Dict[str, Any]:
            if not isinstance(row, dict) or set(row) != set(columns[table]):
                raise ValueError(f"{table}: rows must have exactly the columns {columns[table]}")
            if row["experiment_id"] != exp_id:
                raise ValueError(f"{table}: row belongs to {row['experiment_id']!r}, not {exp_id!r}")
            return row

        def insert_sql(table: str) -> str:
            cols = columns[table]
            return f"INSERT INTO {table} ({','.join(cols)}) VALUES ({','.join('?' * len(cols))})"

        def where(key: Tuple[str, ...]) -> str:
            return " AND ".join(f"{k} = ?" for k in key)

        counts: Dict[str, Any] = {"experiment_id": exp_id, "added": {}, "unchanged": 0}
        previous_status = None
        with self.db.transaction() as tx:
            current = tx.one("SELECT * FROM experiments WHERE id = ?", (exp_id,))
            if current is None:
                tx.execute(insert_sql("experiments"), [exp[c] for c in columns["experiments"]])
                counts["created"] = True
            else:
                changed = [c for c in _FROZEN_EXPERIMENT_COLUMNS if current[c] != exp[c]]
                if changed:
                    raise ImportConflict(f"{exp_id} already exists here with a different {', '.join(changed)}")
                counts["created"] = False
                previous_status = current["status"]
                if current["status"] != exp["status"]:
                    tx.execute("UPDATE experiments SET status = ? WHERE id = ?", (exp["status"], exp_id))

            for table, key in _IMMUTABLE_TABLES.items():
                cols, added = columns[table], 0
                for row in tables.get(table) or []:
                    row = checked(table, row)
                    have = tx.one(f"SELECT * FROM {table} WHERE {where(key)}", [row[k] for k in key])
                    if have is None:
                        tx.execute(insert_sql(table), [row[c] for c in cols])
                        added += 1
                    elif any(have[c] != row[c] for c in cols):
                        raise ImportConflict(f"{table} {tuple(row[k] for k in key)} differs from the copy here: "
                                             "these are two different histories of the same experiment")
                    else:
                        counts["unchanged"] += 1
                counts["added"][table] = added

            key = ("experiment_id", "variant", "example_id", "labeler")
            changed_labels = 0
            for row in tables.get("labels") or []:
                row = checked("labels", row)
                have = tx.one(f"SELECT * FROM labels WHERE {where(key)}", [row[k] for k in key])
                if have is None or any(have[c] != row[c] for c in columns["labels"]):
                    tx.execute(self._LABEL_UPSERT, [row[c] for c in self._LABEL_COLS])
                    changed_labels += 1
            counts["added"]["labels"] = changed_labels
            hidden = {checked("hidden_metrics", r)["metric"] for r in tables.get("hidden_metrics") or []}
            had = {r["metric"] for r in tx.query("SELECT metric FROM hidden_metrics WHERE experiment_id = ?", (exp_id,))}
            if hidden != had:
                tx.execute("DELETE FROM hidden_metrics WHERE experiment_id = ?", (exp_id,))
                tx.executemany("INSERT INTO hidden_metrics VALUES (?, ?)", [(exp_id, m) for m in sorted(hidden)])
            counts["hidden_metrics_changed"] = hidden != had
        counts["status"] = exp["status"]
        counts["status_changed"] = previous_status is not None and previous_status != exp["status"]
        return counts
