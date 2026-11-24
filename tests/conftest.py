"""Run the whole suite on Postgres:  LLMOBS_TEST_PG=postgresql://user:pw@host:port/db  python -m pytest

Every store a test opens on ":memory:" gets a fresh schema of that database; a store opened on a file path
gets a schema named after the path, so two stores on the same file (server + CLI) share it, as they would.
"""
import hashlib
import os
import uuid

PG = os.environ.get("LLMOBS_TEST_PG")

if PG:
    from llmobs_eval import db

    def _resolve(target):
        if db.is_postgres(target):
            return target, None
        key = uuid.uuid4().hex if target == ":memory:" else os.path.abspath(target)
        return PG, "t_" + hashlib.sha1(key.encode()).hexdigest()[:16]

    db.resolve_target = _resolve
