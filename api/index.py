"""Vercel entry point: the FastAPI app as a Python serverless function (vercel.json routes every path here).

Vercel has no persistent disk, so data lives in Postgres ($DATABASE_URL, set by the Neon integration), and
the deployment is public, so an API key is mandatory. If either is missing, every route answers 503 with
what to fix, rather than running an open server or failing on the read-only filesystem.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llmobs_eval.db import database_target, is_postgres  # noqa: E402


def config_problems() -> list:
    problems = []
    if not os.environ.get("LLMOBS_API_KEY"):
        problems.append("LLMOBS_API_KEY is not set: a public deployment must require a key")
    target = database_target(default="")
    if not target:
        problems.append("no database: connect Neon Postgres (Vercel > Storage), which sets DATABASE_URL")
    elif not is_postgres(target):
        problems.append("the database must be a postgres:// URL on Vercel (got a file path; files do not persist)")
    return problems


PROBLEMS = config_problems()

if PROBLEMS:
    from fastapi import FastAPI
    from fastapi.responses import JSONResponse

    app = FastAPI(title="llmobs (not configured)")

    @app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS"])
    def not_configured(path: str):
        return JSONResponse({"detail": "server is not configured", "problems": PROBLEMS}, status_code=503)
else:
    from llmobs_server.app import app  # noqa: E402,F401
