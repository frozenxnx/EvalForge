"""FastAPI ingestion + query API. Run: uvicorn llmobs_server.app:app --reload"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import Depends, FastAPI, Header, HTTPException, Query
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from llmobs_eval.store import ExperimentStore

from . import experiments_api
from .store import SpanStore

STATIC = Path(__file__).parent / "static"


class SpanIn(BaseModel):
    trace_id: str
    span_id: str
    parent_id: Optional[str] = None
    name: str
    kind: str = "chain"
    service: Optional[str] = None
    start_ns: int
    end_ns: Optional[int] = None
    status: str = "ok"
    error: Optional[str] = None
    input: Any = None
    output: Any = None
    attributes: Dict[str, Any] = Field(default_factory=dict)


class IngestBatch(BaseModel):
    spans: List[SpanIn] = Field(max_length=1000)


def create_app(store: Optional[SpanStore] = None, api_key: Optional[str] = None,
               experiments: Optional[ExperimentStore] = None) -> FastAPI:
    db_path = os.environ.get("LLMOBS_DB", "llmobs.db")
    store = store or SpanStore(db_path)
    experiments = experiments or ExperimentStore(db_path)
    api_key = api_key or os.environ.get("LLMOBS_API_KEY")
    app = FastAPI(title="llmobs", version="0.1.0")

    def auth(authorization: Optional[str] = Header(default=None)) -> None:
        if api_key and authorization != f"Bearer {api_key}":
            raise HTTPException(status_code=401, detail="invalid API key")

    @app.post("/v1/spans", status_code=202, dependencies=[Depends(auth)])
    def ingest(batch: IngestBatch):
        n = store.ingest([s.model_dump() for s in batch.spans])
        return {"accepted": n}

    @app.get("/v1/traces")
    def traces(limit: int = Query(50, ge=1, le=200), offset: int = Query(0, ge=0),
               service: Optional[str] = None):
        return {"traces": store.list_traces(limit, offset, service)}

    @app.get("/v1/traces/{trace_id}")
    def trace_detail(trace_id: str):
        spans = store.get_trace(trace_id)
        if not spans:
            raise HTTPException(status_code=404, detail="trace not found")
        return {"trace_id": trace_id, "spans": spans}

    experiments_api.register(app, experiments)

    @app.get("/healthz")
    def healthz():
        return {"ok": True}

    @app.get("/", response_class=HTMLResponse)
    def viewer():
        return (STATIC / "viewer.html").read_text(encoding="utf-8")

    return app


app = create_app()
