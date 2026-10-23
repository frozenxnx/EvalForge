"""Experiment routes, kept separate from the span-ingestion app. All logic lives in llmobs_eval."""
from __future__ import annotations

from typing import Optional

from fastapi import HTTPException, Query

from llmobs_eval.compare import compare_payload, examples_payload
from llmobs_eval.judgecheck import calibration_payload, judge_check_payload
from llmobs_eval.store import ExperimentStore


def register(app, experiments: ExperimentStore) -> None:
    @app.get("/v1/experiments")
    def list_experiments(limit: int = Query(50, ge=1, le=200)):
        return {"experiments": experiments.list_experiments(limit)}

    @app.get("/v1/experiments/{experiment_id}/compare")
    def compare(experiment_id: str, baseline: Optional[str] = None, metric: Optional[str] = None,
                top_k: int = Query(5, ge=1, le=50)):
        try:
            return compare_payload(experiments, experiment_id, baseline, metric, n_boot=1000, top_k=top_k)
        except KeyError:
            raise HTTPException(status_code=404, detail="experiment not found")
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    @app.get("/v1/experiments/{experiment_id}/examples")
    def examples(experiment_id: str, metric: Optional[str] = None, sort: str = "worst",
                 limit: int = Query(25, ge=1, le=200), offset: int = Query(0, ge=0)):
        try:
            return examples_payload(experiments, experiment_id, metric, sort, limit, offset)
        except KeyError:
            raise HTTPException(status_code=404, detail="experiment not found")
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    @app.get("/v1/experiments/{experiment_id}/judge-check")
    def judge_check(experiment_id: str, metric: Optional[str] = None, top_k: int = Query(15, ge=0, le=100)):
        try:
            return judge_check_payload(experiments, experiment_id, metric, top_k)
        except KeyError:
            raise HTTPException(status_code=404, detail="experiment not found")
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    @app.get("/v1/experiments/{experiment_id}/calibration")
    def calibration(experiment_id: str, metric: Optional[str] = None):
        try:
            return calibration_payload(experiments, experiment_id, metric, n_boot=500)
        except KeyError:
            raise HTTPException(status_code=404, detail="experiment not found")
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
