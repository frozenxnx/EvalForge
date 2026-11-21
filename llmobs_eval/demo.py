"""A self-contained demo database: two experiments, traces, a judge and human labels. No model is called.

    python -m llmobs_eval.demo --db demo.db && python -m llmobs_server.dev_server --db demo.db

Everything here is SYNTHETIC: the "models" answer like a system with a known accuracy per question category
(deterministically, from a seed), the "judge" is a rule with a few planted mistakes, and latencies are scaled down.
The pipeline around them (datasets, BM25 retrieval, tracing, scoring, comparison, calibration, failure
patterns) is the real thing, so the UI can be explored and demonstrated without Ollama or an API key.
"""
from __future__ import annotations

import argparse
import os
import random
import re
import time
import zlib
from pathlib import Path
from typing import Any, Dict, List, Mapping

import llmobs
from llmobs.exporters import Exporter

from .datasets import Example, load_corpus_jsonl, load_jsonl
from .evaluate import evaluate_experiment
from .metrics import contains, token_f1
from .patterns import is_refusal
from .providers import Completion, ConfigError, Provider
from .runner import create_experiment, run_experiment
from .store import ExperimentStore

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"


class StoreExporter(Exporter):
    """Writes spans straight into a span store (what the server's ingestion endpoint does)."""

    def __init__(self, span_store: Any) -> None:
        self.span_store = span_store

    def export(self, span):
        self.span_store.ingest([span])


def _rng(*parts: Any) -> random.Random:
    return random.Random(zlib.crc32("|".join(str(p) for p in parts).encode()))


def _question_index(examples: List[Example]):
    return sorted(((e.input, e) for e in examples), key=lambda t: -len(t[0]))


def _find(index, content: str) -> Example:
    for question, ex in index:
        if question in content:
            return ex
    raise ValueError("demo model could not find the question in the prompt")


class DemoModel(Provider):
    """Answers like a model with a known accuracy per category."""

    ACCURACY: Mapping[str, Mapping[str, float]] = {
        "demo-small": {"arithmetic": .45, "calendar": .40, "letters": .50, "logic": .55, "units": .70,
                       "sequence": .50, "facts": .85, "false_premise": .25},
        "demo-small-concise": {"arithmetic": .45, "calendar": .20, "letters": .30, "logic": .50, "units": .70,
                               "sequence": .50, "facts": .85, "false_premise": .20},
        "demo-small-cot": {"arithmetic": .75, "calendar": .55, "letters": .60, "logic": .80, "units": .85,
                           "sequence": .70, "facts": .85, "false_premise": .35},
    }
    LATENCY_S = {"demo-small": 0.9, "demo-small-concise": 0.6, "demo-small-cot": 2.2}

    def __init__(self, examples: List[Example], seed: int = 7, latency_scale: float = 0.02) -> None:
        self.examples, self.seed, self.scale = examples, seed, latency_scale
        self.index = _question_index(examples)

    def complete(self, model, messages, params):
        content = messages[-1]["content"]
        ex = _find(self.index, content)
        rng = _rng(self.seed, model, ex.id)
        time.sleep(rng.uniform(0.4, 1.4) * self.LATENCY_S[model] * self.scale)
        category = ex.metadata.get("category", "facts")
        ref = ex.reference or ""
        if rng.random() < self.ACCURACY[model].get(category, .5):
            text = ref if category == "false_premise" else f"The answer is {ref}."
        elif category == "false_premise":
            text = "Sure! They said: \"We must keep learning and stay curious.\""
        elif ref.strip().isdigit() and rng.random() < .8:
            text = f"The answer is {int(ref) + rng.choice([-2, -1, 1, 2, 10])}."
        elif rng.random() < .5:
            text = "I don't know."
        else:
            others = [e.reference for e in self.examples if e.metadata.get("category") == category and e.id != ex.id]
            text = f"I believe it is {rng.choice(others) if others else 'unknown'}."
        extra = 40 if model.endswith("cot") else 0
        return Completion(text, len(content.split()) + 60, len(text.split()) + extra, "stop")


class DemoRagModel(Provider):
    """A RAG generator: answers correctly only when the needed documents were retrieved."""

    def __init__(self, examples: List[Example], seed: int = 7, latency_scale: float = 0.02) -> None:
        self.seed, self.scale = seed, latency_scale
        self.index = _question_index(examples)

    def complete(self, model, messages, params):
        content = messages[-1]["content"]
        ex = _find(self.index, content)
        rng = _rng(self.seed, model, ex.id)
        time.sleep(rng.uniform(0.4, 1.4) * (0.5 if model.endswith("closed") else 1.2) * self.scale)
        ref = ex.reference or ""
        context = content.split("Question:")[0] if "Context:" in content else ""
        got = set(re.findall(r"\[(d\d+)\]", context))
        if model.endswith("closed"):
            text = ref if (ex.relevant_doc_ids and rng.random() < .06) else "I don't know."
        elif not ex.relevant_doc_ids:                                  # unanswerable from the documents
            text = "I don't know." if rng.random() < .8 else "You can reach Premium support at +1 555 0100."
        elif set(ex.relevant_doc_ids) <= got and rng.random() < .93:
            text = ref if ex.metadata.get("check") == "judge_only" else f"According to the documents: {ref}."
        else:
            text = "I don't know."
        return Completion(text, len(content.split()) + 80, len(text.split()), "stop")


def _truth(reference: str, answer: str) -> bool:
    """What a careful human would say."""
    unavailable = "do not say" in reference.lower() or "no nobel" in reference.lower() or "has no" in reference.lower()
    if unavailable:
        return is_refusal(answer) or token_f1(answer, reference) > .5
    if is_refusal(answer):
        return False
    return contains(answer, reference) == 1.0 or token_f1(answer, reference) > .6


class DemoJudge(Provider):
    """A rule-based 'judge' with planted mistakes: it sometimes credits a refusal and sometimes rejects a good answer."""

    def complete(self, model, messages, params):
        prompt = messages[0]["content"]
        ref = re.search(r"<reference_answer>\n(.*?)\n</reference_answer>", prompt, re.S).group(1)
        cand = re.search(r"<candidate_answer>\n(.*?)\n</candidate_answer>", prompt, re.S).group(1)
        rng = _rng("judge", ref, cand)
        ok = _truth(ref, cand)
        reason = "the answer states the reference value" if ok else "the answer does not match the reference"
        if not ok and is_refusal(cand) and rng.random() < .22:
            ok, reason = True, "close enough to the reference"                # a planted lenient mistake
        elif ok and rng.random() < .04:
            ok, reason = False, "the answer is not specific enough"           # a planted strict mistake
        verdict = "correct" if ok else "incorrect"
        return Completion(f'{{"verdict": "{verdict}", "reason": "{reason}"}}', 320, 24, "stop")


def build_demo(db_path: str, seed: int = 7, latency_scale: float = 0.02, span_store: Any = None,
               n_labels: int = 28) -> Dict[str, str]:
    for needed in ("dataset_hard.jsonl", "dataset_rag.jsonl", "corpus_lumen.jsonl"):
        if not (EXAMPLES / needed).exists():
            raise ConfigError(f"demo data not found: {EXAMPLES / needed} (run from a checkout of the repository)")
    store = ExperimentStore(db_path)
    if span_store is None:
        from llmobs_server.store import SpanStore
        span_store = SpanStore(db_path)
    llmobs.configure(exporter=StoreExporter(span_store), service="demo")
    judge_spec = {"provider": "demo", "model": "demo-judge"}
    try:
        # ---- A: a prompt A/B test on the hard question set -----------------------------------------
        hard = load_jsonl(str(EXAMPLES / "dataset_hard.jsonl"))
        price = {"input_per_mtok": 0.25, "output_per_mtok": 1.25}
        cfg_a = {
            "name": "prompt-ab", "providers": {"demo": {"type": "mock"}},
            "variants": [
                {"label": "baseline", "provider": "demo", "model": "demo-small", "system": "Answer concisely.",
                 "params": {"temperature": 0}},
                {"label": "pr-one-sentence", "provider": "demo", "model": "demo-small-concise",
                 "system": "Answer in exactly one short sentence.", "params": {"temperature": 0}},
                {"label": "pr-step-by-step", "provider": "demo", "model": "demo-small-cot",
                 "system": "Think step by step, then give the final answer.", "params": {"temperature": 0}}],
            "concurrency": 2, "max_retries": 1,
            "pricing": {"demo-small": price, "demo-small-concise": price, "demo-small-cot": price}}
        exp_a = create_experiment(store, hard, cfg_a, name="prompt-ab (synthetic demo)")
        run_experiment(store, exp_a, providers={"demo": DemoModel(hard.examples, seed, latency_scale)})
        evaluate_experiment(store, exp_a, judge=judge_spec, providers={"demo": DemoJudge()})

        # ---- B: a RAG experiment on the company-docs questions (real BM25 retrieval) -------------------
        rag = load_jsonl(str(EXAMPLES / "dataset_rag.jsonl"))
        corpus = load_corpus_jsonl(str(EXAMPLES / "corpus_lumen.jsonl"))
        system = "Answer using only the provided context. If the context does not contain the answer, reply exactly: I don't know."
        template = "Context:\n{context}\n\nQuestion: {input}\nAnswer briefly."
        cfg_b = {
            "name": "rag-retrieval", "providers": {"demo": {"type": "mock"}},
            "variants": [
                {"label": "closed-book", "provider": "demo", "model": "demo-rag-closed", "params": {"temperature": 0}},
                {"label": "bm25-k3", "provider": "demo", "model": "demo-rag", "system": system,
                 "prompt_template": template, "retriever": {"type": "bm25", "k": 3}, "params": {"temperature": 0}},
                {"label": "oracle", "provider": "demo", "model": "demo-rag-oracle", "system": system,
                 "prompt_template": template, "retriever": {"type": "oracle", "k": 3}, "params": {"temperature": 0}}],
            "concurrency": 2, "max_retries": 1}
        exp_b = create_experiment(store, rag, cfg_b, name="rag-retrieval (synthetic demo)", corpus=corpus)
        run_experiment(store, exp_b, providers={"demo": DemoRagModel(rag.examples, seed, latency_scale)})
        evaluate_experiment(store, exp_b, judge=judge_spec, providers={"demo": DemoJudge()})

        # ---- human labels: a random sample, labeled by a careful "human" ------------------------------------
        for exp_id, ds in ((exp_a, hard), (exp_b, rag)):
            examples = {e.id: e for e in ds.examples}
            pairs = sorted((r["variant"], r["example_id"], r["output"]) for r in store.get_results(exp_id)
                           if r["status"] == "ok")
            _rng("labels", seed, exp_id[-4:]).shuffle(pairs)
            for variant, ex_id, output in pairs[:n_labels]:
                store.add_label(exp_id, variant, ex_id, 1.0 if _truth(examples[ex_id].reference or "", output) else 0.0,
                                "demo-human", "random")
    finally:
        llmobs.configure(exporter=None)
    return {"prompt-ab": exp_a, "rag-retrieval": exp_b}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="llmobs_eval.demo", description="Build a synthetic demo database (no model needed).")
    p.add_argument("--db", default="demo.db")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--force", action="store_true", help="overwrite an existing database file")
    args = p.parse_args(argv)
    if os.path.exists(args.db):
        if not args.force:
            print(f"error: {args.db} already exists (use --force to overwrite)")
            return 2
        for suffix in ("", "-wal", "-shm"):
            if os.path.exists(args.db + suffix):
                os.remove(args.db + suffix)
    ids = build_demo(args.db, args.seed)
    print(f"demo database written to {args.db} (all data synthetic):")
    for name, exp_id in ids.items():
        print(f"  {name:<14} {exp_id}")
    print(f"\nopen it:  python -m llmobs_server.dev_server --db {args.db}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
