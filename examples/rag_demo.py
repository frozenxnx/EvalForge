"""Fake RAG app that emits realistic traces.

    python -m uvicorn llmobs_server.app:app      # terminal 1, then open http://127.0.0.1:8000
    python examples/rag_demo.py                  # terminal 2
"""
import os
import random
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import llmobs

DOCS = {
    "d1": "Refunds are processed within 5 business days.",
    "d2": "Premium plans include priority support.",
    "d3": "Passwords must be at least 12 characters.",
}


@llmobs.trace(kind="retrieval")
def retrieve(query: str, k: int = 2):
    time.sleep(random.uniform(0.02, 0.06))
    hits = random.sample(list(DOCS), k)
    llmobs.record_retrieval(
        query, [{"id": h, "text": DOCS[h], "score": round(random.random(), 3)} for h in hits])
    return [DOCS[h] for h in hits]


@llmobs.trace(kind="tool")
def check_policy(topic: str):
    time.sleep(0.01)
    if random.random() < 0.1:
        raise TimeoutError("policy service timed out")
    return {"topic": topic, "ok": True}


@llmobs.trace(kind="llm")
def generate(query: str, context: list):
    time.sleep(random.uniform(0.15, 0.4))  # pretend model latency
    llmobs.record_llm(
        "fake-model-1", input_tokens=80 + 20 * len(context), output_tokens=random.randint(30, 90))
    return f"Based on the docs: {context[0]}"


@llmobs.trace(name="rag_pipeline")
def answer(query: str):
    ctx = retrieve(query)
    try:
        check_policy("refunds")
    except TimeoutError:
        pass  # degrade gracefully; the failed span is still visible in the trace
    return generate(query, ctx)


if __name__ == "__main__":
    exporter_url = os.environ.get("LLMOBS_ENDPOINT", "http://127.0.0.1:8000/v1/spans")
    llmobs.configure(endpoint=exporter_url, service="rag-demo")
    for q in ["How long do refunds take?", "Do premium plans get support?", "Password rules?"] * 3:
        print(answer(q))
    llmobs.get_tracer().exporter.flush()
    print(llmobs.get_tracer().exporter.stats())
