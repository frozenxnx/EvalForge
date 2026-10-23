"""Regressions for type-checker complaints (Pylance/pyright) about Optional spans and exporter stats."""
import ast
import importlib.util
import os
import unittest
from unittest import mock

import llmobs
from llmobs import InMemoryExporter, NoopExporter
from llmobs.exporters import Exporter, HTTPExporter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class RecorderHelperTests(unittest.TestCase):
    def tearDown(self):
        llmobs.configure(exporter=None)

    def test_helpers_annotate_the_current_span(self):
        mem = InMemoryExporter()
        llmobs.configure(exporter=mem, service="t")
        with llmobs.span("generate") as s:
            llmobs.record_llm("model-x", 12, 5, temperature=0)
        with llmobs.span("search"):
            llmobs.record_retrieval("q", [{"id": "d1", "text": "t", "score": 0.5}])
        gen, ret = mem.spans
        self.assertEqual(gen["kind"], "llm")
        self.assertEqual(gen["attributes"], {"gen_ai.request.model": "model-x", "gen_ai.usage.input_tokens": 12,
                                             "gen_ai.usage.output_tokens": 5, "temperature": 0})
        self.assertEqual(ret["kind"], "retrieval")
        self.assertEqual(ret["attributes"]["retrieval.documents"][0]["id"], "d1")
        self.assertIsNotNone(s)

    def test_helpers_do_nothing_outside_a_trace(self):
        self.assertIsNone(llmobs.current_span())
        self.assertIsNone(llmobs.record_llm("m", 1, 2))               # must not raise
        self.assertIsNone(llmobs.record_retrieval("q", []))

    def test_helpers_target_the_innermost_span(self):
        mem = InMemoryExporter()
        llmobs.configure(exporter=mem)
        with llmobs.span("outer"):
            with llmobs.span("inner"):
                llmobs.record_llm("m", 1, 1)
        by = {s["name"]: s for s in mem.spans}
        self.assertIn("gen_ai.request.model", by["inner"]["attributes"])
        self.assertNotIn("gen_ai.request.model", by["outer"]["attributes"])


class ExporterStatsTests(unittest.TestCase):
    def test_every_exporter_has_stats(self):
        self.assertEqual(Exporter().stats(), {})
        self.assertEqual(NoopExporter().stats(), {})
        self.assertEqual(InMemoryExporter().stats(), {})
        self.assertEqual(HTTPExporter("http://127.0.0.1:1/", autostart=False).stats(),
                         {"sent": 0, "failed": 0, "dropped": 0})

    def test_stats_is_reachable_through_the_tracer_like_the_demo_does(self):
        llmobs.configure(exporter=None)
        self.assertIsInstance(llmobs.get_tracer().exporter.stats(), dict)
        llmobs.get_tracer().exporter.flush()


class DemoTests(unittest.TestCase):
    def test_rag_demo_runs_and_emits_the_expected_spans(self):
        spec = importlib.util.spec_from_file_location("rag_demo", os.path.join(ROOT, "examples", "rag_demo.py"))
        demo = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(demo)
        mem = InMemoryExporter()
        llmobs.configure(exporter=mem, service="demo")
        try:
            with mock.patch.object(demo.time, "sleep", lambda s: None):
                demo.answer("How long do refunds take?")
        finally:
            llmobs.configure(exporter=None)
        by = {s["name"]: s for s in mem.spans}
        self.assertEqual(len(by["retrieve"]["attributes"]["retrieval.documents"]), 2)
        self.assertEqual(by["generate"]["attributes"]["gen_ai.request.model"], "fake-model-1")
        self.assertIn("gen_ai.usage.output_tokens", by["generate"]["attributes"])
        self.assertEqual(by["rag_pipeline"]["parent_id"], None)


class StaticGuardTests(unittest.TestCase):
    def test_nobody_dereferences_current_span_directly(self):
        """`current_span()` is Optional[Span]. `current_span().x` is what Pylance rejects and what crashes
        outside a trace; use llmobs.record_llm / record_retrieval, or check for None first."""
        offenders = []
        for folder, _, files in os.walk(ROOT):
            if any(part in folder for part in ("__pycache__", ".git", ".venv", "venv")):
                continue
            for name in files:
                if not name.endswith(".py") or name == "test_typing_safety.py":
                    continue
                path = os.path.join(folder, name)
                with open(path, encoding="utf-8") as fh:
                    tree = ast.parse(fh.read(), path)
                for node in ast.walk(tree):
                    if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Call):
                        f = node.value.func
                        called = f.id if isinstance(f, ast.Name) else getattr(f, "attr", None)
                        if called == "current_span":
                            offenders.append(f"{os.path.relpath(path, ROOT)}:{node.lineno}")
        self.assertEqual(offenders, [], "use llmobs.record_llm/record_retrieval or check for None")


if __name__ == "__main__":
    unittest.main()
