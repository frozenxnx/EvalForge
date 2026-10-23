import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import types
import unittest

from llmobs_eval import (ExperimentStore, create_experiment, evaluate_experiment, examples_payload,
                         load_jsonl)
from llmobs_eval.providers import ProviderError
from tests.test_eval import write_jsonl
from tests.test_step3 import NO_SLEEP, ROOT, ScriptedJudge, cfg2, seed

JUDGE = {"provider": "m", "model": "judge-x"}


def node_render(fn: str, payload, *more) -> str:
    with open(os.path.join(ROOT, "llmobs_server", "static", "viewer.html"), encoding="utf-8") as fh:
        script = re.search(r"<script>([\s\S]*)</script>", fh.read()).group(1)
    args = ", ".join(json.dumps(a) for a in (payload,) + more)
    harness = f"""
const stub = () => ({{ innerHTML: "", classList: {{ toggle() {{}} }}, onclick: null }});
const mod = {{ exports: {{}} }};
new Function("module", "document", "fetch", "setInterval", {json.dumps(script)})(mod,
  {{ querySelector: stub, querySelectorAll: () => [] }}, async () => ({{ ok: true, json: async () => ({{ traces: [], experiments: [] }}) }}), () => 0);
process.stdout.write(mod.exports.{fn}({args}));
"""
    with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False, encoding="utf-8") as fh:
        fh.write(harness)
    out = subprocess.run(["node", fh.name], capture_output=True, text=True, timeout=30)
    assert out.returncode == 0, out.stderr
    return out.stdout


class ExamplesPayloadTests(unittest.TestCase):
    def setUp(self):
        self.store = ExperimentStore()
        self.exp = seed(self.store)
        evaluate_experiment(self.store, self.exp, judge=JUDGE, providers={"m": ScriptedJudge()}, sleep=NO_SLEEP)

    def test_worst_first_with_full_evidence(self):
        d = examples_payload(self.store, self.exp, limit=100)
        self.assertEqual((d["total"], d["primary_metric"], d["variants"]), (20, "judge:judge-x", ["base", "cand"]))
        ids = [e["id"] for e in d["examples"]]
        self.assertEqual(ids[:8], [f"q{i}" for i in range(8)])          # the 8 regressions, ties by dataset order
        self.assertTrue(all(e["primary_min"] == 1.0 for e in d["examples"][8:]))
        first = d["examples"][0]
        self.assertEqual((first["input"], first["reference"]), ("question 0", "answer 0"))
        cand = first["variants"]["cand"]
        self.assertEqual(cand["output"], "wrong 0")
        self.assertEqual(first["variants"]["base"]["output"], "answer 0")
        j = cand["scores"]["judge:judge-x"]
        self.assertEqual((j["value"], j["verdict"], j["reason"]), (0.0, "incorrect", "judged wrong 0"))
        self.assertEqual(cand["scores"]["exact_match"]["value"], 0.0)

    def test_paging_and_dataset_order(self):
        page = examples_payload(self.store, self.exp, limit=5, offset=5)
        self.assertEqual([e["id"] for e in page["examples"]], ["q5", "q6", "q7", "q8", "q9"])
        d = examples_payload(self.store, self.exp, sort="dataset", limit=3)
        self.assertEqual([e["id"] for e in d["examples"]], ["q0", "q1", "q2"])

    def test_validation(self):
        with self.assertRaisesRegex(ValueError, "sort must"):
            examples_payload(self.store, self.exp, sort="random")
        with self.assertRaisesRegex(ValueError, "unknown metric"):
            examples_payload(self.store, self.exp, metric="nope")
        with self.assertRaises(KeyError):
            examples_payload(self.store, "exp_missing")

    def test_failed_and_missing_generations_come_first(self):
        store = ExperimentStore()
        rows = [{"id": f"q{i}", "input": f"q{i}", "reference": f"a{i}"} for i in range(3)]
        exp = create_experiment(store, load_jsonl(write_jsonl(rows)), cfg2())
        base = dict(experiment_id=exp, attempts=1, wall_ms=1, created_at=1)
        for i in range(3):
            store.add_result({**base, "variant": "base", "example_id": f"q{i}", "status": "ok", "output": f"a{i}"})
        store.add_result({**base, "variant": "cand", "example_id": "q0", "status": "ok", "output": "a0"})
        store.add_result({**base, "variant": "cand", "example_id": "q2", "status": "error", "error": "HTTP 500: down"})
        # q1 has no cand result at all
        evaluate_experiment(store, exp)
        d = examples_payload(store, exp)
        self.assertEqual([e["id"] for e in d["examples"]], ["q1", "q2", "q0"])
        self.assertEqual(d["examples"][0]["variants"]["cand"]["status"], "missing")
        self.assertEqual(d["examples"][1]["variants"]["cand"]["error"], "HTTP 500: down")

    def test_works_before_any_evaluation(self):
        store = ExperimentStore()
        exp = seed(store, n=4)
        d = examples_payload(store, exp)
        self.assertEqual((d["metrics"], d["primary_metric"], len(d["examples"])), ([], None, 4))


class EvaluateErrorReportingTests(unittest.TestCase):
    def test_failure_reasons_are_surfaced_not_just_counted(self):
        store = ExperimentStore()
        exp = seed(store, n=6)

        class NoSuchModel(ScriptedJudge):
            def complete(self, model, messages, params):
                raise ProviderError("HTTP 404: model 'oops' not found", 404)

        r = evaluate_experiment(store, exp, lexical=False, judge={"provider": "m", "model": "oops"},
                                providers={"m": NoSuchModel()}, sleep=NO_SLEEP)
        self.assertEqual((r["scored"], r["judge_failed"]), (0, 12))
        self.assertEqual(r["error_samples"], ["HTTP 404: model 'oops' not found"])   # de-duplicated
        self.assertEqual(store.get_scores(exp), [])                                 # nothing recorded


class GeneralDatasetTests(unittest.TestCase):
    def test_example_dataset_is_valid(self):
        ds = load_jsonl(os.path.join(ROOT, "examples", "dataset_general.jsonl"))
        self.assertEqual(len(ds.examples), 20)
        self.assertTrue(all(e.reference for e in ds.examples))


class ExampleConfigTests(unittest.TestCase):
    def test_every_example_config_is_valid(self):
        from llmobs_eval import normalize_config
        names = [n for n in os.listdir(os.path.join(ROOT, "examples")) if n.startswith("experiment_")]
        self.assertGreaterEqual(len(names), 3)
        for n in names:
            with open(os.path.join(ROOT, "examples", n), encoding="utf-8") as fh:
                cfg = normalize_config(json.load(fh))
            self.assertTrue(cfg["variants"], n)


class ApiTests(unittest.TestCase):
    def test_examples_route(self):
        try:
            import fastapi  # noqa: F401
        except ImportError:
            m = types.ModuleType("fastapi")

            class HTTPException(Exception):
                def __init__(self, status_code, detail=None):
                    self.status_code, self.detail = status_code, detail
            m.HTTPException, m.Query = HTTPException, (lambda default=None, **kw: default)
            sys.modules["fastapi"] = m
        from llmobs_server import experiments_api

        class App:
            def __init__(self):
                self.routes = {}

            def get(self, path, **kw):
                return lambda fn: self.routes.setdefault(path, fn)

        store, app = ExperimentStore(), App()
        exp = seed(store)
        evaluate_experiment(store, exp)
        experiments_api.register(app, store)
        route = app.routes["/v1/experiments/{experiment_id}/examples"]
        out = route(experiment_id=exp, metric=None, sort="worst", limit=3, offset=0)
        self.assertEqual((out["total"], len(out["examples"])), (20, 3))
        for kw, code in (({"experiment_id": "exp_x"}, 404), ({"experiment_id": exp, "sort": "bad"}, 400)):
            with self.assertRaises(Exception) as cm:
                route(**{"metric": None, "sort": "worst", "limit": 3, "offset": 0, **kw})
            self.assertEqual(cm.exception.status_code, code)


@unittest.skipUnless(shutil.which("node"), "node not installed")
class ViewerExamplesTests(unittest.TestCase):
    def test_render_examples(self):
        store = ExperimentStore()
        exp = seed(store)
        evaluate_experiment(store, exp, judge=JUDGE, providers={"m": ScriptedJudge()}, sleep=NO_SLEEP)
        h = node_render("renderExamples", examples_payload(store, exp, limit=10))
        self.assertEqual(h.count("<details>"), 10)
        for needle in ("worst first by judge:judge-x", "showing 10 of 20", "chip s-lo", "chip s-hi",
                       "judged wrong 0", "incorrect", 'class="tl"', 'id="more-ex"', "wrong 0", "answer 0",
                       'style="--n:2"', "Reference"):
            self.assertIn(needle, h)
        for bad in ("undefined", "NaN"):
            self.assertNotIn(bad, h)

    def test_render_failed_generation_and_empty(self):
        store = ExperimentStore()
        rows = [{"id": "q0", "input": "q0", "reference": "a0"}]
        exp = create_experiment(store, load_jsonl(write_jsonl(rows)), cfg2())
        store.add_result(dict(experiment_id=exp, variant="base", example_id="q0", status="ok", output="a0",
                              attempts=1, wall_ms=1, created_at=1))
        store.add_result(dict(experiment_id=exp, variant="cand", example_id="q0", status="error",
                              error="HTTP 503: busy", attempts=3, wall_ms=1, created_at=1))
        evaluate_experiment(store, exp)
        h = node_render("renderExamples", examples_payload(store, exp))
        self.assertIn("HTTP 503: busy", h)
        self.assertIn(">error<", h)
        self.assertNotIn("undefined", h)
        self.assertEqual(node_render("renderExamples", {"examples": []}), "")


if __name__ == "__main__":
    unittest.main()
