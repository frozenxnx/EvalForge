"""The FastAPI app enforces the same access rules as the stdlib server (see llmobs_server/auth.py)."""
import unittest

try:
    from fastapi.testclient import TestClient
    HAVE_FASTAPI = True
except ImportError:                                  # fastapi/httpx are optional (the `server` extra)
    HAVE_FASTAPI = False

from llmobs_eval.store import ExperimentStore
from llmobs_server.auth import MAX_BODY, key_ok
from llmobs_server.store import SpanStore

SPAN = {"trace_id": "t", "span_id": "s", "name": "x", "start_ns": 1}


class KeyCheckTests(unittest.TestCase):
    def test_key_ok(self):
        self.assertTrue(key_ok(None, None))
        self.assertTrue(key_ok("anything", ""))
        self.assertTrue(key_ok("Bearer k", "k"))
        for header in (None, "", "k", "Bearer K", "Bearer k ", "bearer k"):
            self.assertFalse(key_ok(header, "k"), header)


@unittest.skipUnless(HAVE_FASTAPI, "fastapi not installed")
class FastApiAuthTests(unittest.TestCase):
    def client(self, api_key):
        from llmobs_server.app import create_app
        return TestClient(create_app(SpanStore(), api_key, ExperimentStore(":memory:")))

    def test_with_a_key_every_data_route_needs_it(self):
        c = self.client("secret")
        ok = {"Authorization": "Bearer secret"}
        for path in ("/v1/traces", "/v1/traces/t", "/v1/experiments", "/v1/experiments/x", "/openapi.json"):
            self.assertEqual(c.get(path).status_code, 401, path)
            self.assertEqual(c.get(path, headers={"Authorization": "Bearer nope"}).status_code, 401, path)
        self.assertEqual(c.post("/v1/spans", json={"spans": [SPAN]}).status_code, 401)
        self.assertEqual(c.post("/v1/spans", json={"spans": [SPAN]}, headers=ok).status_code, 202)
        self.assertEqual(c.get("/v1/traces", headers=ok).json()["traces"][0]["trace_id"], "t")
        self.assertEqual(c.get("/v1/traces/t", headers=ok).status_code, 200)
        self.assertEqual(c.get("/healthz").status_code, 200)
        self.assertEqual(c.get("/").status_code, 200)

    def test_without_a_key_everything_is_open(self):
        c = self.client(None)
        self.assertEqual(c.post("/v1/spans", json={"spans": [SPAN]}).status_code, 202)
        self.assertEqual(c.get("/v1/traces").status_code, 200)

    def test_oversized_bodies_are_refused(self):
        c = self.client(None)
        r = c.post("/v1/spans", content=b"{}", headers={"Content-Length": str(MAX_BODY + 1)})
        self.assertEqual(r.status_code, 413)


if __name__ == "__main__":
    unittest.main()
