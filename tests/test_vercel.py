"""The Vercel entry point refuses to serve unless it has an API key and a Postgres database."""
import importlib
import json
import os
import unittest
from unittest import mock

try:
    from fastapi.testclient import TestClient
    HAVE_FASTAPI = True
except ImportError:
    HAVE_FASTAPI = False

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLEAN = {k: "" for k in ("LLMOBS_API_KEY", "LLMOBS_DB", "DATABASE_URL", "POSTGRES_URL")}


def load_entry(env):
    with mock.patch.dict(os.environ, {**CLEAN, **env}):
        for k in [k for k, v in {**CLEAN, **env}.items() if not v]:
            os.environ.pop(k, None)
        import api.index as entry
        return importlib.reload(entry)


@unittest.skipUnless(HAVE_FASTAPI, "fastapi not installed")
class VercelEntryTests(unittest.TestCase):
    def test_missing_key_and_database_are_both_reported(self):
        entry = load_entry({})
        self.assertEqual(len(entry.PROBLEMS), 2)
        r = TestClient(entry.app).get("/v1/traces")
        self.assertEqual(r.status_code, 503)
        self.assertIn("LLMOBS_API_KEY", json.dumps(r.json()))

    def test_a_file_database_is_refused_on_vercel(self):
        entry = load_entry({"LLMOBS_API_KEY": "k", "LLMOBS_DB": "llmobs.db"})
        self.assertEqual(len(entry.PROBLEMS), 1)
        self.assertIn("postgres://", entry.PROBLEMS[0])
        self.assertEqual(TestClient(entry.app).post("/v1/spans", json={}).status_code, 503)

    def test_config_with_key_and_postgres_has_no_problems(self):
        with mock.patch.dict(os.environ, {"LLMOBS_API_KEY": "k", "DATABASE_URL": "postgresql://u:p@h/db"}):
            os.environ.pop("LLMOBS_DB", None)
            import api.index as entry
            self.assertEqual(entry.config_problems(), [])

    def test_vercel_config_routes_everything_to_the_function(self):
        with open(os.path.join(ROOT, "vercel.json"), encoding="utf-8") as fh:
            cfg = json.load(fh)
        self.assertEqual(cfg["rewrites"], [{"source": "/(.*)", "destination": "/api/index"}])
        self.assertIn("api/index.py", cfg["functions"])
        with open(os.path.join(ROOT, "requirements.txt"), encoding="utf-8") as fh:
            reqs = fh.read()
        for dep in ("fastapi", "pydantic", "psycopg"):
            self.assertIn(dep, reqs)


if __name__ == "__main__":
    unittest.main()
