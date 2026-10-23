# llmobs: tracing, experiment runner, evaluators, comparison, RAG evaluation, judge calibration, CI gating

## Use the SDK
```python
import llmobs
llmobs.configure(endpoint="http://127.0.0.1:8000/v1/spans", service="my-app")

@llmobs.trace(kind="retrieval")
def retrieve(q): ...

@llmobs.trace(name="rag_pipeline")
def answer(q):
    docs = retrieve(q)
    with llmobs.span("generate", kind="llm") as s:
        reply = call_model(q, docs)
        s.record_llm("model-name", input_tokens=120, output_tokens=40)
    return reply
```
`llmobs.record_llm(model, input_tokens, output_tokens)` and `llmobs.record_retrieval(query, documents)` annotate the current span and do nothing outside a trace. (`llmobs.current_span()` returns `Optional[Span]`, so check for `None` before using it directly.) Works with sync and async functions; nesting follows `contextvars`, so concurrent
asyncio tasks get separate traces. Exports are queued and batched on a background
thread; a full queue or dead backend drops spans and never blocks or raises into your app.

## Run it
```bash
pip install -e ".[server]"
uvicorn llmobs_server.app:app          # API + viewer at http://127.0.0.1:8000
python examples/rag_demo.py            # emits sample traces
python -m unittest -v tests.test_llmobs
```
Env: `LLMOBS_DB` (SQLite path, default `llmobs.db`), `LLMOBS_API_KEY` (require `Authorization: Bearer ...`).

## Layout
- `llmobs/` SDK: `tracer.py` (Span, `@trace`, `span()`), `exporters.py` (HTTP batching, in-memory)
- `llmobs_server/` `store.py` (SQLite, swappable), `app.py` (FastAPI), `static/viewer.html` (waterfall UI)
- Span attributes follow OpenTelemetry GenAI names (`gen_ai.request.model`, `gen_ai.usage.*_tokens`).


## Experiments (step 2)
Run the same dataset through several model/prompt variants in parallel, with retries,
rate limiting, tracing and cost tracking.

```bash
python -m llmobs_eval run --dataset examples/dataset.jsonl --config examples/experiment_mock.json
python -m llmobs_eval run --dataset examples/dataset.jsonl --config examples/experiment_real.json \
       --trace-endpoint http://127.0.0.1:8000/v1/spans   # optional: traces show up in the viewer
python -m llmobs_eval list | report <id> | resume <id>
```

**Dataset** (JSONL): `input` (required), optional `id`, `reference`, `reference_docs`,
`relevant_doc_ids`, `metadata`. Errors name the offending line.

**Config** (JSON): `providers` (`mock`, `anthropic`, `openai_compat`; keys come from the env var named
in `api_key_env`, never from the file; optional `rps` rate limit), `variants` (label, provider, model,
`system`, `prompt_template` containing `{input}`, `params` passed straight to the API),
`concurrency`, `max_retries`, `pricing` (USD per million tokens, per model; cost is blank if unset).
`openai_compat` covers OpenAI, Ollama, vLLM, OpenRouter and similar.

**Guarantees**
- Retries: 408/409/425/429/5xx and network errors, exponential backoff with jitter, `Retry-After` honored.
  Auth errors (401/403) fail immediately. An error the provider *answered* with (HTTP 4xx/5xx after retries) is recorded as that call's result and never aborts the run. A call that gets **no answer at all** (connection refused, DNS failure, timeout) is *not* recorded: results are immutable, so recording an outage would ruin the experiment. The run stops after 5 consecutive connection failures, the experiment stays `partial`, and `resume <id>` redoes exactly the missing calls.
- Immutability is enforced by SQLite triggers: experiment config, dataset snapshot and results can't be
  updated or deleted. The dataset is snapshotted and hashed, so editing the file later changes nothing.
- Ctrl+C is safe: `resume <id>` runs only the missing (variant, example) pairs. An outage no longer needs a new experiment (see above). To re-run calls that failed with an
  HTTP error, start a new experiment; old results stay as they were.
- Each result stores the `trace_id` of its call tree (root span -> one `llm_call` span per attempt).

## Evaluating and comparing (step 3)
```bash
python -m llmobs_eval evaluate <id>                                   # exact_match, contains, token_f1
python -m llmobs_eval evaluate <id> --judge anthropic:claude-sonnet-5-5
python -m llmobs_eval evaluate <id> --judge ollama:llama3.1:8b --embed ollama:nomic-embed-text
python -m llmobs_eval compare <id> [--baseline LABEL] [--metric NAME]
```
Open the viewer's **Experiments** tab for the same comparison with a baseline picker and expandable regressions.
API: `GET /v1/experiments`, `GET /v1/experiments/{id}/compare?baseline=&metric=`.

- **Scores are append-only** (DB triggers) and `evaluate` is re-runnable: it only computes what is missing, so
  a flaky judge or embedding endpoint is fixed by simply running it again.
- **Judge**: 3-level verdict (correct / partially_correct / incorrect = 1 / 0.5 / 0) against the reference, JSON
  output, one automatic re-ask if the reply isn't parseable (still unparseable -> stored as an empty score and
  reported as missing coverage). Candidate text is fenced as data to resist prompt injection. The judge provider
  must be listed in the experiment's `providers` (it doesn't need a variant). Judging with a model that is also a
  candidate triggers a self-preference warning.
- **Lexical metrics** are cheap sanity floors. They punish verbose-but-correct answers (token_f1) and can't see
  paraphrase; trust the judge for open-ended answers and check it against human labels (planned: step 6).
- **Comparison** = per-variant mean with a 95% interval (Wilson for pass/fail metrics, so 20/20 is not shown
  as a certain [1.00, 1.00]; bootstrap for graded ones), plus a *paired* bootstrap of (candidate - baseline) over
  the same examples. A delta is starred only if its CI excludes 0 **and** there are at least 10 paired examples.
  Generation errors are excluded from metric means and reported separately; partial experiments are flagged.
- Local models: set `"timeout": 300` on the provider. The first request often includes a cold model load.

## Looking at individual answers
The Experiments tab lists every example under the comparison table, worst first by the chosen metric (examples
where a variant failed to generate come before everything else). Expand one to see the question, the reference,
each variant's output side by side, every metric score, the judge's verdict and reason, and a link to the trace
of that call (needs `--trace-endpoint` when running/evaluating). API: `GET /v1/experiments/{id}/examples`.

## Try it on local models
```bash
python -m llmobs_eval run --dataset examples/dataset_general.jsonl --config examples/experiment_ollama_ab.json \
       --trace-endpoint http://127.0.0.1:8000/v1/spans
python -m llmobs_eval evaluate <id> --judge ollama:qwen2.5:3b --trace-endpoint http://127.0.0.1:8000/v1/spans
```
`dataset_general.jsonl` is 20 general-knowledge questions with short references, so `contains` is meaningful and
there are enough examples (>= 10) for significance. The company-docs questions in `dataset.jsonl` can't be answered
without context; they are for the RAG evaluation in the next step. The judge is `PROVIDER:MODEL` and the provider must appear in the experiment's
config (here the same Ollama server, with a different model than the candidates).

## A harder dataset: `examples/dataset_hard.jsonl`
44 questions where a small local model should land well below 100%, so variants have room to differ:
multi-step arithmetic (6), calendar math (5), letter counting (5), classic trick questions (6), unit conversion (3),
sequences (3), commonly-missed facts (12) and false-premise questions (4). Every computable answer is recomputed
from scratch in `tests/test_dataset_hard.py`.

Each row's `metadata` says how far the lexical metrics can be trusted on it:
- `check: contains` (36 items): short canonical reference that is not in the question, so the `contains` metric is a
  trustworthy cross-check on the LLM judge. Where judge and `contains` disagree, read the example.
- `check: contains_leaky` (h18, h22): the answer also appears in the question, so a wrong answer that restates
  the question would pass `contains`; only the judge (or a human) can validate these.
- `check: judge_only` (4 false-premise items): the right behavior is to push back, so `contains` is always 0 here
  and only the judge can score them (a good test of the judge itself).

```bash
python -m llmobs_eval run --dataset examples/dataset_hard.jsonl --config examples/experiment_ollama_ab.json \
       --trace-endpoint http://127.0.0.1:8000/v1/spans
python -m llmobs_eval evaluate <id> --judge ollama:<a-different-model> --trace-endpoint http://127.0.0.1:8000/v1/spans
```
Number formatting is normalized before matching (`$9` ~ `9`, `10,800` ~ `10800`, `April 11th` ~ `April 11`).

## RAG evaluation (step 4)
A variant can retrieve documents and put them in the prompt. Add a corpus and a `retriever` to the variant:

```json
{"label": "bm25-k3", "provider": "ollama", "model": "llama3.2:latest",
 "system": "Answer using only the provided context. If the context does not contain the answer, reply exactly: I don't know.",
 "prompt_template": "Context:\n{context}\n\nQuestion: {input}\nAnswer in one short sentence.",
 "retriever": {"type": "bm25", "k": 3}}
```
- `bm25`: pure-Python BM25 over title + text with a light stemmer; no dependencies.
- `embedding`: dense retrieval through a provider's `/embeddings` endpoint, e.g.
  `{"type": "embedding", "k": 3, "provider": "ollama", "model": "nomic-embed-text"}` (`ollama pull nomic-embed-text`).
  The corpus is embedded once per run; queries are cached.
- `oracle`: hands the model the gold documents from `relevant_doc_ids`. It is the upper bound for any retriever:
  if `oracle` scores poorly, the problem is generation, not retrieval.
- A variant with no retriever is the **closed-book** baseline. `{context}` and `retriever` must go together.

The corpus is JSONL (`id`, `text`, optional `title`) passed with `--corpus`, and is snapshotted and hashed into the
experiment like the dataset. Which documents each call retrieved (rank + score) is stored and is visible, with
hit/miss marks against the gold labels, in the Examples view; the retrieval step is a span in the same trace as
generation.

```bash
python -m llmobs_eval run --dataset examples/dataset_rag.jsonl --corpus examples/corpus_lumen.jsonl \
       --config examples/experiment_rag.json --trace-endpoint http://127.0.0.1:8000/v1/spans
python -m llmobs_eval evaluate <id> --judge ollama:<a-different-model> --faithfulness \
       --trace-endpoint http://127.0.0.1:8000/v1/spans
# bm25 vs embeddings:  experiment_rag_embed.json
```

**Retrieval metrics** (computed automatically from `relevant_doc_ids`, no LLM): `recall@k`, `precision@k`,
`hit@k`, `mrr`. `precision@k` divides by k, not by how many documents came back, so returning nothing is not
rewarded. Oracle and closed-book variants are skipped (the metrics don't apply); unanswerable questions have no
labels and are skipped (`skipped_no_gold_docs`).

**Faithfulness** (`--faithfulness`, needs `--judge`): the judge first lists the answer's factual claims, then checks
each against *only the context the model was given*. Score = supported / total claims; "contradicted" counts as
unsupported. A claim that is true in the world but absent from the context is unsupported, which is exactly the
hallucination this catches. An answer with no factual claims (an "I don't know") scores 1.0 and is flagged
`no_claims`. Closed-book variants are skipped. Use `--no-correctness` to run only faithfulness.

Reading the two together: low recall -> fix retrieval; good recall but low correctness vs `oracle` -> fix the prompt
or model; right answer but low faithfulness -> the model answered from memory instead of the documents.

**Example data**: `corpus_lumen.jsonl` (31 documents for a fictional company, with deliberate confusers: many
mention "30 days", "14 days", "5 GB") and `dataset_rag.jsonl` (48 questions: lookups, number confusers, multi-doc and
multi-hop, 12 vocabulary-mismatch paraphrases, 5 unanswerable). Measured on this data, BM25 reaches recall@3 = 0.86
overall but only 0.50 on the paraphrase questions (0.25 at k=1), so a better retriever has room to win.
`tests/test_rag_data.py` recomputes every checkable answer from the documents and guards that headroom.

## Is the judge trustworthy? (judge-check, label, calibrate)
An LLM judge is itself a noisy measuring instrument, and a small local one can be badly lenient (for example crediting
an "I don't know" against a reference of "24"). Three tools check it, using only data already in the database:

```bash
python -m llmobs_eval judge-check <id>              # judge vs the lexical `contains` metric, with every disagreement
python -m llmobs_eval label <id> --n 20             # label answers yourself, blind (random sample)
python -m llmobs_eval calibrate <id>                # agreement, Cohen's kappa [95% CI], leniency bias vs your labels
```
- **judge-check** compares the judge with `contains` on answers whose reference is a short value (<= 3 words, not
  already in the question; sentence-length references are excluded). *Too lenient* = the judge credits an answer that
  does not contain the reference value; *too strict* = it rejects one that does. These are cases to read, not
  verdicts: either side can be the wrong one. `compare` warns automatically when they disagree on 20% or more.
- **label** shows question, reference and answer only. The judge's verdict is never shown, so your labels stay
  independent. Every label is saved immediately (`q` to stop; `--sample disagreements` hunts for the judge's mistakes
  instead of building a calibration set). Labels may be corrected by labeling the same item again.
- **calibrate** reports agreement on three levels (incorrect / partial / correct), kappa on correct-or-not with a
  bootstrap interval, and the judge's leniency bias overall and per variant (positive = more lenient than you).
  Random labels give an unbiased estimate; labels chosen from disagreements are reported separately and flagged,
  because oversampling hard cases understates agreement. Fewer than 20 labels is flagged as unstable.
- The viewer shows the same under the comparison table ("Judge check"), including the disagreements to review.
- Compare judges by scoring the same experiment with another model (`evaluate --judge ollama:<other-model>`): each
  judge model is its own metric, and `--metric` selects which one to check.

## Choosing and comparing judges
Labels belong to the *answers*, not to a judge, so any number of judges can be measured against the same human labels.
Each judge model (and rubric) is its own metric, so they never mix:

```bash
# a challenger judge, run only on the answers you already labeled (cheap):
python -m llmobs_eval evaluate <id> --judge ollama:qwen3:4b --only-labeled --no-lexical --no-retrieval
# the same judge with the stricter rubric (metric judge:<model>#v2):
python -m llmobs_eval evaluate <id> --judge ollama:qwen2.5:3b --rubric v2 --only-labeled --no-lexical --no-retrieval
python -m llmobs_eval calibrate <id> --all          # judges side by side: kappa [CI], bias, lenient/strict vs contains
#   (lenient/strict are counted on the answers every judge scored, so a judge run with --only-labeled is comparable)
```
- `--rubric v2` adds explicit rules aimed at the failure small judges show: "I don't know" and approximate values
  are incorrect unless the reference says so. `v1` stays the default and is unchanged.
- Reasoning models (e.g. qwen3) may emit `<think>...</think>`: it is stripped before parsing, and a verdict that
  only appears inside the thinking is ignored. Thinking can be slow on CPU; `--judge-suffix "/no_think"` appends a
  switch to every prompt, and `--judge-param KEY=VALUE` (repeatable, JSON values) passes extra request parameters.
- Decide on a clear difference (a large bias gap, or kappa well beyond the intervals), not a small one: with ~25
  labels the interval is wide.

## CI gating: fail a pull request when quality drops
```bash
python -m llmobs_eval ci --dataset D.jsonl --config experiment.json --gate gate.json   # run + evaluate + gate in one step
python -m llmobs_eval gate <id> --config gate.json [--baseline main] [--candidate pr]  # gate an existing experiment
python -m llmobs_eval gate <id> --metric contains=0.05 --metric recall@3=0.10          # tolerances straight from flags
python -m llmobs_eval baseline-export <id> --variant main --out ci/baseline.json        # a baseline you can commit
```
**Exit codes:** `0` pass (warnings allowed) | `1` fail | `2` error or misconfiguration. `--json PATH` writes the
full result and `--markdown PATH` a summary for a pull request (`$GITHUB_STEP_SUMMARY`).

**The decision rule** (per gated metric, comparing the candidate with the baseline *on the same examples*):
- **FAIL** when the mean dropped by more than `max_drop` **and** a paired bootstrap says the drop is not noise
  (the interval excludes 0, with at least `min_pairs` = 10 examples). Also FAIL if the candidate's mean is below `min`.
- **inconclusive** when it dropped by more than `max_drop` but the drop cannot be told from noise. This is a warning
  by default (`"on_inconclusive": "fail"` makes it a failure) and says how small a difference the comparison can resolve.
- Otherwise PASS. Small significant drops inside the tolerance pass; improvements pass.
- **A gate that cannot decide never passes**: a metric that was never scored, no examples in common, a typo in the
  config (unknown keys are rejected) are errors (exit 2). `ci` checks that every gated metric can be produced
  *before* spending any model call.
- **Generation errors count as failures** (score 0), so a change that breaks calls cannot hide behind the examples
  that still worked (`errors_as_zero`). `max_error_rate` and `max_p95_ratio` gate failures and latency directly.

```json
{
  "baseline": "main",
  "metrics": {"contains": {"max_drop": 0.10}, "recall@3": 0.10, "judge:qwen2.5:3b": {"max_drop": 0.10, "min": 0.6}},
  "max_error_rate": 0.05, "max_p95_ratio": 2.0,
  "on_inconclusive": "warn",
  "judge_trust": {"min_kappa": 0.6, "min_labels": 20},
  "judge_calibration": {"judge:qwen2.5:3b": {"kappa": 0.74, "labels": 25}}
}
```
Other settings: `alpha` (interval level, default 0.05), `n_boot`, `min_pairs`, `bonferroni` (divide alpha by the number of
gated metrics), `candidates`. See `examples/ci/`.

**Two ways to get a baseline.** (A) Put the current production config and the pull request's config in the same
experiment as two variants (`main`, `pr`): same data, same time, but twice the model calls. (B) `baseline-export`
a variant to a JSON file and commit it, like a golden file; CI then runs only the candidate. A snapshot records the
dataset hash and the gate refuses to compare against a different dataset (`--allow-dataset-mismatch` compares
only the examples both contain). Re-export it deliberately when you accept a new baseline.

**Judge metrics need evidence.** With `judge_trust` set, a judge metric cannot give a clean pass unless its
calibration is on record: from your local human labels, or from the `judge_calibration` ledger in the gate config.
Record it with `calibrate <id> --metric judge:... --record gate.json`; the evidence is then reviewed in the same pull
request as the gate, and swapping the judge model invalidates it until it is re-calibrated. (Faithfulness is not
calibrated yet, so it is better left out of a gate.)

### How many examples do I need? (measured by simulation)
The gate is only as sensitive as the dataset is big. Chance of **FAIL** when the candidate is truly worse by the amount
shown (binary metric, `max_drop` 0.02, 95% interval; 150 simulated comparisons per cell, so +/- ~4 points):

| examples | no change | 0.05 worse | 0.10 worse | 0.15 worse | 0.20 worse |
|---:|---:|---:|---:|---:|---:|
| 20  | 2%  | 3%  | 9%  | 15% | 31% |
| 48  | 5%  | 9%  | 23% | 39% | 60% |
| 100 | 1%  | 11% | 41% | 69% | 91% |
| 200 | 4%  | 17% | 60% | 91% | 99% |

- It rarely cries wolf (the "no change" column is the false-failure rate, near the 5% designed for). But with 48
  examples a true 0.10 regression fails the build only about 1 time in 4: **catching small regressions needs
  100-200+ examples**. The rest are warnings at best.
- The tolerance mostly controls *noise warnings*, not detection. Share of unchanged candidates that raise a warning
  or failure: `max_drop` 0.02: 42% at 48 examples, 35% at 100; 0.05: 27% / 18%; **0.10: 15% / 3%**. Set `max_drop`
  near the noise floor, about 0.10 at 50 examples and 0.05 at 100-200, or warnings become noise people ignore.
- Assumptions: each example's outcome varies randomly between runs. With temperature 0 and a deterministic stack an
  unchanged candidate reproduces the baseline exactly (no noise alarms at all), and a real change alters only some
  outputs; the interval then asks whether that generalizes beyond your examples, which is what the table's
  detection columns describe.

**What to gate on.** `contains` (deterministic, for short reference answers), retrieval metrics, and a judge only
after calibrating it. Keep the retriever's `k` the same in baseline and candidate (metric names include `k`, e.g.
`recall@3`); use temperature 0; keep gated metrics few (each one is another chance for a false alarm).

### GitHub Actions
`examples/ci/github-actions.yml` runs a no-secrets smoke test of the pipeline with mock models and has the real
gate as a commented step. The workflow file itself has not been run on GitHub (it was written outside any GitHub
environment); every command in it was run locally.

### Partially scored metrics
A judge run with `--only-labeled` scores only some answers. The comparison table marks this (`0.17 [0.05,0.45] (12/48)`
in the terminal, `n=12/48` in the viewer) and prints `too few pairs to compare (n=2)` instead of a delta computed
from a handful of overlapping examples. Run `evaluate --judge ... ` again without `--only-labeled` to score the rest
(answers already scored are skipped).

## Troubleshooting
**`cannot reach http://localhost:11434/...: [WinError 10061] ... actively refused it`**: nothing is listening there, i.e.
the Ollama server is not running (start the Ollama app or run `ollama serve`; `ollama list` shows whether it is up).
Nothing was lost: connection failures are never recorded as results. `run`/`ci` stop after 5 consecutive failures and
tell you to `resume <id>`; `evaluate` stops the same way and the answers it did not reach are scored by simply
running the same command again. A wrong port in the config produces the same message (it names the URL it tried).
