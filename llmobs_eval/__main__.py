"""CLI:  python -m llmobs_eval run|resume|report|list ..."""
from __future__ import annotations

import argparse
import json
import sys
import time
from typing import Optional

import llmobs

from . import (ConfigError, DatasetError, ExperimentStore, ProviderError, calibration_payload, compare_payload,
               create_experiment, evaluate_experiment, export_snapshot, format_probe, probe_judge, format_calibration, format_compare,
               format_gate, format_judge_check, format_judge_table, format_table, gate_experiment, gate_markdown,
               judge_check_payload, judge_table_payload, label_session, load_corpus_jsonl, load_gate_config,
               load_jsonl, normalize_config, parse_rules, pick_label_items, record_calibration, run_experiment,
               side_from_snapshot, summarize)
from .gate import Side, check_compatible
from .judge import judge_metric_name
from .patterns import failure_patterns, format_patterns


def _dur(seconds: float) -> str:
    s = int(round(seconds))
    return f"{s}s" if s < 60 else (f"{s // 60}m{s % 60:02d}s" if s < 3600 else f"{s // 3600}h{(s % 3600) // 60:02d}m")


class _Progress:
    """`  37/118 calls  (6m10s elapsed, ~13m left)`: a long run should say how long it will take."""

    def __init__(self) -> None:
        self.start = time.monotonic()

    def __call__(self, done: int, total: int) -> None:
        elapsed = time.monotonic() - self.start
        eta = f", ~{_dur(elapsed / done * (total - done))} left" if 0 < done < total else ""
        sys.stderr.write(f"\r  {done}/{total} calls  ({_dur(elapsed)} elapsed{eta})   ")
        sys.stderr.flush()
        if done == total:
            sys.stderr.write("\n")


def _load_extra_providers(path: Optional[str]) -> Optional[dict]:
    """--extra-providers FILE: provider specs usable for this evaluation only (e.g. a native Ollama judge)."""
    if not path:
        return None
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
    except FileNotFoundError:
        raise ConfigError(f"--extra-providers file not found: {path}") from None
    except json.JSONDecodeError as exc:
        raise ConfigError(f"{path}: invalid JSON ({exc.msg})") from None
    data = data.get("providers", data) if isinstance(data, dict) else data
    if not isinstance(data, dict) or not data or any(not isinstance(v, dict) for v in data.values()):
        raise ConfigError(f"{path}: expected an object mapping provider names to specs, "
                          'e.g. {"ollama-native": {"type": "ollama", "think": false}}')
    return data


def _trace(args) -> None:
    if args.trace_endpoint:
        llmobs.configure(endpoint=args.trace_endpoint, service="llmobs-eval")


def _spec(text: str) -> dict:
    """'ollama:llama3.1:8b' -> {'provider': 'ollama', 'model': 'llama3.1:8b'} (split on the first colon)."""
    provider, sep, model = text.partition(":")
    if not sep or not provider or not model:
        raise ConfigError(f"expected PROVIDER:MODEL, got {text!r}")
    return {"provider": provider, "model": model}


def _judge_spec(args) -> Optional[dict]:
    """--judge PROVIDER:MODEL plus optional --rubric, --judge-param KEY=VALUE and --judge-suffix."""
    if not args.judge:
        return None
    spec = _spec(args.judge)
    extra = {}
    for item in args.judge_param or []:
        key, sep, raw = item.partition("=")
        if not sep or not key:
            raise ConfigError(f"--judge-param expects KEY=VALUE, got {item!r}")
        try:
            extra[key] = json.loads(raw)                  # numbers, true/false, null ...
        except json.JSONDecodeError:
            extra[key] = raw                              # ... otherwise a plain string
    if extra:
        spec["params"] = {"temperature": 0, **extra}
    if args.judge_suffix:
        spec["suffix"] = args.judge_suffix
    if args.rubric != "v1":
        spec["rubric"] = args.rubric
    return spec


def _add_eval_args(p) -> None:
    """The scoring options shared by `evaluate` and `ci`."""
    p.add_argument("--judge", help="PROVIDER:MODEL for LLM-as-judge correctness, e.g. anthropic:claude-sonnet-5-5")
    p.add_argument("--embed", help="PROVIDER:MODEL for embedding similarity, e.g. ollama:nomic-embed-text")
    p.add_argument("--no-lexical", action="store_true", help="skip exact_match / contains / token_f1")
    p.add_argument("--faithfulness", action="store_true",
                   help="with --judge: check each answer's claims against the context it was given (RAG variants)")
    p.add_argument("--no-retrieval", action="store_true", help="skip recall@k / precision@k / hit@k / mrr")
    p.add_argument("--rubric", choices=["v1", "v2"], default="v1",
                   help="judge rubric; v2 is stricter about 'I don't know' and approximate values (own metric: judge:<model>#v2)")
    p.add_argument("--judge-param", action="append", metavar="KEY=VALUE",
                   help="extra request parameter for the judge, repeatable (values parsed as JSON), e.g. temperature=0")
    p.add_argument("--judge-suffix", help="text appended to every judge prompt, e.g. /no_think for qwen3 models")
    p.add_argument("--no-correctness", action="store_true",
                   help="with --judge: skip the correctness grade (e.g. to run only --faithfulness)")
    p.add_argument("--extra-providers", metavar="FILE",
                   help="JSON of extra provider specs for this evaluation only, e.g. examples/providers_ollama_native.json")
    p.add_argument("--concurrency", type=int, help="parallel judge calls (default: the experiment's setting)")
    p.add_argument("--judge-timeout", type=float, metavar="SECONDS",
                   help="per-call timeout for the judge/embedding provider (the experiment's is often 300s, "
                        "so a stuck server costs minutes per answer)")


def _print_eval_stats(res: dict) -> None:
    warns, errs = res.pop("warnings"), res.pop("error_samples")
    print("evaluated:", ", ".join(f"{k}={v}" for k, v in res.items()))
    for e in errs:
        print(f"error: {e}")
    for w in warns:
        print(f"warning: {w}")


def _gate_rules(args):
    """Rules from --config and/or repeated --metric NAME=MAX_DROP (the flags win)."""
    cfg = load_gate_config(args.config) if getattr(args, "config", None) else {}
    for item in getattr(args, "metric", None) or []:
        name, sep, raw = item.rpartition("=")
        try:
            value = float(raw)
        except ValueError:
            value = None
        if not sep or not name or value is None:
            raise ConfigError(f"--metric expects NAME=MAX_DROP, e.g. contains=0.05, got {item!r}")
        entry = cfg.setdefault("metrics", {}).get(name, {})
        cfg["metrics"][name] = {**(entry if isinstance(entry, dict) else {}), "max_drop": value}
    return parse_rules(cfg)


def _preflight(rules, cfg: dict, args) -> None:
    """Refuse to start a run that cannot produce a gated metric (before spending any model calls)."""
    produced = set()
    if not args.no_lexical:
        produced |= {"exact_match", "contains", "token_f1"}
    if not args.no_retrieval:
        for v in cfg["variants"]:
            r = v.get("retriever")
            if r and r["type"] != "oracle":
                produced |= {f"recall@{r['k']}", f"precision@{r['k']}", f"hit@{r['k']}", "mrr"}
    judge = _judge_spec(args)
    if judge and not args.no_correctness:
        produced.add(judge_metric_name(judge["model"], judge.get("rubric", "v1")))
    if judge and args.faithfulness:
        produced.add(f"faithfulness:{judge['model']}")
    if args.embed:
        produced.add(f"embed_sim:{_spec(args.embed)['model']}")
    missing = [r.name for r in rules.metrics if r.name not in produced]
    if missing:
        raise ConfigError(f"the gate needs {missing} but this run will not produce it "
                          f"(it produces: {sorted(produced) or 'nothing'}); add the matching options "
                          "(--judge, --rubric, --faithfulness, --embed) or remove the metric from the gate")


def _write(path: str, text: str) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)


def _emit_gate(result: dict, args) -> int:
    print(format_gate(result))
    if getattr(args, "json", None):
        _write(args.json, json.dumps(result, indent=2, default=str))
    if getattr(args, "markdown", None):
        _write(args.markdown, gate_markdown(result))
    return result["exit_code"]


def _flush_tracing() -> None:
    """Deliver queued spans (bounded wait) and say so if some could not be delivered, and why."""
    exporter = llmobs.get_tracer().exporter
    exporter.flush(timeout=15)
    st = exporter.stats()
    if st.get("failed") or st.get("dropped"):
        print(f"tracing: {st.get('sent', 0)} spans sent, {st.get('failed', 0)} not delivered, "
              f"{st.get('dropped', 0)} dropped; last error: {exporter.last_error() or 'n/a'}", file=sys.stderr)


def _finish(store, exp_id, result) -> None:
    print(f"\nexperiment {exp_id}: ran {result['ran']} calls, {result['remaining']} remaining")
    print(format_table(summarize(store, exp_id)))
    _flush_tracing()


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="llmobs_eval")
    p.add_argument("--db", default="llmobs.db", help="SQLite file shared with the trace server")
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="create and run a new experiment")
    r.add_argument("--dataset", required=True)
    r.add_argument("--config", required=True)
    r.add_argument("--name")
    r.add_argument("--corpus", help="JSONL of documents; required when a variant has a retriever")
    r.add_argument("--trace-endpoint", help="e.g. http://127.0.0.1:8000/v1/spans")
    rs = sub.add_parser("resume", help="finish a partial experiment")
    rs.add_argument("experiment_id")
    rs.add_argument("--trace-endpoint")
    rp = sub.add_parser("report")
    rp.add_argument("experiment_id")
    sub.add_parser("list")
    ev = sub.add_parser("evaluate", help="score a finished experiment (re-runnable; only missing scores are computed)")
    ev.add_argument("experiment_id")
    _add_eval_args(ev)
    ev.add_argument("--only-labeled", action="store_true",
                    help="judge only the answers you have labeled: a cheap way to test another judge against your labels")
    ev.add_argument("--max-items", type=int, metavar="N",
                    help="judge at most N answers in this run; run again to continue (every score is saved as it is made)")
    ev.add_argument("--probe", type=int, nargs="?", const=3, default=0, metavar="N",
                    help="time N (default 3) real judge calls with these settings, project the total, store nothing")
    ev.add_argument("--probe-timeout", type=float, default=60.0, metavar="SECONDS",
                    help="with --probe: give up on a call after this long (default 60)")
    ev.add_argument("--trace-endpoint")
    ev.add_argument("--baseline")
    cp = sub.add_parser("compare", help="compare variants with confidence intervals")
    cp.add_argument("experiment_id")
    cp.add_argument("--baseline", help="variant label to compare against (default: first variant)")
    cp.add_argument("--metric", help="metric used to list regressions (default: judge, else token_f1)")
    jc = sub.add_parser("judge-check", help="cross-check the LLM judge against the lexical `contains` metric")
    jc.add_argument("experiment_id")
    jc.add_argument("--metric", help="judge metric (default: the first one)")
    jc.add_argument("--top", type=int, default=10, help="disagreements to list per kind")
    lb = sub.add_parser("label", help="label answers yourself, blind, to measure how far the judge can be trusted")
    lb.add_argument("experiment_id")
    lb.add_argument("--metric")
    lb.add_argument("--n", type=int, default=20, help="how many answers to label")
    lb.add_argument("--sample", choices=["random", "disagreements"], default="random",
                    help="random = unbiased calibration set; disagreements = hunt for the judge's mistakes")
    lb.add_argument("--labeler", default="human")
    lb.add_argument("--seed", type=int, default=0)
    mt = sub.add_parser("metrics", help="list, hide or unhide an experiment's metrics (scores are never deleted)")
    mt.add_argument("experiment_id")
    mt.add_argument("action", choices=["list", "hide", "unhide"])
    mt.add_argument("names", nargs="*", help="metric names, e.g. judge:qwen3:4b")
    cl = sub.add_parser("calibrate", help="agreement, kappa and leniency bias of the judge against your labels")
    cl.add_argument("experiment_id")
    cl.add_argument("--metric")
    cl.add_argument("--labeler")
    cl.add_argument("--all", action="store_true", help="compare every judge metric on the same labels")
    cl.add_argument("--record", metavar="GATE_JSON",
                    help="write this judge's measured kappa/labels into a gate config's judge_calibration ledger")
    for sp in (sub.add_parser("gate", help="CI gate: exit 1 if a candidate is significantly worse than a baseline"),):
        sp.add_argument("experiment_id")
        sp.add_argument("--config", help="gate config JSON (metrics, tolerances, policy)")
        sp.add_argument("--metric", action="append", metavar="NAME=MAX_DROP",
                        help="gate a metric with this tolerance, repeatable (overrides --config)")
        sp.add_argument("--baseline", help="variant to compare against (default: first variant)")
        sp.add_argument("--candidate", action="append", help="variant(s) to gate (default: all the others)")
        sp.add_argument("--baseline-file", help="baseline snapshot from `baseline-export` instead of a variant")
        sp.add_argument("--allow-dataset-mismatch", action="store_true")
        sp.add_argument("--json", help="write the full result as JSON")
        sp.add_argument("--markdown", help="write a summary for a pull request / $GITHUB_STEP_SUMMARY")
        sp.add_argument("--top", type=int, default=5, help="worst regressions to list per metric")
    fp = sub.add_parser("failures", help="where do the failures concentrate? groups them and tests each group against chance")
    fp.add_argument("experiment_id")
    fp.add_argument("--variant", help="variant to analyse (default: the first one after the baseline, which defaults to the first variant)")
    fp.add_argument("--baseline", help="analyse NEW failures only: examples this variant passes and --variant fails")
    fp.add_argument("--metric", help="metric that defines a failure (default: contains if scored, else judge/token_f1)")
    fp.add_argument("--threshold", type=float, default=0.5, help="a failure is a score below this (default 0.5)")
    fp.add_argument("--group-by", default="category", help="dataset metadata key to group by (default: category)")
    fp.add_argument("--no-cluster", action="store_true", help="skip the similar-question clusters")
    fp.add_argument("--clusters", type=int, metavar="K", help="number of clusters (default: from the number of examples)")
    fp.add_argument("--embed", help="PROVIDER:MODEL to cluster by embeddings instead of tf-idf, e.g. ollama:nomic-embed-text")
    fp.add_argument("--extra-providers", metavar="FILE")
    fp.add_argument("--top", type=int, default=12, help="failing examples to keep per group")
    fp.add_argument("--json", help="write the full analysis as JSON")
    be = sub.add_parser("baseline-export", help="save a variant's per-example scores as a committable baseline")
    be.add_argument("experiment_id")
    be.add_argument("--variant", required=True)
    be.add_argument("--out", required=True)
    ci = sub.add_parser("ci", help="one step for CI: run, evaluate and gate; exit code 0 pass / 1 fail / 2 error")
    ci.add_argument("--dataset", required=True)
    ci.add_argument("--config", required=True, help="experiment config (the variants to run)")
    ci.add_argument("--gate", required=True, help="gate config JSON")
    ci.add_argument("--corpus")
    ci.add_argument("--name")
    ci.add_argument("--baseline", help="variant to compare against (default: the gate's `baseline`, else the first)")
    ci.add_argument("--candidate", action="append")
    ci.add_argument("--baseline-file")
    ci.add_argument("--allow-dataset-mismatch", action="store_true")
    ci.add_argument("--json")
    ci.add_argument("--markdown")
    ci.add_argument("--top", type=int, default=5)
    ci.add_argument("--trace-endpoint")
    _add_eval_args(ci)
    args = p.parse_args(argv)

    store = ExperimentStore(args.db)
    code = 0
    try:
        if args.cmd == "run":
            _trace(args)
            dataset = load_jsonl(args.dataset)
            with open(args.config, encoding="utf-8") as fh:
                config = json.load(fh)
            corpus = load_corpus_jsonl(args.corpus) if args.corpus else None
            exp_id = create_experiment(store, dataset, config, args.name, corpus)
            print(f"created {exp_id}  ({len(dataset.examples)} examples, dataset sha256 {dataset.content_hash[:12]})")
            _finish(store, exp_id, run_experiment(store, exp_id, progress=_Progress()))
        elif args.cmd == "resume":
            _trace(args)
            _finish(store, args.experiment_id, run_experiment(store, args.experiment_id, progress=_Progress()))
        elif args.cmd == "report":
            print(format_table(summarize(store, args.experiment_id)))
        elif args.cmd == "evaluate":
            _trace(args)
            extra = _load_extra_providers(args.extra_providers)
            if args.probe:
                judge = _judge_spec(args)
                if judge is None:
                    raise ConfigError("--probe needs --judge PROVIDER:MODEL")
                print(format_probe(probe_judge(store, args.experiment_id, judge, n=args.probe,
                                               extra_providers=extra, concurrency=args.concurrency,
                                               timeout=args.judge_timeout or args.probe_timeout)))
                return 0
            res = evaluate_experiment(
                store, args.experiment_id, lexical=not args.no_lexical, retrieval=not args.no_retrieval,
                faithfulness=args.faithfulness, correctness=not args.no_correctness, only_labeled=args.only_labeled,
                judge=_judge_spec(args), embed=_spec(args.embed) if args.embed else None, progress=_Progress(),
                extra_providers=extra, max_items=args.max_items, concurrency=args.concurrency,
                timeout=args.judge_timeout)
            _print_eval_stats(res)
            print()
            print(format_compare(compare_payload(store, args.experiment_id, args.baseline)))
            _flush_tracing()
        elif args.cmd == "compare":
            print(format_compare(compare_payload(store, args.experiment_id, args.baseline, args.metric)))
        elif args.cmd == "metrics":
            store.get_experiment(args.experiment_id)                               # KeyError for an unknown experiment
            coverage = {}
            for sc in store.get_scores(args.experiment_id):
                c = coverage.setdefault(sc["metric"], [0, 0])
                c[1] += 1
                c[0] += sc["value"] is not None
            if args.action != "list":
                if not args.names:
                    raise ConfigError("name at least one metric, e.g. judge:qwen3:4b")
                unknown = [n for n in args.names if n not in coverage]
                if unknown:
                    raise ConfigError(f"unknown metric(s) {unknown}; this experiment has {sorted(coverage)}")
                for n in args.names:
                    (store.hide_metric if args.action == "hide" else store.unhide_metric)(args.experiment_id, n)
            hidden = store.hidden_metrics(args.experiment_id)
            answers = sum(1 for r in store.get_results(args.experiment_id) if r["status"] == "ok")
            width = max([len(m) for m in coverage] + [6])
            print(f"{'metric'.ljust(width)}  status   scored")
            for m in sorted(coverage):
                print(f"{m.ljust(width)}  {'hidden' if m in hidden else 'shown ':<7}  {coverage[m][0]} of {answers} answers")
            if hidden:
                print("\nhidden metrics are kept in the database but left out of tables and comparisons; "
                      "ask for one by name (--metric) or run `metrics <id> unhide NAME`")
        elif args.cmd == "judge-check":
            print(format_judge_check(judge_check_payload(store, args.experiment_id, args.metric, args.top)))
        elif args.cmd == "label":
            items = pick_label_items(store, args.experiment_id, args.metric, args.n, args.sample, args.seed,
                                     args.labeler)
            if not items:
                print("nothing left to label for this selection")
            else:
                print(f"{len(items)} answers to label. You will not see the judge's verdict, so your labels stay "
                      "independent of it. Type q to stop; every label is saved immediately.")
                saved = label_session(store, args.experiment_id, items, labeler=args.labeler)
                print(f"\nsaved {saved} labels. Next: python -m llmobs_eval calibrate {args.experiment_id}")
        elif args.cmd == "calibrate":
            if args.all and args.record:
                raise ConfigError("--record records one judge: use --metric instead of --all")
            if args.all:
                print(format_judge_table(judge_table_payload(store, args.experiment_id, args.labeler)))
            else:
                payload = calibration_payload(store, args.experiment_id, args.metric, args.labeler)
                print(format_calibration(payload))
                if args.record:
                    entry = record_calibration(args.record, payload["metric"], payload)
                    print(f"\nrecorded in {args.record}: {payload['metric']} kappa {entry['kappa']} on {entry['labels']} labels")
        elif args.cmd == "gate":
            rules = _gate_rules(args)
            code = _emit_gate(gate_experiment(store, args.experiment_id, rules, baseline_label=args.baseline,
                                              candidates=args.candidate, baseline_file=args.baseline_file,
                                              allow_dataset_mismatch=args.allow_dataset_mismatch, top_k=args.top), args)
        elif args.cmd == "failures":
            labels = [v["label"] for v in store.get_experiment(args.experiment_id)["config"]["variants"]]
            variant = args.variant or next((l for l in labels if l != (args.baseline or labels[0])), labels[0])
            payload = failure_patterns(
                store, args.experiment_id, variant, baseline=args.baseline, metric=args.metric,
                threshold=args.threshold, group_by=args.group_by, cluster=not args.no_cluster, k=args.clusters,
                embed=_spec(args.embed) if args.embed else None,
                extra_providers=_load_extra_providers(args.extra_providers), top_examples=args.top)
            print(format_patterns(payload))
            if args.json:
                _write(args.json, json.dumps(payload, indent=2, default=str))
        elif args.cmd == "baseline-export":
            snap = export_snapshot(store, args.experiment_id, args.variant, args.out)
            print(f"wrote {args.out}: variant {snap['variant']!r}, {snap['dataset']['n_examples']} examples, "
                  f"metrics {list(snap['scores'])}, dataset {snap['dataset']['hash'][:12]}")
        elif args.cmd == "ci":
            _trace(args)
            rules = parse_rules(load_gate_config(args.gate))                     # validate everything before any model call
            dataset = load_jsonl(args.dataset)
            with open(args.config, encoding="utf-8") as fh:
                config = json.load(fh)
            corpus = load_corpus_jsonl(args.corpus) if args.corpus else None
            _preflight(rules, normalize_config(config), args)
            if args.baseline_file:
                probe = Side("this run", {}, set(), 0, None, None, {}, {
                    "dataset_hash": dataset.content_hash, "corpus_hash": corpus.content_hash if corpus else None})
                check_compatible(side_from_snapshot(args.baseline_file), probe, args.allow_dataset_mismatch)
            exp_id = create_experiment(store, dataset, config, args.name, corpus)
            print(f"created {exp_id}  ({len(dataset.examples)} examples, dataset sha256 {dataset.content_hash[:12]})")
            ran = run_experiment(store, exp_id, progress=_Progress())
            print(f"ran {ran['ran']} calls")
            res = evaluate_experiment(store, exp_id, lexical=not args.no_lexical, retrieval=not args.no_retrieval,
                                      faithfulness=args.faithfulness, correctness=not args.no_correctness,
                                      judge=_judge_spec(args), embed=_spec(args.embed) if args.embed else None,
                                      progress=_Progress(), extra_providers=_load_extra_providers(args.extra_providers),
                                      concurrency=args.concurrency, timeout=args.judge_timeout)
            _print_eval_stats(res)
            print()
            code = _emit_gate(gate_experiment(store, exp_id, rules, baseline_label=args.baseline,
                                              candidates=args.candidate, baseline_file=args.baseline_file,
                                              allow_dataset_mismatch=args.allow_dataset_mismatch, top_k=args.top), args)
            _flush_tracing()
        elif args.cmd == "list":
            for e in store.list_experiments():
                print(f"{e['id']}  {e['status']:<9} {e['name']}  ({e['dataset_name']})")
    except (ConfigError, DatasetError, ProviderError, KeyError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("\ninterrupted; results so far are saved. Resume with: python -m llmobs_eval resume <id>",
              file=sys.stderr)
        return 130
    return code


if __name__ == "__main__":
    sys.exit(main())
