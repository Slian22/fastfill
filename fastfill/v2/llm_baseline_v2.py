"""Versioned prompt/harness baseline that retains the last valid layout if a repair fails.

The historical ``llm_baseline`` module remains frozen for reproducibility. This entry point
uses its parser, prompts and transport, records both implementation hashes, and writes only
to a new output directory. A failed repair is recorded as ``repair_error``; it never causes
another first-answer request or discards an already valid layout.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import time

from fastfill.v2 import llm_baseline as legacy
from fastfill.v2.evaluate import project_minimal
from fastfill.v2.io import fingerprint, read_samples, safe_output

VERSION = "repair-v2"


def run(data, output, env_file, *, model=None, mode="prompt", repairs=2, workers=4,
        max_samples=None, retries=1, ask=legacy.chat):
    if mode not in ("prompt", "harness"):
        raise ValueError("mode must be prompt or harness")
    for name, value, minimum in (("repairs", repairs, 0), ("retries", retries, 0), ("workers", workers, 1)):
        if type(value) is not int or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    if max_samples is not None and (type(max_samples) is not int or max_samples < 1):
        raise ValueError("max_samples must be None or a positive integer")
    target = safe_output(output, create=True)
    env = legacy.load_env(env_file)
    model = model or env.get("OPENAI_MODEL")
    if not model:
        raise ValueError("pass --model or set OPENAI_MODEL in the env file")
    limiter = legacy.RateLimiter(float(env.get("FASTFILL_LLM_RPM", 30)))
    identity = {"data_sha256": fingerprint(data), "implementation_sha256": fingerprint(__file__),
                "legacy_implementation_sha256": fingerprint(legacy.__file__), "harness_version": VERSION}
    rows = [p for p in map(project_minimal, read_samples(data)) if p is not None][:max_samples]
    calls = [0]

    def call(messages):
        calls[0] += 1
        return (ask(env, model, messages) if ask is not legacy.chat
                else legacy.chat(env, model, messages, limiter=limiter))

    def one(row):
        start, error, condition, valid = time.perf_counter(), None, row["condition"], []
        for attempt in range(1, retries + 2):
            try:
                messages = [{"role": "user", "content": legacy.prompt(condition)}]
                reply = call(messages)
                valid.append(False)
                layout = legacy.to_layout(reply, condition)
                valid[-1] = True
                found = legacy.problems(layout, condition)
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"[:500]
                continue
            rounds, first, repair_error = 0, len(found), None
            while mode == "harness" and found and rounds < repairs:
                messages = [*messages, {"role": "assistant", "content": reply}, {"role": "user", "content":
                            "Your layout has these problems:\n- " + "\n- ".join(found)
                            + "\nReturn the complete corrected JSON for every id."}]
                try:
                    next_reply = call(messages)
                    candidate = legacy.to_layout(next_reply, condition)
                    next_problems = legacy.problems(candidate, condition)
                except Exception as exc:
                    repair_error = f"{type(exc).__name__}: {exc}"[:500]
                    break
                rounds += 1
                reply, layout, found = next_reply, candidate, next_problems
            return {"layout": layout, "error": None, "repair_error": repair_error,
                    "latency_s": time.perf_counter() - start, "attempts": attempt,
                    "first_answer_valid": valid[0], "repair_rounds": rounds,
                    "problems_first": first, "problems_final": len(found)}
        return {"layout": None, "error": error, "latency_s": time.perf_counter() - start,
                "attempts": retries + 1, "first_answer_valid": valid[0] if valid else None}

    with ThreadPoolExecutor(workers) as pool:
        results = list(pool.map(one, rows))
    with (target / "rows.jsonl").open("x") as stream:
        stream.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    with (target / "predictions.jsonl").open("x") as stream:
        stream.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in results)
    ok = [r for r in results if r["layout"] is not None]
    summary = {"model": model, "mode": mode, "request_parameters": legacy.request_parameters(env, model),
               "endpoint": env.get("OPENAI_BASE_URL", "").rstrip("/") + "/chat/completions",
               "settings": {"repairs": repairs, "retries": retries, "max_samples": max_samples},
               "rows": len(rows), "failed": len(results) - len(ok), "api_calls": calls[0],
               "first_answer_invalid": sum(r["first_answer_valid"] is False for r in results),
               "unanswered": sum(r["first_answer_valid"] is None for r in results),
               "repair_failed": sum(r.get("repair_error") is not None for r in results),
               "mean_latency_s": sum(r["latency_s"] for r in results) / max(1, len(results)),
               "mean_problems_first": sum(r["problems_first"] for r in ok) / max(1, len(ok)),
               "mean_problems_final": sum(r["problems_final"] for r in ok) / max(1, len(ok)),
               "data": str(data), **identity}
    with (target / "summary.json").open("x") as stream:
        stream.write(json.dumps(summary, indent=2) + "\n")
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--env", type=Path, required=True)
    parser.add_argument("--model")
    parser.add_argument("--mode", choices=("prompt", "harness"), default="prompt")
    parser.add_argument("--repairs", type=int, default=2)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-samples", type=int)
    args = parser.parse_args(argv)
    print(json.dumps(run(args.data, args.output, args.env, model=args.model, mode=args.mode,
                         repairs=args.repairs, workers=args.workers, max_samples=args.max_samples)))


if __name__ == "__main__":
    main()
