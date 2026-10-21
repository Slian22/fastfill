"""Offline contract tests. Run with PYTHONDONTWRITEBYTECODE=1 python -m unittest
discover -s docs/research/20261011-v11-clean-dpo/paper-metrics-20261013
-p test_server_api_judge.py -v. No real HTTP transport is used.
"""

import base64
import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import signal
import struct
import tempfile
import time
import unittest
from unittest import mock
import urllib.error
import zlib

import server_api_judge as judge


MODEL = "gpt-6.1-sol"
SCORES = {"functionality_score": 7, "layout_score": 8, "aesthetics_score": 6}
FINAL = json.dumps(SCORES)


def png_chunk(kind, data):
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))


PNG = (b"\x89PNG\r\n\x1a\n" + png_chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0))
       + png_chunk(b"IDAT", zlib.compress(b"\x00\xff\xff\xff")) + png_chunk(b"IEND", b""))


def digest(value):
    return hashlib.sha256(value).hexdigest()


def event(kind, **values):
    return ("event: " + kind + "\r\ndata: "
            + json.dumps({"type": kind, **values}) + "\r\n\r\n").encode()


def completed(text=FINAL, model=MODEL, status="completed", **values):
    return event("response.completed", response={
        "model": model, "status": status,
        "output": [{"type": "message", "role": "assistant", "content": [
            {"type": "output_text", "text": text}]}], **values})


class Response(io.BytesIO):
    status = 200
    headers = {"Content-Type": "text/event-stream; charset=utf-8"}


class ParserTests(unittest.TestCase):
    def test_scores_and_optional_fields(self):
        expected = {**SCORES, "overall_score": 7, "comments": "Visible access is clear."}
        self.assertEqual(judge.parse_scores(json.dumps(expected)), expected)

    def test_optional_overall_accepts_finite_averages_and_bounds(self):
        for overall in [0, 0.0, 7.0, 22 / 3, 10, 10.0]:
            expected = {**SCORES, "overall_score": overall}
            with self.subTest(overall=overall):
                self.assertEqual(judge.parse_scores(json.dumps(expected)), expected)

    def test_rejects_invalid_or_extracted_scores(self):
        invalid = ["```json\n" + FINAL + "\n```", "Result: " + FINAL,
                   FINAL + FINAL, "[]", "null", "{}",
                   FINAL[:-1] + ', "layout_score": 9}',
                   json.dumps({**SCORES, "extra": 0}),
                   json.dumps({**SCORES, "comments": []})]
        for field in SCORES:
            invalid.extend(json.dumps({**SCORES, field: value})
                           for value in [True, False, 1.0, "5", -1, 11, None, float("nan")])
        invalid.extend(json.dumps({**SCORES, "overall_score": value})
                       for value in [True, False, "5", -0.1, 10.1, None,
                                     float("nan"), float("inf"), -float("inf")])
        invalid.append(FINAL[:-1] + ', "overall_score": 1e400}')
        for value in invalid:
            with self.subTest(value=value), self.assertRaises(judge.JudgeError):
                judge.parse_scores(value)

    def test_sse_deltas_and_terminal_usage(self):
        stream = (b": keepalive\r\n\r\n"
                  + event("response.output_text.delta", delta=FINAL[:12])
                  + event("response.output_text.delta", delta=FINAL[12:])
                  + completed(usage={"input_tokens": 15, "output_tokens": 8,
                                     "total_tokens": 23, "private": "omit"}))
        state = judge.StreamState()
        judge.read_sse(io.BytesIO(stream), MODEL, state)
        self.assertEqual(state.raw_final_text, FINAL)
        self.assertEqual(state.returned_model, MODEL)
        self.assertEqual(state.usage, {"input_tokens": 15, "output_tokens": 8, "total_tokens": 23})

    def test_multiline_data_and_unicode(self):
        text = json.dumps({**SCORES, "comments": "Café 空間"}, ensure_ascii=False)
        payload = json.dumps({"type": "response.output_text.delta", "delta": text},
                             ensure_ascii=False).replace(', "delta"', ',\ndata: "delta"')
        stream = ("data: " + payload + "\n\n").encode() + completed(text)
        state = judge.StreamState()
        judge.read_sse(io.BytesIO(stream), MODEL, state)
        self.assertEqual(state.raw_final_text, text)

    def test_completion_is_mandatory_and_status_checked(self):
        for ending in [b"", b"data: [DONE]\n\n", completed(status="incomplete"),
                       event("response.failed", response={"error": "private"}),
                       event("response.incomplete", response={}), event("error", message="secret")]:
            state = judge.StreamState()
            with self.subTest(ending=ending), self.assertRaises(judge.JudgeError):
                judge.read_sse(io.BytesIO(event("response.output_text.delta", delta=FINAL)
                                          + ending), MODEL, state)
            self.assertEqual(state.raw_final_text, FINAL)

    def test_model_mismatch_retains_evidence(self):
        state = judge.StreamState()
        with self.assertRaisesRegex(judge.JudgeError, "model_mismatch"):
            judge.read_sse(io.BytesIO(completed(model="another-model")), MODEL, state)
        self.assertEqual(state.returned_model, "another-model")
        self.assertEqual(state.raw_final_text, FINAL)

    def test_rejects_missing_model_refusal_and_inconsistent_text(self):
        refusal = event("response.completed", response={"model": MODEL, "status": "completed",
                        "output": [{"type": "message", "role": "assistant",
                                    "content": [{"type": "refusal", "refusal": "No"}]}]})
        for stream in [completed(model=None), refusal,
                       event("response.output_text.delta", delta="different") + completed(),
                       b"data: invalid\n\n", b"data: \xff\n\n",
                       b"event: error\ndata: {\"type\":\"response.completed\"}\n\n"]:
            with self.subTest(stream=stream), self.assertRaises(judge.JudgeError):
                judge.read_sse(io.BytesIO(stream), MODEL, judge.StreamState())

    def test_stream_size_is_bounded(self):
        with mock.patch.object(judge, "MAX_STREAM_BYTES", 20):
            with self.assertRaisesRegex(judge.JudgeError, "stream_too_large"):
                judge.read_sse(io.BytesIO(completed()), MODEL, judge.StreamState())

    def test_failed_terminal_preserves_final_evidence(self):
        stream = event("response.incomplete", response={"model": MODEL, "status": "incomplete",
                       "usage": {"output_tokens": 4}, "output": [{"type": "message",
                       "role": "assistant", "content": [{"type": "output_text", "text": "partial"}]}]})
        state = judge.StreamState()
        with self.assertRaises(judge.JudgeError):
            judge.read_sse(io.BytesIO(stream), MODEL, state)
        self.assertEqual(state.raw_final_text, "partial")
        self.assertEqual(state.returned_model, MODEL)
        self.assertEqual(state.usage, {"output_tokens": 4})

    def test_earlier_response_model_mismatch_cannot_be_hidden(self):
        stream = event("response.created", response={"model": "wrong", "status": "in_progress"}) + completed()
        with self.assertRaisesRegex(judge.JudgeError, "model_mismatch"):
            judge.read_sse(io.BytesIO(stream), MODEL, judge.StreamState())


class EvaluatorTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.results = self.root / "results"
        self.prompt = self.root / "prompt.txt"
        self.prompt.write_text("Assess only the supplied scene renders. Return JSON.", encoding="utf-8")
        self.png = self.root / "view.png"
        self.png.write_bytes(PNG)
        self.case = {"id": "../../unsafe/scene α", "prompt_file": "prompt.txt",
                     "prompt_sha256": digest(self.prompt.read_bytes()),
                     "images": [{"path": "view.png", "sha256": digest(PNG)}]}
        self.manifest = self.root / "manifest.json"
        self.write_manifest([self.case])
        self.secret = "unit-test-secret-never-persist"
        self.env = self.root / "api.env"
        self.env.write_text("OPENAI_API_KEY='" + self.secret + "'\n"
                            "OPENAI_BASE_URL=https://judge.invalid/codex/v1\n"
                            "OPENAI_MODEL=" + MODEL + "\n", encoding="utf-8")
        self.env.chmod(0o600)
        # Fail immediately if any test accidentally uses the live transport.
        patcher = mock.patch.object(judge, "open_response", side_effect=AssertionError("network forbidden"))
        self.transport = patcher.start()
        self.addCleanup(patcher.stop)

    def write_manifest(self, cases, **extra):
        self.manifest.write_text(json.dumps({"schema_version": 1, "expected_model": MODEL,
                                             "cases": cases, **extra}), encoding="utf-8")

    def run_judge(self, payload=None, **kwargs):
        self.transport.side_effect = None
        self.transport.return_value = Response(completed() if payload is None else payload)
        return judge.evaluate(self.manifest, self.results, self.env, **kwargs)

    def result(self):
        return json.loads(next(self.results.glob("*.result.json")).read_text())

    def test_payload_and_success_are_auditable(self):
        records = self.run_judge()
        self.assertEqual(records[0]["status"], "completed")
        request, timeout = self.transport.call_args.args
        body = json.loads(request.data)
        self.assertEqual(timeout, 90)
        self.assertEqual(request.full_url, "https://judge.invalid/codex/v1/responses")
        self.assertEqual(request.method, "POST")
        self.assertFalse(body["store"])
        self.assertTrue(body["stream"])
        self.assertEqual(body["model"], MODEL)
        parts = body["input"][0]["content"]
        self.assertEqual(parts[0], {"type": "input_text", "text": self.prompt.read_text()})
        self.assertEqual(parts[1]["type"], "input_image")
        self.assertEqual(base64.b64decode(parts[1]["image_url"].split(",", 1)[1]), PNG)
        record = self.result()
        self.assertEqual(record["scores"], SCORES)
        self.assertEqual(record["raw_final_text"], FINAL)
        self.assertEqual(record["returned_model"], MODEL)
        self.assertEqual(len(record["request_sha256"]), 64)
        self.assertEqual(len(record["content_sha256"]), 64)
        self.assertEqual(len(list(self.results.iterdir())), 2)
        for path in self.results.iterdir():
            self.assertEqual(path.parent, self.results)
            self.assertNotIn(self.secret, path.read_text())

    def test_matching_completion_reused_without_mutation(self):
        self.run_judge()
        before = {p.name: p.read_bytes() for p in self.results.iterdir()}
        records = judge.evaluate(self.manifest, self.results, self.env)
        self.assertEqual(records[0]["status"], "completed")
        self.assertEqual(self.transport.call_count, 1)
        self.assertEqual(before, {p.name: p.read_bytes() for p in self.results.iterdir()})

    def test_fractional_overall_stream_completes_and_is_reused(self):
        expected = {**SCORES, "aesthetics_score": 7, "overall_score": 22 / 3}
        text = json.dumps(expected)
        stream = (event("response.output_text.delta", delta=text[:20])
                  + event("response.output_text.delta", delta=text[20:])
                  + completed(text))
        records = self.run_judge(stream)
        self.assertEqual(records[0]["status"], "completed")
        self.assertEqual(self.result()["scores"], expected)
        before = {p.name: p.read_bytes() for p in self.results.iterdir()}
        reused = judge.evaluate(self.manifest, self.results, self.env)
        self.assertEqual(reused[0]["scores"], expected)
        self.assertEqual(self.transport.call_count, 1)
        self.assertEqual(before, {p.name: p.read_bytes() for p in self.results.iterdir()})

    def test_changed_prompt_with_correct_hash_refuses_reuse(self):
        self.run_judge()
        self.prompt.write_text("Different rubric", encoding="utf-8")
        self.write_manifest([{**self.case, "prompt_sha256": digest(self.prompt.read_bytes())}])
        with self.assertRaisesRegex(judge.JudgeError, "incompatible_reuse"):
            judge.evaluate(self.manifest, self.results, self.env)
        self.assertEqual(self.transport.call_count, 1)

    def test_changed_image_with_correct_hash_refuses_reuse(self):
        self.run_judge()
        changed = (PNG[:33] + png_chunk(b"IDAT", zlib.compress(b"\x00\x00\x00\x00"))
                   + png_chunk(b"IEND", b""))
        self.png.write_bytes(changed)
        self.write_manifest([{**self.case, "images": [{"path": "view.png", "sha256": digest(changed)}]}])
        with self.assertRaisesRegex(judge.JudgeError, "incompatible_reuse"):
            judge.evaluate(self.manifest, self.results, self.env)
        self.assertEqual(self.transport.call_count, 1)

    def test_changed_endpoint_or_instructions_refuses_reuse(self):
        self.run_judge()
        with mock.patch.object(judge, "INSTRUCTIONS", "Changed instructions"):
            with self.assertRaisesRegex(judge.JudgeError, "incompatible_reuse"):
                judge.evaluate(self.manifest, self.results, self.env)
        self.env.write_text(self.env.read_text().replace("judge.invalid", "other.invalid"))
        with self.assertRaisesRegex(judge.JudgeError, "incompatible_reuse"):
            judge.evaluate(self.manifest, self.results, self.env)
        self.assertEqual(self.transport.call_count, 1)

    def test_invalid_artifacts_never_call_transport(self):
        for case in [{**self.case, "prompt_sha256": "0" * 64},
                     {**self.case, "images": [{"path": "view.png", "sha256": "0" * 64}]},
                     {**self.case, "prompt_file": "missing.txt"}]:
            self.write_manifest([case])
            with self.subTest(case=case), self.assertRaises(judge.JudgeError):
                judge.evaluate(self.manifest, self.results, self.env)
        self.png.write_bytes(b"not PNG")
        self.write_manifest([{**self.case, "images": [{"path": "view.png", "sha256": digest(b"not PNG")}]}])
        with self.assertRaisesRegex(judge.JudgeError, "invalid_png"):
            judge.evaluate(self.manifest, self.results, self.env)
        self.transport.assert_not_called()

    def test_manifest_validation(self):
        for cases in [[self.case, self.case], [{**self.case, "images": []}],
                      [{**self.case, "id": ""}], [{**self.case, "prompt_sha256": "no"}]]:
            self.write_manifest(cases)
            with self.subTest(cases=cases), self.assertRaises(judge.JudgeError):
                judge.evaluate(self.manifest, self.results, self.env)
        self.write_manifest([self.case], expected_model="other")
        with self.assertRaisesRegex(judge.JudgeError, "configured_model_mismatch"):
            judge.evaluate(self.manifest, self.results, self.env)
        self.transport.assert_not_called()

    def test_limit_default_one_and_explicit_bounds(self):
        self.write_manifest([self.case, {**self.case, "id": "second"}])
        self.assertEqual(len(self.run_judge()), 1)
        self.assertEqual(self.transport.call_count, 1)
        self.assertEqual(len(judge.evaluate(self.manifest, self.results, self.env, limit=1)), 1)
        for limit, timeout in [(0, 90), (-1, 90), (1, 0), (1, float("nan")), (True, 90)]:
            with self.subTest(limit=limit, timeout=timeout), self.assertRaises(judge.JudgeError):
                judge.evaluate(self.manifest, self.results, self.env, limit=limit, timeout=timeout)

    def test_limit_extension_reuses_first_and_evaluates_next(self):
        self.write_manifest([self.case, {**self.case, "id": "second"}])
        self.run_judge()
        self.transport.return_value = Response(completed())
        results = judge.evaluate(self.manifest, self.results, self.env, limit=2)
        self.assertEqual(len(results), 2)
        self.assertEqual(self.transport.call_count, 2)

    def test_all_selected_artifacts_preflight_before_first_request(self):
        self.write_manifest([self.case, {**self.case, "id": "second", "prompt_sha256": "0" * 64}])
        with self.assertRaisesRegex(judge.JudgeError, "artifact_hash_mismatch"):
            judge.evaluate(self.manifest, self.results, self.env, limit=2)
        self.transport.assert_not_called()

    def test_png_checksum_and_truncation_fail_before_request(self):
        for payload in [PNG[:-1], PNG[:40] + b"x" + PNG[41:], PNG + b"trailing"]:
            self.png.write_bytes(payload)
            self.write_manifest([{**self.case, "images": [{"path": "view.png", "sha256": digest(payload)}]}])
            with self.assertRaisesRegex(judge.JudgeError, "invalid_png"):
                judge.evaluate(self.manifest, self.results, self.env)
        self.transport.assert_not_called()

    def test_failure_is_retained_and_never_retried(self):
        self.run_judge(event("response.output_text.delta", delta="partial"))
        result = self.result()
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["raw_final_text"], "partial")
        self.assertNotIn("scores", result)
        with self.assertRaisesRegex(judge.JudgeError, "previous_failure"):
            judge.evaluate(self.manifest, self.results, self.env)
        self.assertEqual(self.transport.call_count, 1)

    def test_model_mismatch_stops_remaining_batch(self):
        self.write_manifest([self.case, {**self.case, "id": "second"}])
        results = self.run_judge(completed(model="wrong"), limit=2)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["error"], "model_mismatch")
        self.assertEqual(self.transport.call_count, 1)
        self.assertEqual(len(list(self.results.glob("*.attempt.json"))), 1)

    def test_reserved_attempt_blocks_after_crash(self):
        self.transport.side_effect = KeyboardInterrupt()
        with self.assertRaises(KeyboardInterrupt):
            judge.evaluate(self.manifest, self.results, self.env)
        self.assertEqual(len(list(self.results.glob("*.attempt.json"))), 1)
        with self.assertRaisesRegex(judge.JudgeError, "unfinished_attempt"):
            judge.evaluate(self.manifest, self.results, self.env)
        self.assertEqual(self.transport.call_count, 1)

    def test_invalid_json_and_model_mismatch_have_no_scores(self):
        for payload in [completed("prose " + FINAL), completed(model="wrong")]:
            with self.subTest(payload=payload):
                self.results = self.root / digest(payload)
                self.run_judge(payload)
                self.assertEqual(self.result()["status"], "failed")
                self.assertNotIn("scores", self.result())

    def test_exception_bodies_and_raw_output_redact_secrets(self):
        self.transport.side_effect = urllib.error.HTTPError(
            "https://judge.invalid/" + self.secret, 429, "Authorization: Bearer " + self.secret,
            {"Authorization": self.secret}, io.BytesIO(self.secret.encode()))
        judge.evaluate(self.manifest, self.results, self.env)
        result = self.result()
        self.assertEqual(result["error"], "http_status_429")
        self.assertNotIn(self.secret, json.dumps(result))
        self.assertNotIn("Authorization", json.dumps(result))
        self.results = self.root / "raw-redaction"
        self.run_judge(completed(json.dumps({**SCORES, "comments": self.secret})))
        self.assertNotIn(self.secret, json.dumps(self.result()))

    def test_redacted_json_remains_valid_and_reusable(self):
        text = json.dumps({**SCORES, "comments": "Authorization: Bearer " + self.secret})
        self.run_judge(completed(text))
        judge.evaluate(self.manifest, self.results, self.env)
        self.assertNotIn(self.secret, json.dumps(self.result()))
        self.assertEqual(self.transport.call_count, 1)

    def test_transport_failure_codes_do_not_echo_exception(self):
        for error, code in [(TimeoutError(self.secret), "timeout"),
                            (urllib.error.URLError(self.secret), "transport_error"),
                            (RuntimeError(self.secret), "unexpected_error")]:
            self.results = self.root / code
            self.transport.side_effect = error
            judge.evaluate(self.manifest, self.results, self.env)
            self.assertEqual(self.result()["error"], code)
            self.assertNotIn(self.secret, json.dumps(self.result()))

    def test_wrapped_socket_timeout_is_classified(self):
        self.transport.side_effect = urllib.error.URLError(TimeoutError(self.secret))
        judge.evaluate(self.manifest, self.results, self.env)
        self.assertEqual(self.result()["error"], "timeout")

    def test_total_deadline_covers_blocked_transport_and_is_restored(self):
        previous = signal.getsignal(signal.SIGALRM)
        self.transport.side_effect = lambda *_: time.sleep(0.5)
        started = time.monotonic()
        judge.evaluate(self.manifest, self.results, self.env, timeout=0.02)
        self.assertLess(time.monotonic() - started, 0.4)
        self.assertEqual(self.result()["error"], "timeout")
        self.assertEqual(signal.getsignal(signal.SIGALRM), previous)
        self.assertEqual(signal.getitimer(signal.ITIMER_REAL), (0.0, 0.0))
        self.assertEqual(self.transport.call_count, 1)

    def test_non_sse_or_non_200_response_is_failure(self):
        for status, content_type in [(200, "application/json"), (201, "text/event-stream")]:
            self.results = self.root / str(status)
            response = Response(completed())
            response.status = status
            response.headers = {"Content-Type": content_type}
            self.transport.side_effect = None
            self.transport.return_value = response
            judge.evaluate(self.manifest, self.results, self.env)
            self.assertEqual(self.result()["status"], "failed")

    def test_config_requires_private_regular_file_and_valid_url(self):
        self.env.chmod(0o644)
        with self.assertRaisesRegex(judge.JudgeError, "config_permissions"):
            judge.load_config(self.env)
        self.env.chmod(0o600)
        original = self.env.read_text()
        for url in ["http://judge.invalid/v1", "https://user:pass@judge.invalid/v1",
                    "https://judge.invalid/v1?key=private", "https://judge.invalid/v1#fragment"]:
            self.env.write_text(original.replace("https://judge.invalid/codex/v1", url))
            with self.subTest(url=url), self.assertRaises(judge.JudgeError):
                judge.load_config(self.env)
        self.env.write_text(original)
        self.assertNotIn(self.secret, repr(judge.load_config(self.env)))

    def test_config_export_quotes_comments_and_duplicate_rejection(self):
        self.env.write_text("# local test configuration\nexport OPENAI_MODEL='" + MODEL + "'\n"
                            "OPENAI_BASE_URL=https://judge.invalid/v1 # test endpoint\n"
                            "OPENAI_API_KEY='" + self.secret + "'\nIGNORED=irrelevant\n")
        self.assertEqual(judge.load_config(self.env).model, MODEL)
        self.env.write_text(self.env.read_text() + "OPENAI_MODEL=duplicate\n")
        with self.assertRaisesRegex(judge.JudgeError, "invalid_config"):
            judge.load_config(self.env)

    def test_config_symlink_rejected(self):
        alias = self.root / "alias.env"
        alias.symlink_to(self.env)
        with self.assertRaises(judge.JudgeError):
            judge.load_config(alias)

    def test_fifo_artifact_rejected_without_blocking(self):
        pipe = self.root / "pipe"
        os.mkfifo(pipe)
        started = time.monotonic()
        with judge.deadline(0.5):
            with self.assertRaisesRegex(judge.JudgeError, "artifact_read_error"):
                judge.bounded_read(pipe, 100, "artifact_read_error")
        self.assertLess(time.monotonic() - started, 0.3)

    def test_reused_scores_require_integer_types(self):
        self.run_judge()
        path = next(self.results.glob("*.result.json"))
        result = self.result()
        result["scores"]["layout_score"] = 8.0
        path.write_text(json.dumps(result))
        with self.assertRaisesRegex(judge.JudgeError, "invalid_record"):
            judge.evaluate(self.manifest, self.results, self.env)
        self.assertEqual(self.transport.call_count, 1)

    def test_exclusive_result_write_and_corrupt_reuse_fail_closed(self):
        path = self.root / "existing.json"
        judge.write_immutable(path, {"original": True})
        before = path.read_bytes()
        with self.assertRaises(judge.JudgeError):
            judge.write_immutable(path, {"original": False})
        self.assertEqual(path.read_bytes(), before)
        self.run_judge()
        next(self.results.glob("*.result.json")).write_text("{")
        with self.assertRaisesRegex(judge.JudgeError, "invalid_record"):
            judge.evaluate(self.manifest, self.results, self.env)
        self.assertEqual(self.transport.call_count, 1)

    def test_cli_outputs_only_summary_and_error_codes(self):
        stdout, stderr = io.StringIO(), io.StringIO()
        self.transport.side_effect = None
        self.transport.return_value = Response(completed())
        argv = ["--manifest", str(self.manifest), "--results-dir", str(self.results),
                "--env-file", str(self.env)]
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            self.assertEqual(judge.main(argv), 0)
        self.assertNotIn(self.secret, stdout.getvalue() + stderr.getvalue())
        self.assertNotIn("Bearer", stdout.getvalue())
        self.write_manifest([self.case], expected_model=self.secret)
        with contextlib.redirect_stderr(stderr):
            self.assertEqual(judge.main(argv), 1)
        self.assertNotIn(self.secret, stderr.getvalue())


class TransportTests(unittest.TestCase):
    def test_no_redirect_or_proxy_and_exactly_one_open(self):
        request = judge.urllib.request.Request("https://judge.invalid/responses", data=b"{}")
        with mock.patch.object(judge.urllib.request, "build_opener") as build:
            build.return_value.open.side_effect = TimeoutError()
            with self.assertRaises(TimeoutError):
                judge.open_response(request, 90)
        build.return_value.open.assert_called_once_with(request, timeout=90)
        proxy, redirects = build.call_args.args
        self.assertEqual(proxy.proxies, {})
        for code in [301, 302, 303, 307, 308]:
            with self.assertRaisesRegex(judge.JudgeError, "redirect_refused"):
                redirects.redirect_request(request, None, code, "moved", {}, "https://other.invalid")


if __name__ == "__main__":
    unittest.main()
