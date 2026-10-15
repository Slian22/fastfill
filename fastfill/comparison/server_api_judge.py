#!/usr/bin/env python3
"""Bounded, standalone, stdlib-only vision evaluation of existing scene PNGs.

Run ON THE SERVER that owns the private config (never copy its API secret):
  python3 server_api_judge.py --manifest scenes.json --results-dir results \
      --env-file ~/.fastfill_api.env --limit 1 --timeout 90

Manifest schema (paths relative to the manifest, or absolute):
  {"schema_version": 1, "expected_model": "gpt-6.1-sol", "cases": [
    {"id": "scene-001", "prompt_file": "rubric-and-scene.txt",
     "prompt_sha256": "<64 lowercase hexadecimal characters>", "images": [
       {"path": "renders/scene-001.png", "sha256": "<64 lowercase hex>"}]}]}

The UTF-8 prompt must contain the caller's rubric and scene description. This
script supplies no geometry-based scores or substitute rubric. The private,
regular, owner-only mode-0600 env file supplies OPENAI_API_KEY, OPENAI_BASE_URL,
and OPENAI_MODEL; shell expansion/execution and ambient env overrides are absent.

--limit bounds the FIRST N manifest cases examined, INCLUDING reused cases.
Increase it to extend a run. All selected artifacts are checked before any API
request. A case ID's SHA-256 names its immutable .attempt.json and .result.json.
Keep the results directory across invocations: a reservation prevents retries
after errors, crashes, concurrent runs, and interrupted result writes. Never
delete reservations to resume a run. Matching completed results are reused;
failed, unfinished, corrupt, or incompatible records stop the run explicitly.
Any new API/evaluation failure also stops the batch before further requests.

Amendment 023 options (absent = the pinned run-002 client's exact payload):
--model replaces the env file's OPENAI_MODEL (key and base URL still come from
the env file); --reasoning-effort adds {"reasoning": {"effort": ...}} and
raises max_output_tokens to 32768 because reasoning tokens count against it.

Amendment 025: Config.instructions/Config.parse (evaluate's instructions/parse)
let critic_client_025 reuse this client with its own system prompt and output
parser; the parsed object is stored under "scores". Defaults change nothing.

Content hashes cover scene ID and ordered verified artifact bytes (via SHA-256).
Request hashes additionally cover endpoint, exact model, instructions and full
payload, including image data URLs. Credentials, artifact paths, and timeout are
not part of request identity. Result files retain model, minimal usage, final or
partial output text, scores on success, and a fixed error code on failure. Known
credentials are redacted from stored text; HTTP bodies/headers and exception
messages are never stored or printed. Stdout contains counts only; exit status
is nonzero on any failure. POSIX main-thread execution enforces a wall deadline.
"""

import argparse
import base64
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shlex
import signal
import stat
import struct
import sys
import threading
import urllib.error
import urllib.parse
import urllib.request
import zlib


SCHEMA_VERSION = 1
MAX_MANIFEST_BYTES = 4 * 1024 * 1024
MAX_PROMPT_BYTES = 1024 * 1024
MAX_IMAGE_BYTES = 32 * 1024 * 1024
MAX_REQUEST_BYTES = 64 * 1024 * 1024
MAX_STREAM_BYTES = 4 * 1024 * 1024
MAX_IMAGES = 16
INSTRUCTIONS = (
    "Apply the supplied prompt's rubric to the supplied scene renders. "
    "Return only one JSON object: functionality_score, layout_score, and "
    "aesthetics_score must each be integers from 0 to 10. Optional overall_score "
    "must be a finite number from 0 to 10 (a fractional average is allowed); "
    "optional comments must be a string. "
    "Do not invent evidence that is absent from the supplied renders."
)
HASH_PATTERN = re.compile(r"[0-9a-f]{64}\Z")


class JudgeError(Exception):
    """Only fixed, non-sensitive error codes may be passed to this exception."""


@dataclass(frozen=True)
class Config:
    api_key: str = field(repr=False)
    base_url: str
    model: str
    reasoning_effort: str = ""
    instructions: str = INSTRUCTIONS
    parse: object = None  # None = parse_scores


@dataclass
class StreamState:
    raw_final_text: str = ""
    returned_model: str = ""
    usage: dict = field(default_factory=dict)


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")


def sha256(value):
    return hashlib.sha256(value).hexdigest()


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def invalid_constant(_value):
    raise ValueError("non-finite JSON value")


def strict_json(text, error_code):
    try:
        return json.loads(text, object_pairs_hook=unique_object, parse_constant=invalid_constant)
    except (ValueError, TypeError, RecursionError, UnicodeError):
        raise JudgeError(error_code) from None


def bounded_read(path, maximum, code):
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_NONBLOCK)
        with os.fdopen(descriptor, "rb") as handle:
            if not stat.S_ISREG(os.fstat(handle.fileno()).st_mode):
                raise JudgeError(code)
            data = handle.read(maximum + 1)
    except (OSError, ValueError):
        raise JudgeError(code) from None
    if len(data) > maximum:
        raise JudgeError(code)
    return data


def validate_base_url(value):
    try:
        parsed = urllib.parse.urlsplit(value)
        if (parsed.scheme != "https" or not parsed.hostname or parsed.username is not None
                or parsed.password is not None or parsed.query or parsed.fragment
                or any(c.isspace() or ord(c) < 32 for c in value)
                or (parsed.port is not None and not 1 <= parsed.port <= 65535)):
            raise ValueError()
    except ValueError:
        raise JudgeError("invalid_base_url") from None
    return value.rstrip("/")


def load_config(path):
    try:
        descriptor = os.open(Path(path).expanduser(), os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        with os.fdopen(descriptor, "r", encoding="utf-8") as handle:
            info = os.fstat(handle.fileno())
            if (not stat.S_ISREG(info.st_mode) or stat.S_IMODE(info.st_mode) != 0o600
                    or info.st_uid != os.getuid()):
                raise JudgeError("config_permissions")
            text = handle.read(65537)
        if len(text) > 65536:
            raise JudgeError("invalid_config")
    except (OSError, UnicodeError, ValueError):
        raise JudgeError("config_read_error") from None
    values = {}
    required = {"OPENAI_API_KEY", "OPENAI_BASE_URL", "OPENAI_MODEL"}
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("export "):
            line = line[7:].lstrip()
        if not line or line.startswith("#"):
            continue
        name, separator, value = line.partition("=")
        name = name.strip()
        if name not in required:
            continue
        try:
            parts = shlex.split(value, comments=False, posix=True)
        except ValueError:
            raise JudgeError("invalid_config") from None
        if (not separator or name in values or not parts or not parts[0]
                or (len(parts) > 1 and not parts[1].startswith("#"))):
            raise JudgeError("invalid_config")
        values[name] = parts[0]
    if set(values) != required or any(any(ord(c) < 33 or ord(c) > 126 for c in v)
                                      for v in values.values()):
        raise JudgeError("invalid_config")
    return Config(values["OPENAI_API_KEY"], validate_base_url(values["OPENAI_BASE_URL"]),
                  values["OPENAI_MODEL"])


def valid_hash(value):
    return isinstance(value, str) and HASH_PATTERN.fullmatch(value) is not None


def valid_path(value):
    return isinstance(value, str) and bool(value) and "\x00" not in value


def validate_case(case):
    if (not isinstance(case, dict) or not isinstance(case.get("id"), str)
            or not case["id"].strip() or len(case["id"]) > 1024
            or not valid_path(case.get("prompt_file")) or not valid_hash(case.get("prompt_sha256"))
            or not isinstance(case.get("images"), list) or not 1 <= len(case["images"]) <= MAX_IMAGES):
        raise JudgeError("invalid_case")
    for item in case["images"]:
        if (not isinstance(item, dict) or not valid_path(item.get("path"))
                or not valid_hash(item.get("sha256"))):
            raise JudgeError("invalid_image_entry")


def load_manifest(path):
    manifest = strict_json(bounded_read(path, MAX_MANIFEST_BYTES, "manifest_read_error"), "invalid_manifest")
    if (not isinstance(manifest, dict) or type(manifest.get("schema_version")) is not int
            or manifest["schema_version"] != SCHEMA_VERSION
            or not isinstance(manifest.get("expected_model"), str) or not manifest["expected_model"]
            or not isinstance(manifest.get("cases"), list) or not 1 <= len(manifest["cases"]) <= 10000):
        raise JudgeError("invalid_manifest")
    for case in manifest["cases"]:
        validate_case(case)
    if len({case["id"] for case in manifest["cases"]}) != len(manifest["cases"]):
        raise JudgeError("duplicate_case_id")
    return manifest


def verified_artifact(root, path, expected_hash, maximum):
    data = bounded_read(root / path, maximum, "artifact_read_error")
    if sha256(data) != expected_hash:
        raise JudgeError("artifact_hash_mismatch")
    return data


def verify_png(data):
    """Validate signature, framing and CRCs without image dependencies or decoding."""
    if not data.startswith(b"\x89PNG\r\n\x1a\n"):
        raise JudgeError("invalid_png")
    offset, seen_idat = 8, False
    while offset + 12 <= len(data):
        size = struct.unpack_from(">I", data, offset)[0]
        end = offset + 12 + size
        if end > len(data):
            break
        kind = data[offset + 4:offset + 8]
        chunk = data[offset + 8:end - 4]
        crc = struct.unpack_from(">I", data, end - 4)[0]
        if zlib.crc32(kind + chunk) != crc:
            break
        if offset == 8:
            if kind != b"IHDR" or size != 13 or not all(struct.unpack_from(">II", chunk)):
                break
        elif kind == b"IHDR":
            break
        seen_idat = seen_idat or kind == b"IDAT"
        if kind == b"IEND":
            if size == 0 and end == len(data) and seen_idat:
                return
            break
        offset = end
    raise JudgeError("invalid_png")


def prepare_case(case, root, config):
    prompt = verified_artifact(root, case["prompt_file"], case["prompt_sha256"], MAX_PROMPT_BYTES)
    try:
        prompt_text = prompt.decode("utf-8")
    except UnicodeError:
        raise JudgeError("invalid_prompt_utf8") from None
    if not prompt_text.strip():
        raise JudgeError("empty_prompt")
    content = [{"type": "input_text", "text": prompt_text}]
    total_bytes = len(prompt)
    for item in case["images"]:
        data = verified_artifact(root, item["path"], item["sha256"], MAX_IMAGE_BYTES)
        verify_png(data)
        total_bytes += 4 * ((len(data) + 2) // 3)
        if total_bytes > MAX_REQUEST_BYTES:
            raise JudgeError("request_too_large")
        content.append({"type": "input_image", "image_url": "data:image/png;base64,"
                        + base64.b64encode(data).decode("ascii")})
    payload = {"model": config.model, "input": [{"role": "user", "content": content}],
               "instructions": config.instructions, "store": False, "stream": True, "max_output_tokens": 4096}
    if config.reasoning_effort:
        payload["reasoning"] = {"effort": config.reasoning_effort}
        payload["max_output_tokens"] = 32768
    body = canonical(payload)
    if len(body) > MAX_REQUEST_BYTES:
        raise JudgeError("request_too_large")
    endpoint = config.base_url + "/responses"
    identity = {"schema_version": SCHEMA_VERSION, "id": case["id"], "expected_model": config.model,
                "content_sha256": sha256(canonical({"id": case["id"], "prompt_sha256": case["prompt_sha256"],
                    "image_sha256": [item["sha256"] for item in case["images"]]})),
                "request_sha256": sha256(canonical({"schema_version": SCHEMA_VERSION,
                    "endpoint": endpoint, "payload": payload}))}
    return identity, body


def parse_scores(text):
    scores = strict_json(text, "invalid_scores_json")
    required = {"functionality_score", "layout_score", "aesthetics_score"}
    if (not isinstance(scores, dict) or not required <= scores.keys()
            or not scores.keys() <= required | {"overall_score", "comments"}):
        raise JudgeError("invalid_score_fields")
    for key in required:
        if type(scores[key]) is not int or not 0 <= scores[key] <= 10:
            raise JudgeError("invalid_score_value")
    if "overall_score" in scores:
        overall = scores["overall_score"]
        if type(overall) not in (int, float) or not 0 <= overall <= 10:
            raise JudgeError("invalid_score_value")
    if "comments" in scores and not isinstance(scores["comments"], str):
        raise JudgeError("invalid_comments")
    return scores


def completed_text(response):
    output = response.get("output")
    if not isinstance(output, list):
        raise JudgeError("missing_final_output")
    texts = []
    for item in output:
        if not isinstance(item, dict):
            raise JudgeError("invalid_final_output")
        if item.get("type") != "message":
            continue
        if item.get("role") != "assistant" or not isinstance(item.get("content"), list):
            raise JudgeError("invalid_final_output")
        for part in item["content"]:
            if not isinstance(part, dict) or part.get("type") != "output_text" or not isinstance(part.get("text"), str):
                raise JudgeError("invalid_final_output")
            texts.append(part["text"])
    if not texts:
        raise JudgeError("missing_final_output")
    return "".join(texts)


def capture_metadata(response, state):
    if "model" in response:
        state.returned_model = response["model"] if isinstance(response["model"], str) else ""
    usage = response.get("usage")
    if isinstance(usage, dict):
        state.usage = {key: usage[key] for key in ("input_tokens", "output_tokens", "total_tokens")
                       if type(usage.get(key)) is int and usage[key] >= 0}
        details = usage.get("output_tokens_details")
        if isinstance(details, dict) and type(details.get("reasoning_tokens")) is int:
            state.usage["reasoning_tokens"] = details["reasoning_tokens"]


def handle_event(event_name, data, expected_model, state):
    if data == "[DONE]":
        raise JudgeError("missing_completed_event")
    message = strict_json(data, "invalid_sse_json")
    if not isinstance(message, dict) or not isinstance(message.get("type"), str):
        raise JudgeError("invalid_sse_event")
    kind = message["type"]
    if event_name and event_name != kind:
        raise JudgeError("sse_event_type_mismatch")
    response = message.get("response")
    if isinstance(response, dict):
        capture_metadata(response, state)
        if kind != "response.completed" and "model" in response and response["model"] != expected_model:
            raise JudgeError("model_mismatch")
    if kind in {"error", "response.failed", "response.incomplete"}:
        if isinstance(response, dict) and response.get("output"):
            state.raw_final_text = completed_text(response)
        raise JudgeError("remote_response_failed")
    if kind == "response.output_text.delta":
        if not isinstance(message.get("delta"), str):
            raise JudgeError("invalid_text_delta")
        state.raw_final_text += message["delta"]
    if kind != "response.completed":
        return False
    if not isinstance(response, dict):
        raise JudgeError("invalid_completed_event")
    model = response.get("model")
    final = completed_text(response)
    delta_text = state.raw_final_text
    state.raw_final_text = final
    if response.get("status") != "completed":
        raise JudgeError("response_not_completed")
    if model != expected_model:
        raise JudgeError("model_mismatch")
    if delta_text and delta_text != final:
        raise JudgeError("delta_final_mismatch")
    return True


def read_sse(stream, expected_model, state):
    """Read bounded UTF-8 SSE events, ending only at a valid completed event."""
    total, event_name, lines = 0, "", []
    while True:
        raw = stream.readline(MAX_STREAM_BYTES - total + 1)
        total += len(raw)
        if total > MAX_STREAM_BYTES:
            raise JudgeError("stream_too_large")
        if not raw:
            raise JudgeError("missing_completed_event")
        try:
            line = raw.decode("utf-8").rstrip("\r\n")
        except UnicodeError:
            raise JudgeError("invalid_sse_utf8") from None
        if not line:
            if lines and handle_event(event_name, "\n".join(lines), expected_model, state):
                return
            event_name, lines = "", []
        elif not line.startswith(":"):
            name, separator, value = line.partition(":")
            if value.startswith(" "):
                value = value[1:]
            if name == "event":
                event_name = value
            elif name == "data":
                lines.append(value if separator else "")


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise JudgeError("redirect_refused")


def open_response(request, timeout):
    """Exactly one POST; no retries, redirects, or ambient proxy credentials."""
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())
    return opener.open(request, timeout=timeout)


@contextmanager
def deadline(seconds):
    if threading.current_thread() is not threading.main_thread() or not hasattr(signal, "setitimer"):
        raise JudgeError("deadline_unavailable")
    if signal.getitimer(signal.ITIMER_REAL) != (0.0, 0.0):
        raise JudgeError("deadline_already_active")
    previous = signal.getsignal(signal.SIGALRM)

    def expire(_signum, _frame):
        raise TimeoutError()

    signal.signal(signal.SIGALRM, expire)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def redact(value, secret):
    if isinstance(value, str):
        text = value
        for variant in sorted({secret, json.dumps(secret)[1:-1]}, key=len, reverse=True):
            text = text.replace(variant, "[REDACTED]")
        return re.sub(r"(?i)(\bbearer\s+|\bOPENAI_API_KEY\s*=\s*)[A-Za-z0-9._~+/=-]+",
                      r"\1[REDACTED]", text)
    if isinstance(value, dict):
        return {key: redact(item, secret) for key, item in value.items()}
    if isinstance(value, list):
        return [redact(item, secret) for item in value]
    return value


def write_immutable(path, value):
    data = canonical(value) + b"\n"
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        directory = os.open(Path(path).parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except FileExistsError:
        raise JudgeError("record_already_exists") from None
    except OSError:
        raise JudgeError("record_write_error") from None


def read_record(path, identity):
    record = strict_json(bounded_read(path, MAX_STREAM_BYTES * 3, "invalid_record"), "invalid_record")
    if not isinstance(record, dict):
        raise JudgeError("invalid_record")
    if any(record.get(key) != value for key, value in identity.items()):
        raise JudgeError("incompatible_reuse")
    return record


def reuse_result(attempt_path, result_path, identity, parse=None):
    parse = parse or parse_scores
    if not os.path.lexists(attempt_path):
        if os.path.lexists(result_path):
            raise JudgeError("missing_attempt_record")
        return None
    attempt = read_record(attempt_path, identity)
    if attempt.get("status") != "attempted":
        raise JudgeError("invalid_record")
    if not os.path.lexists(result_path):
        raise JudgeError("unfinished_attempt")
    result = read_record(result_path, identity)
    if result.get("status") == "failed":
        raise JudgeError("previous_failure")
    if (result.get("status") != "completed" or result.get("returned_model") != identity["expected_model"]
            or "error" in result or not isinstance(result.get("raw_final_text"), str)):
        raise JudgeError("invalid_record")
    try:
        if parse(result["raw_final_text"]) != parse(canonical(result.get("scores"))):
            raise JudgeError("invalid_record")
    except (JudgeError, ValueError):
        raise JudgeError("invalid_record") from None
    return result


def call_judge(identity, body, config, timeout):
    state, scores, error = StreamState(), None, None
    try:
        request = urllib.request.Request(config.base_url + "/responses", data=body, method="POST",
            headers={"Authorization": "Bearer " + config.api_key,
                     "Content-Type": "application/json", "Accept": "text/event-stream",
                     "User-Agent": "fastfill-evaluation/1.0"})
        with deadline(timeout), open_response(request, timeout) as response:
            if response.status != 200:
                raise JudgeError("unexpected_http_status")
            if response.headers.get("Content-Type", "").split(";", 1)[0].strip().lower() != "text/event-stream":
                raise JudgeError("unexpected_content_type")
            read_sse(response, config.model, state)
            scores = (config.parse or parse_scores)(state.raw_final_text)
    except JudgeError as exc:
        error = str(exc)
    except urllib.error.HTTPError as exc:
        error = "http_status_" + str(exc.code) if type(exc.code) is int else "http_error"
        exc.close()
    except TimeoutError:
        error = "timeout"
    except urllib.error.URLError as exc:
        error = "timeout" if isinstance(exc.reason, TimeoutError) else "transport_error"
    except OSError:
        error = "transport_error"
    except Exception:
        error = "unexpected_error"
    result = {**identity, "status": "failed" if error else "completed", "finished_at": utc_now(),
              "returned_model": state.returned_model, "usage": state.usage,
              "raw_final_text": state.raw_final_text}
    result = {**result, "error": error} if error else {**result, "scores": scores}
    return redact(result, config.api_key)


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def evaluate(manifest_path, results_dir, env_file, limit=1, timeout=90, model="", reasoning_effort="",
             instructions=INSTRUCTIONS, parse=None):
    if (type(limit) is not int or limit < 1 or type(timeout) not in (int, float)
            or not math.isfinite(timeout) or timeout <= 0):
        raise JudgeError("invalid_bounds")
    config = load_config(env_file)
    if reasoning_effort not in ("", "low", "medium", "high"):
        raise JudgeError("invalid_reasoning_effort")
    config = replace(config, model=model or config.model, reasoning_effort=reasoning_effort,
                     instructions=instructions, parse=parse)
    manifest_path = Path(manifest_path).expanduser().resolve()
    manifest = load_manifest(manifest_path)
    if manifest["expected_model"] != config.model:
        raise JudgeError("configured_model_mismatch")
    selected = manifest["cases"][:limit]
    # Preflight every selected case; retain only hashes, not N large payloads.
    identities = [prepare_case(case, manifest_path.parent, config)[0] for case in selected]
    results_dir = Path(results_dir).expanduser()
    try:
        results_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    except OSError:
        raise JudgeError("results_directory_error") from None
    paths = [(results_dir / (sha256(case["id"].encode()) + ".attempt.json"),
              results_dir / (sha256(case["id"].encode()) + ".result.json")) for case in selected]
    reused = [reuse_result(*pair, identity, parse) for pair, identity in zip(paths, identities)]
    records = []
    for case, identity, pair, prior in zip(selected, identities, paths, reused):
        if prior is not None:
            records.append(prior)
            continue
        verified_identity, body = prepare_case(case, manifest_path.parent, config)
        if verified_identity != identity:
            raise JudgeError("artifact_changed_after_preflight")
        attempt_path, result_path = pair
        write_immutable(attempt_path, {**identity, "status": "attempted", "started_at": utc_now(),
                                       "timeout_seconds": timeout})
        result = call_judge(identity, body, config, timeout)
        write_immutable(result_path, result)
        records.append(result)
        if result["status"] != "completed":
            break
    return records


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--results-dir", required=True, type=Path)
    parser.add_argument("--env-file", type=Path, default=Path("~/.fastfill_api.env"))
    parser.add_argument("--limit", type=int, default=1, help="first N cases, including reuse (default: 1)")
    parser.add_argument("--timeout", type=float, default=90, help="wall seconds per request (default: 90)")
    parser.add_argument("--model", default="", help="override the env file's OPENAI_MODEL")
    parser.add_argument("--reasoning-effort", default="", choices=("", "low", "medium", "high"))
    args = parser.parse_args(argv)
    try:
        records = evaluate(args.manifest, args.results_dir, args.env_file, args.limit, args.timeout,
                           args.model, args.reasoning_effort)
    except JudgeError as exc:
        print("judge_error: " + str(exc), file=sys.stderr)
        return 1
    except Exception:
        print("judge_error: unexpected_error", file=sys.stderr)
        return 1
    failed = sum(record["status"] != "completed" for record in records)
    print(json.dumps({"examined": len(records), "completed": len(records) - failed, "failed": failed}))
    return int(bool(failed))


if __name__ == "__main__":
    sys.exit(main())
