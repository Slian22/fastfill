"""Hash-bound identities and fail-closed completion for full-cohort evaluation.

Legacy metrics files are deliberately insufficient evidence of completion. A
completion marker binds the complete ordered input cohort, model bytes, settings,
evaluation code, predictions, and metrics. This module has no ML dependencies.
"""
from __future__ import annotations

import hashlib
from functools import lru_cache
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping


SPLITS = ("dev_rooms", "dev_constrained_rooms", "test_rooms", "test_constrained_rooms")
SCHEMA_VERSION = 1


def write_json_atomic(path: str | Path, value: Any) -> None:
    """Serialize before replacing a file, so errors preserve the old artifact."""
    target = Path(path)
    payload = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n"
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
    finally:
        Path(temporary).unlink(missing_ok=True)


def file_identity(path: str | Path) -> dict[str, Any]:
    """Hash bytes without loading large checkpoints into memory."""
    target = Path(path)
    before = target.stat()
    digest = hashlib.sha256()
    with target.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    after = target.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError(f"Artifact changed while hashing: {target}")
    return {"sha256": digest.hexdigest(), "size_bytes": after.st_size}


def _cohort_identity(path: Path, *, require_output: bool = False) -> dict[str, Any]:
    digest = hashlib.sha256()
    count = 0
    with path.open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            try:
                row = json.loads(line)
            except (ValueError, TypeError) as exc:
                raise ValueError(f"Invalid cohort JSON at {path}:{number}") from exc
            if not isinstance(row, dict) or not isinstance(row.get("uid"), str) or not row["uid"]:
                raise ValueError(f"Invalid cohort uid at {path}:{number}")
            if require_output and "output" not in row:
                raise ValueError(f"Missing prediction output at {path}:{number}")
            digest.update((json.dumps(row["uid"], ensure_ascii=False) + "\n").encode("utf-8"))
            count += 1
    if not count:
        raise ValueError(f"Empty cohort: {path}")
    return {"line_count": count, "uids_sha256": digest.hexdigest()}


def dataset_identity(path: str | Path) -> dict[str, Any]:
    target = Path(path).resolve()
    before = target.stat()
    result = {"path": str(target), **file_identity(target), **_cohort_identity(target)}
    after = target.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError(f"Dataset changed while identifying cohort: {target}")
    return result


def _weight_files(model: Path) -> set[Path]:
    indexes = [model / name for name in ("model.safetensors.index.json", "pytorch_model.bin.index.json")]
    existing = [index for index in indexes if index.is_file()]
    weights: set[Path] = set()
    for index in existing:
        content = json.loads(index.read_text(encoding="utf-8"))
        weight_map = content.get("weight_map") if isinstance(content, dict) else None
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError(f"Invalid model weight index: {index}")
        for name in weight_map.values():
            if not isinstance(name, str) or not name:
                raise ValueError(f"Invalid model weight name in {index}")
            weight = model / name
            if Path(name).is_absolute() or ".." in Path(name).parts or not weight.is_file():
                raise ValueError(f"Missing or invalid model weight: {name}")
            weights.add(weight)
    if not existing:
        weights = {model / name for name in ("model.safetensors", "pytorch_model.bin") if (model / name).is_file()}
    if not weights or any(weight.stat().st_size == 0 for weight in weights):
        raise ValueError(f"Missing or empty model weights in {model}")
    return weights | set(existing)


def _validate_safetensors(path: Path) -> None:
    """Reject incomplete merges using safetensors header and data boundaries."""
    with path.open("rb") as stream:
        header_size = int.from_bytes(stream.read(8), "little")
        if not 2 <= header_size <= min(100_000_000, path.stat().st_size - 8):
            raise ValueError(f"Invalid or incomplete model weight header: {path}")
        try:
            header = json.loads(stream.read(header_size))
        except (ValueError, UnicodeError) as exc:
            raise ValueError(f"Invalid model weight header JSON: {path}") from exc
    if not isinstance(header, dict):
        raise ValueError(f"Invalid model weight header: {path}")
    tensors = [entry for name, entry in header.items() if name != "__metadata__"]
    data_size = path.stat().st_size - 8 - header_size
    ends = []
    for tensor in tensors:
        offsets = tensor.get("data_offsets") if isinstance(tensor, dict) else None
        if not isinstance(offsets, list) or len(offsets) != 2 or any(type(x) is not int for x in offsets):
            raise ValueError(f"Invalid model weight data offsets: {path}")
        shape = tensor.get("shape")
        if not isinstance(tensor.get("dtype"), str) or not isinstance(shape, list) or any(type(x) is not int or x < 0 for x in shape):
            raise ValueError(f"Invalid model weight tensor metadata: {path}")
        if not 0 <= offsets[0] <= offsets[1] <= data_size:
            raise ValueError(f"Incomplete model weight data: {path}")
        ends.append(offsets[1])
    if not ends or max(ends) != data_size:
        raise ValueError(f"Incomplete or invalid model weight data: {path}")


@lru_cache(maxsize=512)
def _model_file_digest(path: str, signature: tuple[int, ...]) -> tuple[str, int]:
    """Cache model hashing only in this process, keyed by file stat identity."""
    if path.endswith(".safetensors"):
        _validate_safetensors(Path(path))
    identity = file_identity(path)
    return identity["sha256"], identity["size_bytes"]


def _model_file_identity(path: Path) -> dict[str, Any]:
    state = path.stat()
    signature = (state.st_dev, state.st_ino, state.st_size, state.st_mtime_ns, state.st_ctime_ns)
    digest, size = _model_file_digest(str(path), signature)
    return {"sha256": digest, "size_bytes": size}


def model_identity(path: str | Path) -> dict[str, Any]:
    """Validate merged-model completeness and hash all inference assets."""
    model = Path(path).resolve()
    for name in ("config.json", "tokenizer_config.json"):
        if not (model / name).is_file():
            raise ValueError(f"Missing model configuration: {model / name}")
        if not isinstance(json.loads((model / name).read_text(encoding="utf-8")), dict):
            raise ValueError(f"Invalid model configuration JSON object: {model / name}")
    tokenizer_files = ("tokenizer.json", "tokenizer.model", "spiece.model")
    if not any((model / name).is_file() and (model / name).stat().st_size for name in tokenizer_files):
        raise ValueError(f"Missing tokenizer artifact in {model}")
    if (model / "tokenizer.json").is_file():
        if not isinstance(json.loads((model / "tokenizer.json").read_text(encoding="utf-8")), dict):
            raise ValueError(f"Invalid tokenizer JSON object: {model}")
    weights = _weight_files(model)
    assets = {
        item for item in model.iterdir()
        if item.is_file() and item.suffix in {".json", ".model", ".txt", ".tiktoken", ".jinja"}
    }
    return {"path": str(model), "files": {
        str(item.relative_to(model)): _model_file_identity(item) for item in sorted(weights | assets)
    }}


def _json_copy(value: Any) -> Any:
    return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))


def evaluation_identity(
    rooms_path: str | Path,
    model_path: str | Path,
    settings: Mapping[str, Any],
    *,
    code_hashes: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "dataset": dataset_identity(rooms_path),
        "model": model_identity(model_path),
        "settings": _json_copy(dict(settings)),
        "code_hashes": _json_copy(dict(code_hashes or {})),
    }


def freeze_protocol(
    data_dir: str | Path,
    out_dir: str | Path,
    settings: Mapping[str, Any],
    *,
    code_hashes: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Pin all four full cohorts; existing run directories reject input drift."""
    expected = {
        "schema_version": SCHEMA_VERSION,
        "datasets": {split: dataset_identity(Path(data_dir) / f"{split}.jsonl") for split in SPLITS},
        "settings": _json_copy(dict(settings)),
        "code_hashes": _json_copy(dict(code_hashes or {})),
    }
    target = Path(out_dir) / "protocol.json"
    if target.exists():
        if json.loads(target.read_text(encoding="utf-8")) != expected:
            raise ValueError(f"Frozen evaluation protocol differs from current inputs: {target}")
    else:
        write_json_atomic(target, expected)
    return expected


def _validated_artifacts(out_dir: Path, identity: Mapping[str, Any]) -> dict[str, Any]:
    predictions = out_dir / "predictions.jsonl"
    metrics_path = out_dir / "metrics.json"
    observed = _cohort_identity(predictions, require_output=True)
    cohort = identity["dataset"]
    if any(observed[key] != cohort[key] for key in ("line_count", "uids_sha256")):
        raise ValueError(f"Prediction cohort differs from frozen full input: {predictions}")
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    if not isinstance(metrics, dict) or not {"in_dist", "held_out"}.issubset(metrics):
        raise ValueError(f"Invalid metrics population summaries: {metrics_path}")
    populations = [metrics[key] for key in ("in_dist", "held_out") if metrics[key] is not None]
    if any(not isinstance(group, dict) or type(group.get("rooms")) is not int or group["rooms"] < 0 for group in populations):
        raise ValueError(f"Invalid metrics room counts: {metrics_path}")
    if sum(group["rooms"] for group in populations) != cohort["line_count"]:
        raise ValueError(f"Evaluation metrics count differs from frozen cohort: {metrics_path}")
    return {name: file_identity(out_dir / name) for name in ("predictions.jsonl", "metrics.json")}


def finalize_evaluation(out_dir: str | Path, identity: Mapping[str, Any]) -> dict[str, Any]:
    """Write a completion marker only after full output validation succeeds."""
    target = Path(out_dir)
    marker = {
        "schema_version": SCHEMA_VERSION,
        "identity": _json_copy(dict(identity)),
        "artifacts": _validated_artifacts(target, identity),
    }
    write_json_atomic(target / "completion.json", marker)
    return marker


def evaluation_complete(out_dir: str | Path, identity: Mapping[str, Any]) -> bool:
    target = Path(out_dir)
    try:
        marker = json.loads((target / "completion.json").read_text(encoding="utf-8"))
        return (
            marker["schema_version"] == SCHEMA_VERSION
            and marker["identity"] == identity
            and marker["artifacts"] == _validated_artifacts(target, identity)
        )
    except (OSError, ValueError, KeyError, TypeError):
        return False


def summarize_evaluations(
    evaluations: Mapping[str, tuple[str | Path, Mapping[str, Any]]],
) -> dict[str, Any]:
    """Report every requested run, preserving raw/repaired metrics without selection."""
    def summarize(entry: tuple[str | Path, Mapping[str, Any]]) -> dict[str, Any]:
        path, identity = entry
        complete = evaluation_complete(path, identity)
        return {
            "status": "complete" if complete else "incomplete",
            "out_dir": str(Path(path).resolve()),
            "expected_rooms": identity["dataset"]["line_count"],
            "metrics": json.loads((Path(path) / "metrics.json").read_text(encoding="utf-8")) if complete else None,
        }

    results = {name: summarize(entry) for name, entry in evaluations.items()}
    return {
        "complete": bool(results) and all(item["status"] == "complete" for item in results.values()),
        "evaluations": results,
        "interpretation": "Completion verifies coverage and identity; it does not assert model quality or select a test-set winner.",
    }
