#!/usr/bin/env python3
"""Build a pinned legacy release candidate, then audit identities; never launch refits."""
import argparse
from contextlib import ExitStack
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time

SPLITS = ("train", "dev", "test")
CONFIG_KEYS = {"code_root", "code_manifest_sha256", "ir_dir", "ir_manifest",
               "ir_manifest_sha256", "old_release_dir", "old_manifest_sha256", "output_root"}
BOOL_ARGS = {"keep_duplicates", "no_rot90"}
LIST_ARGS = {"sources", "reject_flagged", "boundary_types", "anchors", "cap"}
SCALAR_ARGS = {"min_objects", "max_vertices", "oob_tol", "hidden_max", "dev", "test",
               "constraint_frac", "field_dropout", "desc_dropout"}
ARG_KEYS = BOOL_ARGS | LIST_ARGS | SCALAR_ARGS | {"ir", "out", "source_anchors"}
OUTPUT_FILES = {s + ".jsonl" for s in SPLITS} | {
    "dev_rooms.jsonl", "test_rooms.jsonl", "dev_constrained_rooms.jsonl",
    "test_constrained_rooms.jsonl", "stats.json"}
BUILD_DEPS = {"__init__.py", "build.py", "anchors.py", "scene.py", "split.py", "validate.py"}
CORRECTED_IR = {"Scan2CAD.jsonl", "SpatialGen.jsonl", "IL3D_3dfront.jsonl"}
ADAPTER_OVERLAY = {"adapters/scan2cad.py", "adapters/spatialgen.py", "adapters/il3d_3dfront.py"}


class AuditError(RuntimeError):
    pass


def require(condition, message):
    if not condition:
        raise AuditError(message)


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, f"duplicate JSON key: {key}")
        result[key] = value
    return result


def load_json(path):
    def invalid(value):
        raise AuditError(f"non-finite JSON value: {value}")
    with Path(path).open(encoding="utf-8") as stream:
        return json.load(stream, object_pairs_hook=unique_object, parse_constant=invalid)


def measure(path):
    digest, lines, last = hashlib.sha256(), 0, b""
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
            lines += block.count(b"\n")
            last = block[-1:]
    return digest.hexdigest(), lines + int(bool(last) and last != b"\n")


def verify_file(path, expected, count=None, label="lines"):
    require(isinstance(expected, str) and re.fullmatch(r"[0-9a-f]{64}", expected),
            f"invalid sha256 for {path}")
    require(path.is_file() and not path.is_symlink(), f"missing/nonregular file: {path}")
    digest, lines = measure(path)
    require(digest == expected, f"sha256 mismatch: {path}")
    if count is not None:
        require(type(count) is int and count >= 0 and count == lines, f"{label} mismatch: {path}")


def member(root, name):
    require(isinstance(name, str) and name and not Path(name).is_absolute(), "invalid manifest path")
    require(all(p not in (".", "..") for p in name.split("/")), f"unsafe manifest path: {name}")
    path = root / name
    require(path.resolve().is_relative_to(root), f"manifest path escapes root: {name}")
    require(not any(p.is_symlink() for p in [path, *path.parents] if p != root and p.is_relative_to(root)),
            f"symlink in manifest path: {name}")
    return path


def pinned_json(path, sha):
    verify_file(path, sha)
    value = load_json(path)
    require(isinstance(value, dict), f"manifest/config must be an object: {path}")
    return value


def validate_args(args):
    require(isinstance(args, dict) and set(args) == ARG_KEYS, "unexpected or missing historical args")
    for name in BOOL_ARGS:
        require(type(args[name]) is bool, f"invalid historical args: {name}")
    for name in LIST_ARGS:
        require(isinstance(args[name], list) and all(isinstance(v, str) for v in args[name]),
                f"invalid historical args: {name}")
    for name in SCALAR_ARGS:
        value = args[name]
        require(type(value) in (int, float) and math.isfinite(value), f"invalid historical args: {name}")
    anchors = args["source_anchors"]
    require(isinstance(anchors, dict) and all(isinstance(k, str) and isinstance(v, list)
            and all(isinstance(x, str) for x in v) for k, v in anchors.items()), "invalid source_anchors args")
    require(len(args["sources"]) == 18 and len(set(args["sources"])) == 18, "expected 18 unique sources")
    require(args["constraint_frac"] == .3, "historical v3.2 constraint_frac must be 0.3")


def configuration(path):
    config = load_json(path)
    require(isinstance(config, dict) and set(config) == CONFIG_KEYS, "unexpected/missing config keys")
    for name in CONFIG_KEYS - {k for k in CONFIG_KEYS if k.endswith("sha256")}:
        require(isinstance(config[name], str) and Path(config[name]).is_absolute(), f"absolute path required: {name}")
    require(not os.path.lexists(config["output_root"]), "output_root already exists; refusing overwrite")
    paths = {k: Path(config[k]).resolve() for k in ("code_root", "ir_dir", "ir_manifest", "old_release_dir", "output_root")}
    require(paths["output_root"].parent.is_dir(), "output_root parent must exist")
    for name in ("code_root", "ir_dir", "old_release_dir"):
        require(paths[name].is_dir(), f"input directory missing: {name}")
        require(not paths["output_root"].is_relative_to(paths[name]), "output_root overlaps an input directory")
    require(not paths["ir_manifest"].is_relative_to(paths["ir_dir"]), "IR-MANIFEST must be outside IR directory")
    return {**config, **paths}


def verify_code(config):
    root = config["code_root"]
    manifest = pinned_json(root / "CORRECTION-MANIFEST.json", config["code_manifest_sha256"])
    files = manifest.get("files")
    require(isinstance(files, dict) and files, "empty code file inventory")
    entries = list(root.rglob("*"))
    require(not any(p.is_symlink() for p in entries), "symlink in code package")
    actual = {str(p.relative_to(root)) for p in entries if p.is_file()
              and "__pycache__" not in p.relative_to(root).parts}
    require(actual == set(files) | {"CORRECTION-MANIFEST.json"}, "code file inventory mismatch")
    for name, sha in files.items():
        verify_file(member(root, name), sha)
    require({"fastfill/" + n for n in BUILD_DEPS} <= set(files), "missing pinned build dependencies")
    implementation = {}
    for name, sha in files.items():
        parts = Path(name).parts
        if parts[0] == "fastfill" and name.endswith(".py") and not any(
                p.startswith(("review", ".", "__pycache__")) or p == "tests" for p in parts[1:-1]):
            implementation[str(Path(*parts[1:]))] = sha
    return implementation


def verify_ir(config, old):
    manifest = pinned_json(config["ir_manifest"], config["ir_manifest_sha256"])
    files = manifest.get("files")
    expected = {s + ".jsonl" for s in old["args"]["sources"]}
    require(isinstance(files, dict) and set(files) == expected == set(old["ir_sha256"]),
            "IR manifest must contain exactly the historical 18 sources")
    actual = {p.name for p in config["ir_dir"].iterdir()}
    require(actual == expected, "IR directory inventory mismatch")
    for name, record in files.items():
        require(isinstance(record, dict) and {"sha256", "rows"} <= set(record), "invalid IR file record")
        require(type(record["rows"]) is int and record["rows"] >= 0, f"invalid rows: {name}")
        if name not in CORRECTED_IR:
            require(record["sha256"] == old["ir_sha256"][name], f"unchanged source hash differs: {name}")
        verify_file(member(config["ir_dir"], name), record["sha256"], record["rows"], "rows")
    return {name: record["sha256"] for name, record in files.items()}


def verify_release_files(root, manifest, exact=False):
    files = manifest.get("files")
    require(isinstance(files, dict) and set(files) == OUTPUT_FILES, "release file inventory mismatch")
    if exact:
        require({p.name for p in root.iterdir()} == OUTPUT_FILES | {"MANIFEST.json"}, "unexpected output files")
    for name, record in files.items():
        require(isinstance(record, dict) and {"sha256", "lines"} <= set(record), "invalid release file record")
        require(type(record["lines"]) is int and record["lines"] >= 0, f"invalid lines: {name}")
        verify_file(member(root, name), record["sha256"], record["lines"])


def verify_inputs(config):
    old = pinned_json(config["old_release_dir"] / "MANIFEST.json", config["old_manifest_sha256"])
    validate_args(old.get("args"))
    code = verify_code(config)
    frozen = old.get("code_sha256")
    require(isinstance(frozen, dict) and set(frozen) <= set(code)
            and set(code) - set(frozen) <= {"adapters/upright_bounds.py"}, "frozen implementation inventory differs")
    require(all(code[n] == sha for n, sha in frozen.items() if n not in ADAPTER_OVERLAY),
            "frozen implementation changed outside three source adapters")
    ir = verify_ir(config, old)
    verify_release_files(config["old_release_dir"], old)
    return old, code, ir


def build_arguments(args, ir, out):
    effective = {**args, "ir": str(ir), "out": str(out)}
    command = []
    for key, value in effective.items():
        if key in BOOL_ARGS:
            command.extend(["--" + key] if value else [])
        elif key == "source_anchors":
            command.extend(["--source_anchors", *(k + "=" + ",".join(v) for k, v in value.items())])
        else:
            command.extend(["--" + key, *(str(v) for v in value)] if key in LIST_ARGS else ["--" + key, str(value)])
    return command, effective


def python_prefix(output):
    # -B alone still reads existing .pyc. An isolated, new cache prefix prevents that.
    return [sys.executable, "-I", "-B", "-X", "pycache_prefix=" + str(output / "unused-pycache")]


def runtime_versions(output):
    code = ("import sys,json,platform,numpy,shapely; print(json.dumps(dict("
            "python=sys.version,executable=sys.executable,platform=platform.platform(),"
            "numpy=numpy.__version__,shapely=shapely.__version__,"
            "geos=getattr(shapely,'geos_version_string',None))))")
    result = subprocess.run(python_prefix(output) + ["-c", code], check=True, capture_output=True,
                            text=True, timeout=60)
    return json.loads(result.stdout)


def execute_build(command, cwd, log_path):
    with log_path.open("xb") as log:
        subprocess.run(command, cwd=cwd, stdout=log, stderr=subprocess.STDOUT, check=True)


def canonical_digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                   separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def identity_index(root, sources):
    index = {}
    for split in SPLITS:
        with (root / (split + ".jsonl")).open(encoding="utf-8") as stream:
            for number, line in enumerate(stream, 1):
                record = json.loads(line, object_pairs_hook=unique_object)
                uid, source = record.get("uid"), record.get("source")
                require(isinstance(uid, str) and uid.strip(), f"invalid UID: {split}:{number}")
                require(uid not in index, f"duplicate UID: {uid}")
                require(source in sources, f"invalid source: {uid}")
                require(isinstance(record.get("flags"), dict) and isinstance(record.get("messages"), list),
                        f"missing flags/messages: {uid}")
                index[uid] = dict(split=split, source=source,
                                  flags_sha256=canonical_digest(record["flags"]),
                                  messages_sha256=canonical_digest(record["messages"]))
    return index


def source_counters(old, new, sources):
    fields = ("added", "dropped", "moved_in", "moved_out", "source_changed_in", "source_changed_out",
              "common", "flags_changed", "flags_unchanged", "messages_changed", "messages_unchanged")
    result = {s: {**dict.fromkeys(fields, 0), "old_by_split": dict.fromkeys(SPLITS, 0),
                  "new_by_split": dict.fromkeys(SPLITS, 0)}
              for s in sorted(sources)}
    for label, index in (("old", old), ("new", new)):
        for entry in index.values():
            result[entry["source"]][label + "_by_split"][entry["split"]] += 1
    return result


def emit(stream, value):
    stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True) + "\n")


def compare_identities(old, new, output, sources):
    counts = dict.fromkeys(("added", "dropped", "moved", "source_changed", "common", "flags_changed",
                            "messages_changed", "old_heldout_moved", "old_heldout_to_train", "old_heldout_dropped"), 0)
    by_source = source_counters(old, new, sources)
    names = ("added", "dropped", "moved", "source_changed", "changed", "identity-map")
    with ExitStack() as stack:
        logs = {n: stack.enter_context((output / (n + ".jsonl")).open("x", encoding="utf-8")) for n in names}
        for uid in sorted(old.keys() | new.keys()):
            before, after = old.get(uid), new.get(uid)
            entry = dict(uid=uid, old=before, new=after)
            emit(logs["identity-map"], entry)
            if before is None or after is None:
                kind = "added" if before is None else "dropped"
                counts[kind] += 1
                by_source[(after or before)["source"]][kind] += 1
                counts["old_heldout_dropped"] += int(after is None and before["split"] in ("dev", "test"))
                emit(logs[kind], entry)
                continue
            compare_common(entry, counts, by_source, logs)
    counts.update(old=len(old), new=len(new))
    return counts, by_source


def compare_common(entry, counts, by_source, logs):
    before, after = entry["old"], entry["new"]
    src = by_source[before["source"]]
    counts["common"] += 1
    src["common"] += 1
    moved, source_changed = before["split"] != after["split"], before["source"] != after["source"]
    if moved:
        counts["moved"] += 1
        src["moved_out"] += 1
        by_source[after["source"]]["moved_in"] += 1
        heldout = before["split"] in ("dev", "test")
        counts["old_heldout_moved"] += int(heldout)
        counts["old_heldout_to_train"] += int(heldout and after["split"] == "train")
        emit(logs["moved"], entry)
    if source_changed:
        counts["source_changed"] += 1
        src["source_changed_out"] += 1
        by_source[after["source"]]["source_changed_in"] += 1
        emit(logs["source_changed"], entry)
    changed = {}
    for field in ("flags", "messages"):
        different = before[field + "_sha256"] != after[field + "_sha256"]
        changed[field + "_changed"] = different
        counts[field + "_changed"] += int(different)
        src[field + ("_changed" if different else "_unchanged")] += 1
    if any(changed.values()):
        emit(logs["changed"], {**entry, **changed})


def decision(counts):
    preserved = not any(counts[k] for k in ("added", "dropped", "moved", "source_changed"))
    blockers = ["full_component_closure_not_checked", "scientific_acceptance_not_performed"]
    for field, reason in (("old_heldout_moved", "old_heldout_split_migration"),
                          ("old_heldout_dropped", "old_heldout_dropped"),
                          ("source_changed", "source_identity_changed")):
        if counts[field]:
            blockers.append(reason)
    if not preserved:
        blockers.append("identity_population_or_split_changed")
    return dict(status="candidate_complete_identity_" + ("preserved" if preserved else "changed"),
                identity_gate_passed=preserved, refit_ready=False, scientifically_accepted=False,
                full_component_closure_checked=False, readiness_blockers=blockers)


def write_json(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, allow_nan=False, indent=2)
        stream.write("\n")


def run(config_path, runner=execute_build, version_reader=runtime_versions):
    started = time.time()
    config_path = Path(config_path).resolve()
    config_sha = measure(config_path)[0]
    config = configuration(config_path)
    old_manifest, code_hashes, ir_hashes = verify_inputs(config)
    old = identity_index(config["old_release_dir"], old_manifest["args"]["sources"])
    verify_file(config_path, config_sha)
    output = config["output_root"]
    output.mkdir()  # Exclusive creation, including an empty or concurrently created destination.
    receipt = dict(config_path=str(config_path), config_sha256=config_sha, started_unix=started,
                   code_manifest_sha256=config["code_manifest_sha256"],
                   ir_manifest_sha256=config["ir_manifest_sha256"],
                   old_manifest_sha256=config["old_manifest_sha256"])
    try:
        receipt["runtime"] = version_reader(output)
        arguments, effective = build_arguments(old_manifest["args"], config["ir_dir"], output / "release")
        boot = "import runpy,sys; sys.path.insert(0,sys.argv.pop(1)); sys.argv[0]='fastfill.build'; runpy.run_module('fastfill.build',run_name='__main__')"
        command = python_prefix(output) + ["-c", boot, str(config["code_root"]), *arguments]
        receipt.update(effective_build_args=effective, command=command)
        write_json(output / "invocation.json", receipt)
        try:
            runner(command, config["code_root"], output / "build.log")
        finally:
            verify_file(config_path, config_sha)
            verify_inputs(config)
        candidate = load_json(output / "release/MANIFEST.json")
        require(candidate.get("args") == effective, "candidate args differ from exact historical args")
        require(candidate.get("ir_sha256") == ir_hashes, "candidate IR hashes differ from pinned inputs")
        require(candidate.get("code_sha256") == candidate.get("build_start_code_sha256") == code_hashes,
                "candidate code hashes differ from pinned implementation")
        verify_release_files(output / "release", candidate, exact=True)
        new = identity_index(output / "release", old_manifest["args"]["sources"])
        counts, by_source = compare_identities(old, new, output, old_manifest["args"]["sources"])
        ledgers = {p.name: dict(zip(("sha256", "rows"), measure(p))) for p in sorted(output.glob("*.jsonl"))}
        result = {**receipt, **decision(counts), "counts": counts, "by_source": by_source, "ledgers": ledgers,
                  "candidate_manifest_sha256": measure(output / "release/MANIFEST.json")[0],
                  "input_integrity_before_after": "passed", "elapsed_seconds": time.time() - started}
        write_json(output / "summary.json", result)
        return result
    except Exception as exc:
        write_json(output / "summary.json", {**receipt, "status": "failed", "refit_ready": False,
                   "error_type": type(exc).__name__, "error": str(exc), "elapsed_seconds": time.time() - started})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = run(args.config)
        print(json.dumps({k: result[k] for k in ("status", "counts", "refit_ready", "readiness_blockers")}))
        return 0 if result["identity_gate_passed"] else 2
    except Exception as exc:
        print(json.dumps({"status": "failed", "error_type": type(exc).__name__, "error": str(exc)}), file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
