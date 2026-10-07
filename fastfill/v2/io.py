"""Dataset safety, immutable outputs and reproducible run metadata."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess

import torch

from fastfill.v2.schema import migrate_legacy_row

SOURCE_ROOT = Path("/Volumes/harddisk/3D_Room_Collections")
# RoomGenBench benchmark rooms (SAGE layouts) never train; qualification writes them to test
# and every training-data reader refuses them. multisource_verify keeps an independent copy.
ROOMGENBENCH_HOLDOUT_GROUPS = ("sage:layout_61ebde9f", "sage:layout_6b049b06", "sage:layout_fef15043",
                               "sage:layout_60ee2ae3", "sage:layout_46cdcce3")
HOLDOUT_REASON = "roomgenbench_benchmark_room"


def safe_output(path, *, create=False):
    target = Path(path).resolve()
    source = SOURCE_ROOT.resolve()
    if target == source or source in target.parents or target in source.parents:
        raise ValueError("generated outputs must stay outside the read-only source root and ancestors")
    if target.exists():
        raise FileExistsError(f"output already exists; choose a new run path: {target}")
    if create:
        target.mkdir(parents=True, exist_ok=False)
    return target


def read_samples(path, *, training=False, max_samples=None):
    """Read audited v2 rows; pre-C1 rows get their exchangeable groups migrated into validity once, here."""
    if max_samples is not None and (isinstance(max_samples, bool) or not isinstance(max_samples, int) or max_samples < 1):
        raise ValueError("max_samples must be a positive integer")
    path = Path(path)
    with path.open() as stream:
        rows = []
        for line in stream:
            if line.strip():
                rows.append(json.loads(line))
                if max_samples is not None and len(rows) >= max_samples:
                    break
    if not rows:
        raise ValueError("dataset is empty")
    for row in rows:
        if row.get("schema_version") != "fastfill.v2" or not isinstance(row.get("provenance"), dict):
            raise ValueError("audited v2 sample with provenance is required")
        if training and row["provenance"].get("split") != "train":
            raise ValueError("training accepts only explicitly marked training-split rows")
        if training and (row["provenance"].get("group") in ROOMGENBENCH_HOLDOUT_GROUPS or row["provenance"].get("holdout_reason")):
            raise ValueError("RoomGenBench benchmark room in training data; build training data through qualification")
    return [migrate_legacy_row(row) for row in rows]


def ensure_disjoint(train, holdout):
    def identity(row):
        p = row["provenance"]
        if not p.get("source") or not p.get("house_id"):
            raise ValueError("split integrity needs source and underlying house identity")
        return p["source"], p["house_id"]
    if {identity(r) for r in train} & {identity(r) for r in holdout}:
        raise ValueError("training and held-out scenes share an underlying source house")


def fingerprint(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_metadata(data_path):
    import importlib.metadata
    packages = {}
    for name in ("torch", "transformers", "peft", "scipy", "accelerate"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    repo_root = Path(__file__).resolve().parents[2]
    result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo_root, capture_output=True, text=True, check=False)
    status = subprocess.run(["git", "status", "--porcelain"], cwd=repo_root, capture_output=True, text=True, check=False)
    source_files = sorted(Path(__file__).parent.glob("*.py"))
    hashes = {str(path.relative_to(repo_root)): fingerprint(path) for path in source_files}
    tree_hash = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    return {"data_path": str(Path(data_path).resolve()), "data_sha256": fingerprint(data_path),
            "code_commit": result.stdout.strip() if result.returncode == 0 else None,
            "code_dirty": bool(status.stdout.strip()) if status.returncode == 0 else None,
            "implementation_files_sha256": hashes, "implementation_sha256": tree_hash,
            "packages": packages}


def to_device(value, device):
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {k: to_device(v, device) for k, v in value.items()}
    # Condition metadata lists remain Python values; tensor fields live in dicts.
    return value
