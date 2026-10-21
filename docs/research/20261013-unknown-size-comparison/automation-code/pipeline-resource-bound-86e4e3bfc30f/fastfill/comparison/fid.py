"""Offline, conditional real-image FID on a frozen, explicitly enumerated cohort.

Manifest: {cases: [{uid, stratum, images: {reference: {path, sha256},
joint: {path, sha256}, placement: {path, sha256}}, failure: ...}],
config: {...}, provenance: {...}}. ``featurecases`` is an alias for ``cases``.
Include EVERY frozen UID, including failures with absent images. Optional
frozen_uids must exactly match cases. Paths are relative to the manifest.
Each entry selects ONE perspective RGB 512x512 image per room/arm; never top.
The parent maps rendered ledger layout/perspective paths and image_sha256 into
these entries, retaining ledger/config/input/layout/material hashes in provenance.
This worker does not infer arm/stage mappings, render images, or download files.

API: run_manifest(manifest, out, *, vendor, weights, batch_size=8, threads=4,
device='cpu'). Output is a JSON file with sibling .features.npz/.features.json.
``extractor`` is an explicit testing hook; its outputs are labeled injected.
CLI: python -m fastfill.comparison.fid --manifest MANIFEST --out RESULT.json
--vendor VENDOR --weights WEIGHTS [--batch-size 8 --threads 4 --device cpu].
FID uses sample covariance (ddof=1), separately by stratum, on the same complete
three-arm UID intersection. N<2 has no score; strata are never averaged.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import re


ARMS = ("reference", "joint", "placement")
FEATURE_DIM = 2048
# Recorded in paper-metrics-20261013/fid-environment.json; independently hashed
# from the existing 95,628,359-byte official checkpoint. Never trust its filename.
WEIGHTS_SHA256 = "6726825d0af5f729cebd5821db510b11b1cfad8faad88a03f1befd49fb9129b2"
DISCLOSURE = (
    "FID is conditional on shared successful reference, joint and placement images "
    "and features, with one fixed RGB 512x512 perspective per frozen room UID. "
    "Failures remain in requested counts; no missing image or feature is imputed "
    "or substituted. Reference eligibility means a verified reference image. "
    "Small cohorts give unstable, biased FID estimates. Scores are reported "
    "separately by stratum and are neither pooled nor averaged."
)


@dataclass(frozen=True)
class ImageRef:
    path: str
    sha256: str


def file_sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify_weights(path):
    digest = file_sha256(path)
    if digest != WEIGHTS_SHA256:
        raise ValueError("weights sha256 mismatch: official FID checkpoint required")
    return digest


def _json_write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def _feature_array(value, rows=None, dims=None):
    import numpy as np

    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] == 0:
        raise ValueError("features must be an N x D matrix")
    if rows is not None and array.shape[0] != rows:
        raise ValueError("features row count mismatch")
    if dims is not None and array.shape[1] != dims:
        raise ValueError(f"features dimension must be {dims}")
    if not np.isfinite(array).all():
        raise ValueError("features must be finite")
    return array


def frechet_distance(x, y):
    """Standard Gaussian FID via the equivalent low-rank nuclear norm.

    For A=X_centered/sqrt(N-1), B=Y_centered/sqrt(M-1), the covariance
    square-root trace equals sum(svd(A @ B.T)); no diagonal jitter is added.
    This avoids singular 2048x2048 covariance square roots for small cohorts.
    """
    import numpy as np

    x, y = _feature_array(x), _feature_array(y)
    if min(len(x), len(y)) < 2 or x.shape[1] != y.shape[1]:
        raise ValueError("FID requires N,M >= 2 and constant feature dimension")
    mx, my = x.mean(axis=0), y.mean(axis=0)
    a = (x - mx) / np.sqrt(len(x) - 1)
    b = (y - my) / np.sqrt(len(y) - 1)
    trace = np.linalg.svd(a @ b.T, compute_uv=False).sum()
    scale = float(np.square(mx - my).sum() + np.square(a).sum() + np.square(b).sum())
    score = float(scale - 2 * trace)
    if not np.isfinite(score) or score < -1e-8 * max(1.0, scale):
        raise ValueError("nonfinite or materially negative FID")
    return max(0.0, score)


def read_image(ref):
    """Decode exactly the bytes whose hash was checked, without conversion."""
    from PIL import Image

    data = Path(ref.path).read_bytes()
    if hashlib.sha256(data).hexdigest() != ref.sha256:
        raise ValueError("image hash mismatch")
    with Image.open(io.BytesIO(data)) as image:
        if image.mode != "RGB" or image.size != (512, 512):
            raise ValueError("image must be RGB 512x512; conversion/resizing forbidden")
        if getattr(image, "n_frames", 1) != 1:
            raise ValueError("image must have exactly one frame")
        image.load()
        return image.copy()


def _image_ref(entry, base):
    if not isinstance(entry, dict) or not isinstance(entry.get("path"), str) or not entry["path"]:
        raise ValueError("missing image path")
    digest = entry.get("sha256")
    if not isinstance(digest, str) or re.fullmatch(r"[0-9a-fA-F]{64}", digest) is None:
        raise ValueError("missing or invalid image sha256")
    if entry.get("view", "perspective") != "perspective":
        raise ValueError("only perspective images are eligible")
    return ImageRef(str((base / entry["path"]).resolve()), digest.lower())


def _load_inception(vendor, weights):
    """Load a private vendor module, replacing its URL loader with local bytes."""
    from importlib.metadata import Distribution
    import torch

    verify_weights(weights)
    vendor = Path(vendor).resolve()
    dist = Distribution.at(vendor / "pytorch_fid-0.3.0.dist-info")
    if dist.version != "0.3.0":
        raise ValueError("vendor must contain pytorch_fid 0.3.0")
    spec = importlib.util.spec_from_file_location("_optiscene_fid_inception", vendor / "pytorch_fid/inception.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    def local_weights(url, **_kwargs):
        if url != module.FID_WEIGHTS_URL:
            raise ValueError("unexpected weights URL; downloads disabled")
        data = Path(weights).read_bytes()
        if hashlib.sha256(data).hexdigest() != WEIGHTS_SHA256:
            raise ValueError("weights sha256 changed before consumption")
        return torch.load(io.BytesIO(data), map_location="cpu", weights_only=True)

    module.load_state_dict_from_url = local_weights
    return module.InceptionV3([3], resize_input=True, normalize_input=True,
                              requires_grad=False, use_fid_inception=True)


def extract_features(images, *, vendor, weights, batch_size=8, threads=4, device="cpu"):
    """Actual pytorch-fid pool3 features; no network or alternative extractor."""
    import numpy as np
    import torch

    _settings(batch_size, threads, device)
    torch.set_num_threads(threads)
    model = _load_inception(vendor, weights).to(device).eval()
    batches = []
    with torch.inference_mode():
        for start in range(0, len(images), batch_size):
            arrays = [np.array(read_image(ref), dtype=np.float32).transpose(2, 0, 1) / 255.0
                      for ref in images[start:start + batch_size]]
            inputs = torch.from_numpy(np.stack(arrays)).to(device)
            values = model(inputs)[0].squeeze(3).squeeze(2).cpu().numpy()
            batches.append(_feature_array(values, len(arrays), FEATURE_DIM))
    return np.concatenate(batches) if batches else np.empty((0, FEATURE_DIM))


def _settings(batch_size, threads, device):
    if any(isinstance(v, bool) or not isinstance(v, int) or v < 1 for v in (batch_size, threads)):
        raise ValueError("batch_size and threads must be positive integers")
    if not isinstance(device, str) or not device:
        raise ValueError("device must be a nonempty torch device string")


def _load_manifest(path):
    raw = Path(path).read_bytes()
    data = json.loads(raw)
    if not isinstance(data, dict) or ("cases" in data and "featurecases" in data):
        raise ValueError("manifest requires exactly one cases/featurecases list")
    cases = data.get("cases", data.get("featurecases"))
    if not isinstance(cases, list):
        raise ValueError("manifest cases must enumerate all frozen UIDs")
    uids = []
    for case in cases:
        if not isinstance(case, dict) or any(not isinstance(case.get(k), str) or not case[k]
                                             for k in ("uid", "stratum")):
            raise ValueError("each frozen case requires uid and stratum")
        uids.append(case["uid"])
    if len(set(uids)) != len(uids):
        raise ValueError("duplicate frozen UID")
    if "frozen_uids" in data:
        frozen = data["frozen_uids"]
        if (not isinstance(frozen, list) or not all(isinstance(uid, str) for uid in frozen)
                or sorted(frozen) != sorted(uids)):
            raise ValueError("cases must enumerate all frozen_uids, including failures")
    return data, cases, hashlib.sha256(raw).hexdigest()


def _inspect_cases(cases, base):
    """Inspect all arms, retaining every failure and the original denominator."""
    checked, failures, memo = {}, [], {}
    for case in cases:
        uid, stratum = case["uid"], case["stratum"]
        declared = case.get("failure")
        if declared:
            failures.append({"uid": uid, "stratum": stratum, "arm": "case",
                             "reason": "declared_failure", "detail": declared})
        images = case.get("images") or {}
        good = {}
        for arm in ARMS:
            try:
                if not isinstance(images, dict):
                    raise ValueError("images must be an arm mapping")
                entry = images.get(arm)
                if isinstance(entry, dict) and entry.get("failure"):
                    raise ValueError(f"declared image failure: {entry['failure']}")
                ref = _image_ref(entry, base)
                if ref not in memo:
                    try:
                        read_image(ref)
                        memo[ref] = None
                    except (OSError, ValueError) as exc:
                        memo[ref] = f"{type(exc).__name__}: {exc}"
                if memo[ref] is not None:
                    raise ValueError(memo[ref])
                good[arm] = ref
            except (OSError, ValueError) as exc:
                failures.append({"uid": uid, "stratum": stratum, "arm": arm,
                                 "reason": f"{type(exc).__name__}: {exc}"})
        checked[uid] = {"images": good, "candidate": len(good) == 3 and not declared}
    return checked, failures


def _cache_receipt(path, metadata_path):
    return {"path": str(path), "sha256": file_sha256(path),
            "metadata_path": str(metadata_path), "metadata_sha256": file_sha256(metadata_path)}


def _cached_features(refs, out, *, extractor, options, weight_hash):
    import numpy as np

    if not refs:
        return {}, None, None
    path, meta_path = out.with_suffix(".features.npz"), out.with_suffix(".features.json")
    identity = {"schema_version": 1, "dims": FEATURE_DIM, "weights_sha256": weight_hash,
                "extractor": "injected" if extractor else "pytorch_fid:0.3.0:pool3",
                "preprocessing": "RGB512; float32/255; bilinear299 align_corners=False; 2*x-1",
                "images": [{"path": ref.path, "sha256": ref.sha256} for ref in refs],
                "device": options["device"], "batch_size": options["batch_size"],
                "threads": options["threads"]}
    if not extractor:
        from importlib.metadata import version

        identity = {**identity,
                    "vendor_inception_sha256": file_sha256(Path(options["vendor"]) / "pytorch_fid/inception.py"),
                    "versions": {name: version(name) for name in ("torch", "torchvision", "numpy", "Pillow")}}
    if path.exists() and meta_path.exists():
        metadata = json.loads(meta_path.read_text())
        if metadata.get("identity") == identity:
            if file_sha256(path) != metadata.get("npz_sha256"):
                raise ValueError("feature cache hash mismatch")
            try:
                with np.load(path, allow_pickle=False) as stored:
                    values = _feature_array(stored["features"], len(refs), FEATURE_DIM)
            except (OSError, ValueError, KeyError) as exc:
                raise ValueError(f"invalid feature cache: {exc}") from exc
            return dict(zip(refs, values)), _cache_receipt(path, meta_path), None
    try:
        values = _feature_array((extractor or extract_features)(refs, **options), len(refs), FEATURE_DIM)
        # Also protect injected extractors and detect changes during a run.
        for ref in refs:
            if file_sha256(ref.path) != ref.sha256:
                raise ValueError("image hash changed during feature extraction")
    except Exception as exc:
        return {}, None, f"feature_extraction_failed: {type(exc).__name__}: {exc}"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, features=values)
    temporary.replace(path)
    _json_write(meta_path, {"identity": identity, "npz_sha256": file_sha256(path),
                            "shape": list(values.shape), "dtype": str(values.dtype)})
    return dict(zip(refs, values)), _cache_receipt(path, meta_path), None


def _stratum_result(cases, checked, features):
    shared = [case for case in cases if checked[case["uid"]]["candidate"]
              and all(ref in features for ref in checked[case["uid"]]["images"].values())]
    scores = {f"{arm}_vs_reference": None for arm in ARMS[1:]}
    if len(shared) >= 2:
        arrays = {arm: [features[checked[c["uid"]]["images"][arm]] for c in shared] for arm in ARMS}
        scores = {f"{arm}_vs_reference": frechet_distance(arrays[arm], arrays["reference"]) for arm in ARMS[1:]}
    return {"counts": {"requested": len(cases), "reference_eligible": sum(
                "reference" in checked[c["uid"]]["images"] for c in cases), "shared_success": len(shared)},
            "shared_uids": [c["uid"] for c in shared], **scores,
            "status": "ok" if len(shared) >= 2 else "insufficient_sample_size"}


def _protect_outputs(manifest, out, weights, cases):
    inputs = {manifest, Path(weights).resolve()}
    for case in cases:
        images = case.get("images")
        if isinstance(images, dict):
            for entry in images.values():
                if isinstance(entry, dict) and isinstance(entry.get("path"), str):
                    inputs.add((manifest.parent / entry["path"]).resolve())
    outputs = {out, out.with_suffix(".features.npz").resolve(), out.with_suffix(".features.json").resolve()}
    outputs = outputs | {path.with_name(path.name + ".tmp").resolve() for path in outputs}
    if inputs & outputs:
        raise ValueError("output must not overwrite inputs")


def run_manifest(manifest, out, *, vendor, weights, batch_size=8, threads=4, device="cpu", extractor=None):
    """Write scores, complete failure ledger, and a hash-bound feature cache."""
    _settings(batch_size, threads, device)
    manifest, out = Path(manifest).resolve(), Path(out).resolve()
    data, cases, manifest_hash = _load_manifest(manifest)
    _protect_outputs(manifest, out, weights, cases)
    weight_hash = verify_weights(weights)
    checked, failures = _inspect_cases(cases, manifest.parent)
    refs = list(dict.fromkeys(ref for c in cases if checked[c["uid"]]["candidate"]
                             for ref in checked[c["uid"]]["images"].values()))
    options = {"vendor": vendor, "weights": weights, "batch_size": batch_size,
               "threads": threads, "device": device}
    features, cache, error = _cached_features(refs, out, extractor=extractor, options=options, weight_hash=weight_hash)
    if error:
        failures = failures + [{"uid": c["uid"], "stratum": c["stratum"], "arm": "features", "reason": error}
                               for c in cases if checked[c["uid"]]["candidate"]]
    strata = {s: _stratum_result([c for c in cases if c["stratum"] == s], checked, features)
              for s in sorted({c["stratum"] for c in cases})}
    result = {"schema_version": 1, "metric": "FID", "feature_dimension": FEATURE_DIM,
              "extractor": "injected" if extractor else "pytorch_fid:0.3.0:pool3",
              "covariance_ddof": 1, "weights_sha256": weight_hash,
              "manifest": {"path": str(manifest), "sha256": manifest_hash},
              "provenance": {k: v for k, v in data.items() if k not in ("cases", "featurecases")},
              "case_provenance": [{k: v for k, v in c.items() if k != "images"} for c in cases],
              "counts": {key: sum(s["counts"][key] for s in strata.values())
                         for key in ("requested", "reference_eligible", "shared_success")},
              "strata": strata, "failures": failures, "feature_cache": cache, "disclosure": DISCLOSURE}
    _json_write(out, result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for flag in ("manifest", "out", "vendor", "weights"):
        parser.add_argument(f"--{flag}", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)
    try:
        result = run_manifest(**vars(args))
    except (OSError, ValueError) as exc:
        parser.exit(2, f"FID worker: {exc}\n")
    print(json.dumps({"out": str(args.out), "counts": result["counts"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
