"""Synthetic images and mocked features; never run Inception in unit tests."""
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from scipy.linalg import sqrtm

from fastfill.comparison import fid


def covariance_fid(x, y):
    delta = x.mean(0) - y.mean(0)
    a, b = np.cov(x, rowvar=False, ddof=1), np.cov(y, rowvar=False, ddof=1)
    return delta @ delta + np.trace(a + b - 2 * sqrtm(a @ b).real)


@pytest.mark.parametrize("kind", ["identical", "translated", "scaled", "different_n", "singular"])
def test_low_rank_matches_sample_covariance(kind):
    rng = np.random.default_rng(7)
    x = rng.normal(size=(12, 4))
    y = {"identical": x, "translated": x + 3, "scaled": x * 2,
         "different_n": rng.normal(size=(9, 4)), "singular": x[:2]}[kind]
    actual = fid.frechet_distance(x, y)
    assert actual == pytest.approx(covariance_fid(x, y), abs=2e-6)
    if kind == "translated":
        assert actual == pytest.approx(36)
    if kind == "scaled":
        assert actual == pytest.approx(x.mean(0) @ x.mean(0) + np.var(x, axis=0, ddof=1).sum())


def test_kid_matches_naive_paired_u_statistic():
    rng = np.random.default_rng(3)
    x, y = rng.normal(size=(7, 5)), rng.normal(size=(7, 5)) + .5
    k = lambda a, b: (a @ b / 5 + 1) ** 3  # noqa: E731
    h = [[k(x[i], x[j]) + k(y[i], y[j]) - k(x[i], y[j]) - k(x[j], y[i]) for j in range(7)] for i in range(7)]
    h1 = [sum(h[i][j] for j in range(7) if j != i) / 6 for i in range(7)]
    result = fid.kernel_inception_distance(x, y)
    assert result["kid"] == pytest.approx(np.mean(h1))
    assert result["se"] == pytest.approx(2 * np.sqrt(np.var(h1, ddof=1) / 7))
    assert fid.kernel_inception_distance(x, x)["kid"] == pytest.approx(0, abs=1e-12)
    with pytest.raises(ValueError):
        fid.kernel_inception_distance(x[:1], y[:1])


def test_systems_mode_amendment_011_populations_kid_and_small_n(tmp_path, worker):
    run, calls = worker
    entry = image_entry(tmp_path)
    other = image_entry(tmp_path, "other.png", 90)

    def systems_case(uid, stratum="front", **images):
        return {"uid": uid, "stratum": stratum,
                "images": {arm: dict(images.get(arm, entry)) for arm in ("reference", "J", "P", "L")}}

    cases = [systems_case("a"), systems_case("b", J=other), systems_case("c", L={"failure": "schema_incomplete"}),
             systems_case("d", P={"path": "absent.png", "sha256": entry["sha256"]}), systems_case("e", "holo")]
    result = run(cases, systems=["J", "P", "L"], judged=["J", "P", "L"])
    front = result["strata"]["front"]
    # Primary = reference & J & P: an agent failure (c) no longer removes the room from J/P.
    assert front["counts"] == {"requested": 4, "reference_eligible": 4, "shared_success": 3}
    assert front["shared_uids"] == ["a", "b", "c"] and front["small_n"] is True and set(front["kid"]) == {"J", "P"}
    assert front["P_vs_reference"] == pytest.approx(0) and front["J_vs_reference"] > 0 and "L_vs_reference" not in front
    pops = front["populations"]
    assert set(pops) == {"primary", "agent/L", "judged"} and pops["primary"]["uids"] == ["a", "b", "c"]
    agent = pops["agent/L"]
    assert (agent["systems"], agent["n"], agent["uids"]) == (["J", "P", "L"], 2, ["a", "b"])
    assert agent["fid"]["L"] == pytest.approx(0) and agent["fid"]["J"] > 0 and set(agent["kid"]) == {"J", "P", "L"}
    assert agent["fid"]["J"] != pytest.approx(pops["primary"]["fid"]["J"])  # J re-scored on the agent's rooms.
    assert pops["judged"]["uids"] == ["a", "b"] and front["kid"]["P"] == {"kid": pytest.approx(0), "se": pytest.approx(0)}
    holo = result["strata"]["holo"]
    assert holo["kid"] == {"J": None, "P": None} and holo["populations"]["agent/L"]["status"] == "insufficient_sample_size"
    pooled = result["pooled"]
    assert pooled["shared_uids"] == ["a", "b", "c", "e"] and pooled["populations"]["agent/L"]["uids"] == ["a", "b", "e"]
    assert pooled["counts"] == {"requested": 5, "reference_eligible": 5, "shared_success": 4}
    assert {(f["uid"], f["arm"]) for f in result["failures"]} == {("c", "L"), ("d", "P")}
    assert result["systems"] == ["J", "P", "L"] and result["small_n_threshold"] == 50 and result["judged"] == ["J", "P", "L"]
    assert result["amendment"] == "011" and "Amendment 011" in result["disclosure"] and len(calls) == 1
    with pytest.raises(ValueError, match="pre-specified judged set"):  # Never silently left out.
        run(cases, systems=["J", "P", "L"])
    assert "kid" not in run([case("a", entry), case("b", entry)])["strata"]["front"]  # J/P path unchanged
    for bad in ({"systems": ["reference"]}, {"systems": ["J", "L"]}):
        with pytest.raises(ValueError, match="systems"):
            run(cases, **bad)
    with pytest.raises(ValueError, match="judged"):
        run(cases, systems=["J", "P", "L"], judged=["J", "L"])


def test_recompute_reuses_a_visual_output_and_verifies_its_manifest(tmp_path, monkeypatch):
    monkeypatch.setattr(fid, "verify_weights", lambda path: fid.WEIGHTS_SHA256)
    visual = tmp_path / "visual"
    visual.mkdir()
    entry = image_entry(visual)
    cases = [{"uid": u, "stratum": "front", "images": {a: dict(entry) for a in ("reference", "J", "P", "O6")}}
             for u in ("a", "b")]
    (visual / "fid-manifest.json").write_text(json.dumps({"cases": cases, "systems": ["J", "P", "O6"]}))
    (visual / "preparation.json").write_text(json.dumps({"judged": ["J", "P", "O6"]}))
    (visual / "MANIFEST.json").write_text(json.dumps({"files": {n: fid.file_sha256(visual / n)
                                                                for n in ("fid-manifest.json", "preparation.json")}}))
    extract = lambda images, **kwargs: np.zeros((len(images), 2048))  # noqa: E731
    out = tmp_path / "fid" / "fid.json"
    result = fid.recompute(visual, out, vendor=tmp_path, weights=tmp_path / "w", extractor=extract)
    assert result["judged"] == ["J", "P", "O6"] and result["strata"]["front"]["populations"]["judged"]["n"] == 2
    assert result["manifest"]["path"] == str((visual / "fid-manifest.json").resolve())
    with pytest.raises(ValueError, match="outside the visual directory"):
        fid.recompute(visual, visual / "fid.json", vendor=tmp_path, weights=tmp_path / "w", extractor=extract)
    seen = []
    monkeypatch.setattr(fid, "recompute", lambda visual_dir, **kw: seen.append((visual_dir, kw)) or result)
    assert fid.main(["--visual-dir", str(visual), "--out", str(out), "--vendor", str(tmp_path), "--weights", "w"]) == 0
    assert seen[0][0] == visual and seen[0][1]["out"] == out and "manifest" not in seen[0][1]
    monkeypatch.undo()
    (visual / "preparation.json").write_text(json.dumps({"judged": ["J", "P"]}))
    with pytest.raises(ValueError, match="MANIFEST hash mismatch: preparation.json"):
        fid.recompute(visual, out, vendor=tmp_path, weights=tmp_path / "w", extractor=extract)
    (visual / "fid-manifest.json").write_text(json.dumps({"cases": cases, "systems": ["J", "P", "O6"],
                                                          "judged": ["J", "P", "O6"]}))
    (visual / "MANIFEST.json").write_text(json.dumps({"files": {n: fid.file_sha256(visual / n)
                                                                for n in ("fid-manifest.json", "preparation.json")}}))
    with pytest.raises(ValueError, match="different judged sets"):
        fid.recompute(visual, out, vendor=tmp_path, weights=tmp_path / "w", extractor=extract)


@pytest.mark.parametrize("x,y", [(np.ones((1, 4)), np.ones((3, 4))),
                                  (np.ones((3, 4)), np.ones((3, 2))),
                                  (np.full((3, 4), np.nan), np.ones((3, 4)))])
def test_formula_rejects_invalid_features(x, y):
    with pytest.raises(ValueError):
        fid.frechet_distance(x, y)


def image_entry(tmp_path, name="image.png", color=20, mode="RGB", size=(512, 512)):
    path = tmp_path / name
    Image.new(mode, size, color).save(path)
    return {"path": name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def case(uid, entry, stratum="front"):
    return {"uid": uid, "stratum": stratum,
            "images": {arm: dict(entry) for arm in fid.ARMS}}


@pytest.fixture
def worker(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(fid, "verify_weights", lambda path: fid.WEIGHTS_SHA256)

    def extract(images, **kwargs):
        calls.append((images, kwargs))
        return np.stack([np.full(2048, i / 10) for i in range(len(images))])

    def run(cases, extractor=extract, **extra):
        manifest = tmp_path / "manifest.json"
        manifest.write_text(json.dumps({"cases": cases, **extra}))
        return fid.run_manifest(manifest, tmp_path / "result.json", vendor=tmp_path,
                                weights=tmp_path / "weights", extractor=extractor)

    return run, calls


def test_shared_cohort_counts_failures_and_strata(tmp_path, worker):
    run, calls = worker
    entry = image_entry(tmp_path)
    missing = case("missing", entry)
    del missing["images"]["placement"]
    bad_hash = case("hash", entry)
    bad_hash["images"]["joint"]["sha256"] = "0" * 64
    failed = {**case("failed", entry), "failure": "renderer rejected furniture"}
    no_gt = case("no-gt", entry)
    no_gt["images"]["reference"]["path"] = "absent.png"
    cases = [case("a", entry), case("b", entry), missing, bad_hash, failed, no_gt,
             case("holo", entry, "holo")]
    result = run(cases, config={"resolution": 512}, provenance={"ledger_sha256": "f" * 64})
    front = result["strata"]["front"]
    assert front["counts"] == {"requested": 6, "reference_eligible": 5, "shared_success": 2}
    assert front["shared_uids"] == ["a", "b"]
    assert front["joint_vs_reference"] == pytest.approx(0)
    assert front["placement_vs_reference"] == pytest.approx(0)
    assert result["counts"]["requested"] == 7
    assert result["strata"]["holo"]["joint_vs_reference"] is None
    assert result["strata"]["holo"]["status"] == "insufficient_sample_size"
    assert {f["uid"] for f in result["failures"]} == {"missing", "hash", "failed", "no-gt"}
    assert "conditional" in result["disclosure"].lower()
    assert "joint_vs_reference" not in result  # no pooled/averaged score
    assert len(calls) == 1 and len(calls[0][0]) == 1  # shared GT/arms read once
    assert calls[0][1]["device"] == "cpu"
    assert calls[0][1]["batch_size"] == 8 and calls[0][1]["threads"] == 4
    assert json.loads((tmp_path / "result.json").read_text()) == result


def test_cache_reused_and_integrity_checked(tmp_path, worker):
    run, calls = worker
    cases = [case("a", image_entry(tmp_path)), case("b", image_entry(tmp_path, "b.png", 40))]
    result = run(cases)
    cache = result["feature_cache"]
    assert fid.file_sha256(cache["path"]) == cache["sha256"]
    assert fid.file_sha256(cache["metadata_path"]) == cache["metadata_sha256"]
    with np.load(cache["path"], allow_pickle=False) as stored:
        assert stored["features"].shape == (2, 2048)
    run(cases)
    assert len(calls) == 1
    Path(cache["path"]).write_bytes(b"tampered")
    with pytest.raises(ValueError, match="cache"):
        run(cases)


@pytest.mark.parametrize("mode,size", [("RGBA", (512, 512)), ("L", (512, 512)), ("RGB", (256, 512))])
def test_no_conversion_or_resize_of_invalid_inputs(tmp_path, worker, mode, size):
    run, calls = worker
    result = run([case("bad", image_entry(tmp_path, mode=mode, size=size))])
    assert result["counts"]["shared_success"] == 0
    assert result["counts"]["requested"] == 1
    assert len(result["failures"]) == 3
    assert not calls


def test_hash_reverified_even_with_existing_cache(tmp_path, worker):
    run, calls = worker
    entry = image_entry(tmp_path)
    cases = [case("a", entry), case("b", entry)]
    run(cases)
    image_entry(tmp_path, color=90)
    result = run(cases)
    assert result["counts"]["shared_success"] == 0
    assert len(calls) == 1
    assert all("hash" in f["reason"] for f in result["failures"])


@pytest.mark.parametrize("bad", [np.zeros((1, 64)), np.full((1, 2048), np.nan)])
def test_invalid_extracted_features_are_failures(tmp_path, worker, bad):
    run, _ = worker
    result = run([case("a", image_entry(tmp_path))], extractor=lambda *a, **kw: bad)
    assert result["counts"]["shared_success"] == 0
    assert result["strata"]["front"]["joint_vs_reference"] is None
    assert result["failures"]


def test_empty_and_singleton_have_no_score(tmp_path, worker):
    run, _ = worker
    assert run([])["strata"] == {}
    result = run([case("a", image_entry(tmp_path))])
    assert result["strata"]["front"]["joint_vs_reference"] is None


def test_duplicate_uid_rejected(tmp_path, worker):
    run, _ = worker
    row = case("a", image_entry(tmp_path))
    with pytest.raises(ValueError, match="duplicate"):
        run([row, row])


def test_weight_filename_is_not_trusted(tmp_path):
    spoof = tmp_path / "pt_inception-2015-12-05-6726825d.pth"
    spoof.write_bytes(b"untrusted weights")
    with pytest.raises(ValueError, match="weights.*sha256"):
        fid.verify_weights(spoof)
    assert fid.WEIGHTS_SHA256 == "6726825d0af5f729cebd5821db510b11b1cfad8faad88a03f1befd49fb9129b2"


def test_import_and_help_are_ml_lazy():
    # comparison.__init__ already imports NumPy through contract; fid adds none.
    code = ("import sys; import fastfill.comparison; before=set(sys.modules); "
            "from fastfill.comparison import fid; "
            "assert not any(k in set(sys.modules)-before for k in "
            "['torch','torchvision','numpy','scipy','PIL'])")
    subprocess.run([sys.executable, "-c", code], check=True)
    result = subprocess.run([sys.executable, "-m", "fastfill.comparison.fid", "--help"],
                            capture_output=True, text=True, check=True)
    assert all(flag in result.stdout for flag in ["--manifest", "--out", "--vendor", "--weights",
                                                  "--batch-size", "--threads", "--device"])


def test_actual_extraction_boundary_with_mock_model(tmp_path, monkeypatch):
    import torch

    entry = image_entry(tmp_path, color=255)
    ref = fid.ImageRef(str(tmp_path / entry["path"]), entry["sha256"])
    seen = []

    class Model(torch.nn.Module):
        def forward(self, inputs):
            assert not torch.is_grad_enabled()
            seen.append(inputs.detach().clone())
            return [inputs.mean((1, 2, 3))[:, None, None, None].expand(-1, 2048, 1, 1)]

    monkeypatch.setattr(fid, "_load_inception", lambda *args: Model())
    values = fid.extract_features([ref] * 3, vendor=tmp_path, weights=tmp_path / "unused",
                                  batch_size=2, threads=1)
    assert values.shape == (3, 2048)
    assert [len(batch) for batch in seen] == [2, 1]
    assert seen[0].shape == (2, 3, 512, 512)
    assert seen[0].min() >= 0 and seen[0].max() <= 1
    assert np.isfinite(values).all()


def test_local_weight_loader_never_calls_url_loader(tmp_path, monkeypatch):
    import torch

    package = tmp_path / "pytorch_fid"
    package.mkdir()
    dist = tmp_path / "pytorch_fid-0.3.0.dist-info"
    dist.mkdir()
    (dist / "METADATA").write_text("Name: pytorch-fid\nVersion: 0.3.0\n")
    (package / "inception.py").write_text(
        "FID_WEIGHTS_URL = 'official-test-url'\n"
        "def load_state_dict_from_url(*a, **kw):\n"
        "    raise AssertionError('network loader called')\n"
        "def InceptionV3(blocks, **kw):\n"
        "    return blocks, kw, load_state_dict_from_url(FID_WEIGHTS_URL, progress=True)\n")
    weights = tmp_path / "weights"
    weights.write_bytes(b"test local checkpoint")
    monkeypatch.setattr(fid, "WEIGHTS_SHA256", fid.file_sha256(weights))
    loaded = []

    def load(stream, **kwargs):
        loaded.append((stream.read(), kwargs))
        return {"test": True}

    monkeypatch.setattr(torch, "load", load)
    blocks, settings, state = fid._load_inception(tmp_path, weights)
    assert blocks == [3] and settings["use_fid_inception"] is True
    assert settings["normalize_input"] and settings["resize_input"]
    assert state == {"test": True}
    assert loaded == [(b"test local checkpoint", {"map_location": "cpu", "weights_only": True})]


@pytest.mark.parametrize("entry", [{}, {"path": "missing.png"},
                                    {"path": "missing.png", "sha256": "short"}])
def test_missing_hash_is_not_inferred(tmp_path, worker, entry):
    run, calls = worker
    result = run([case("bad", entry)])
    assert result["counts"]["requested"] == 1
    assert result["counts"]["shared_success"] == 0
    assert len(result["failures"]) == 3 and not calls


def test_frozen_denominator_and_missing_stratum_rejected(tmp_path, worker):
    run, _ = worker
    entry = image_entry(tmp_path)
    with pytest.raises(ValueError, match="all frozen_uids"):
        run([case("a", entry)], frozen_uids=["a", "omitted"])
    with pytest.raises(ValueError, match="stratum"):
        run([{"uid": "a", "images": {}}])


def test_failure_during_extraction_recorded(tmp_path, worker):
    run, _ = worker

    def broken(*args, **kwargs):
        raise RuntimeError("mock model unavailable")

    result = run([case("a", image_entry(tmp_path))], extractor=broken)
    assert "mock model unavailable" in result["failures"][0]["reason"]


def test_image_change_during_extraction_is_rejected(tmp_path, worker):
    run, _ = worker

    def changed(images, **kwargs):
        image_entry(tmp_path, color=10)
        return np.zeros((len(images), 2048))

    result = run([case("a", image_entry(tmp_path))], extractor=changed)
    assert "hash changed" in result["failures"][0]["reason"]
    assert result["feature_cache"] is None


def test_cli_writes_empty_results(tmp_path, monkeypatch):
    monkeypatch.setattr(fid, "verify_weights", lambda path: fid.WEIGHTS_SHA256)
    manifest = tmp_path / "input.json"
    manifest.write_text('{"featurecases": []}')
    assert fid.main(["--manifest", str(manifest), "--out", str(tmp_path / "out.json"),
                     "--vendor", str(tmp_path), "--weights", str(tmp_path / "weights")]) == 0
    assert json.loads((tmp_path / "out.json").read_text())["counts"]["requested"] == 0
    with pytest.raises(SystemExit) as exc:
        fid.main(["--manifest", str(manifest), "--out", str(tmp_path / "out.json"),
                  "--vendor", str(tmp_path), "--weights", "unused", "--batch-size", "0"])
    assert exc.value.code == 2


@pytest.mark.parametrize("target", ["manifest.json", "image.png", "result.features.npz"])
def test_outputs_cannot_overwrite_inputs(tmp_path, monkeypatch, target):
    monkeypatch.setattr(fid, "verify_weights", lambda path: fid.WEIGHTS_SHA256)
    entry = image_entry(tmp_path)
    if target == "result.features.npz":
        (tmp_path / target).write_bytes((tmp_path / "image.png").read_bytes())
        entry = {**entry, "path": target}
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"cases": [case("a", entry)]}))
    output = tmp_path / ("result.json" if target == "result.features.npz" else target)
    with pytest.raises(ValueError, match="overwrite"):
        fid.run_manifest(manifest, output, vendor=tmp_path, weights=tmp_path / "unused")
