"""Offline driver contracts: real orchestration/files, synthetic Popen boundary."""
import hashlib
import importlib
import json
from pathlib import Path
import shlex
import shutil
import signal
import socket
import struct
import subprocess
from unittest.mock import Mock
import zlib

import pytest


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path):
    return json.loads(path.read_text())


def png(path, color):
    def chunk(kind, data):
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))

    path.parent.mkdir(parents=True, exist_ok=True)
    raw = (b"\0" + bytes(color) * 512) * 512
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", 512, 512, 8, 2, 0, 0, 0))
                     + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))
    return {"path": str(path), "sha256": sha256(path)}


def seal(root, files=None):
    manifest = {"files": files if files is not None else {
        path.relative_to(root).as_posix(): sha256(path)
        for path in root.rglob("*")
        if path.is_file() and path.name != "MANIFEST.json"
    }}
    write_json(root / "MANIFEST.json", manifest)
    return manifest


@pytest.fixture(autouse=True)
def no_external_activity(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("unexpected real subprocess/network activity")

    monkeypatch.setattr(subprocess, "Popen", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket.socket, "connect_ex", forbidden)
    monkeypatch.setattr(importlib.import_module("os"), "killpg", forbidden)


@pytest.fixture
def overnight():
    return importlib.import_module("fastfill.comparison.overnight")


@pytest.fixture
def config(tmp_path):
    code = tmp_path / "code"
    write_json(code / "fastfill/comparison/__init__.py", {})
    for name in ("judge_batch.py", "visual_pipeline.py", "fid.py"):
        (code / "fastfill/comparison" / name).write_text("# synthetic code\n")
    reference = tmp_path / "reference"
    write_json(reference / "reference-cases.jsonl", {"uid": "room-1", "stratum": "bedroom"})
    seal(reference)
    vendor = tmp_path / "vendor"
    vendor.mkdir()
    (vendor / "inception.py").write_text("# synthetic FID vendor\n")
    weights = tmp_path / "weights.pt"
    weights.write_bytes(b"synthetic weights")
    blender = tmp_path / "blender"
    blender.write_text("#!/bin/sh\nexit 99\n")
    blender.chmod(0o700)
    python = tmp_path / "python"
    python.write_text("#!/bin/sh\nexit 99\n")
    python.chmod(0o700)
    return dict(
        ssh_host="fixture-host", remote_root="/remote/project",
        remote_run_root="/remote/project/run", geometry_job_id="103",
        prerequisite_job_ids=["101", "102"], local_code_root=str(code),
        local_python=str(python), output_root=str(tmp_path / "output"),
        reference_data=str(reference), blender=str(blender), vendor=str(vendor),
        weights=str(weights), remote_api_code="/remote/project/api-code",
        api_client_path="/remote/project/server_api_judge.py", api_client_sha256="c" * 64, receipt_paths=["receipts/geometry.json"],
        remote_env_file="/home/u6uv/shanliantian.u6uv/.fastfill_api.env",
        poll_seconds=60, total_timeout_seconds=180,
        render_timeout_seconds=10, api_timeout_seconds=180,
    )


def load(overnight, tmp_path, config):
    path = tmp_path / "config.json"
    write_json(path, config)
    return overnight.load_config(path)


def test_load_config_defaults_and_no_output_side_effects(overnight, tmp_path, config):
    value = {key: item for key, item in config.items() if key != "poll_seconds"}
    result = load(overnight, tmp_path, value)
    assert isinstance(result, dict)
    assert result["remote_python"] == "python3"
    assert result["remote_env_file"] == value["remote_env_file"]
    assert result["poll_seconds"] == 60
    assert result["receipt_paths"] == value["receipt_paths"]
    assert not Path(value["output_root"]).exists()


@pytest.mark.parametrize("field,limit", [
    ("total_timeout_seconds", 432000),
    ("render_timeout_seconds", 86400),
    ("api_timeout_seconds", 36000),
])
@pytest.mark.parametrize("bad", [0, -1, True, float("nan"), float("inf"), "over"])
def test_config_rejects_invalid_timeout(overnight, tmp_path, config, field, limit, bad):
    value = limit + 1 if bad == "over" else bad
    with pytest.raises(ValueError):
        load(overnight, tmp_path, {**config, field: value})


def test_queue_watchdog_allows_five_days_without_extending_execution_caps(overnight, tmp_path, config):
    result = load(overnight, tmp_path, {**config, "total_timeout_seconds": 432000,
        "render_timeout_seconds": 86400, "api_timeout_seconds": 36000})
    assert result["total_timeout_seconds"] == 432000
    assert result["render_timeout_seconds"] == 86400
    assert result["api_timeout_seconds"] == 36000


def test_existing_control_socket_used_for_ssh_and_rsync(overnight, config):
    path = "/tmp/fastfill-private/control.sock"
    driver = overnight.Driver({**config, "ssh_control_path": path})
    driver.command = Mock(return_value="ok")
    assert driver.ssh("true") == "ok"
    argv = driver.command.call_args.args[0]
    assert "ControlPath=" + path in argv and "ControlMaster=no" in argv
    driver.rsync("fixture-host:/remote/input", "/tmp/local-output")
    argv = driver.command.call_args.args[0]
    transport = argv[argv.index("-e") + 1]
    assert "ControlPath=" + path in transport and "ControlMaster=no" in transport


@pytest.mark.parametrize("path", ["relative.sock", "/tmp/../unsafe.sock", "/tmp/%C.sock", "/tmp/a\nsock"])
def test_invalid_control_socket_path_rejected(overnight, config, path):
    with pytest.raises(ValueError):
        overnight.validate_config({**config, "ssh_control_path": path})


@pytest.mark.parametrize("field", ["ssh_host", "remote_run_root", "geometry_job_id",
                                  "local_code_root", "output_root", "reference_data"])
def test_config_rejects_missing_required_field(overnight, tmp_path, config, field):
    with pytest.raises(ValueError):
        load(overnight, tmp_path, {k: v for k, v in config.items() if k != field})


@pytest.mark.parametrize("receipt", ["../outside.json", "/absolute.json", "a/../../escape"])
def test_config_rejects_escaping_receipts(overnight, tmp_path, config, receipt):
    with pytest.raises(ValueError):
        load(overnight, tmp_path, {**config, "receipt_paths": [receipt]})


def test_parse_sacct_filters_steps_and_unrequested_jobs(overnight):
    text = ("JobIDRaw|State|ExitCode\n"
            "101|COMPLETED|0:0\n101.batch|FAILED|1:0\n"
            "102|RUNNING|0:0\n102.extern|COMPLETED|0:0\n"
            "103|FAILED|2:0\n999|COMPLETED|0:0\n\n")
    assert overnight.parse_sacct(text, ["101", "102", "103", "104"]) == {
        "101": {"state": "COMPLETED", "exit_code": "0:0"},
        "102": {"state": "RUNNING", "exit_code": "0:0"},
        "103": {"state": "FAILED", "exit_code": "2:0"},
    }


def test_parse_sacct_empty_is_not_success(overnight):
    assert overnight.parse_sacct("\n", ["103"]) == {}


@pytest.fixture
def manifested(tmp_path):
    root = tmp_path / "manifested"
    write_json(root / "nested/payload.json", {"synthetic": True})
    seal(root)
    return root


def test_manifest_validates_relative_files_and_pinned_digest(overnight, manifested):
    expected = sha256(manifested / "MANIFEST.json")
    overnight.verify_manifest(manifested)
    overnight.verify_manifest(manifested, expected=expected)


@pytest.mark.parametrize("mutation", ["changed", "missing", "manifest_digest"])
def test_manifest_rejects_corruption(overnight, manifested, mutation):
    expected = sha256(manifested / "MANIFEST.json")
    if mutation == "changed":
        (manifested / "nested/payload.json").write_bytes(b"tampered")
    elif mutation == "missing":
        (manifested / "nested/payload.json").unlink()
    else:
        expected = "0" * 64
    with pytest.raises((ValueError, FileNotFoundError)):
        overnight.verify_manifest(manifested, expected=expected)


@pytest.mark.parametrize("kind", ["traversal", "absolute", "file_symlink",
                                 "directory_symlink", "manifest_symlink"])
def test_manifest_rejects_escape_even_when_hash_matches(overnight, tmp_path, manifested, kind):
    outside = tmp_path / "outside"
    write_json(outside / "payload.json", {"private": True})
    payload = outside / "payload.json"
    if kind == "manifest_symlink":
        original = (manifested / "MANIFEST.json").read_bytes()
        (outside / "MANIFEST.json").write_bytes(original)
        (manifested / "MANIFEST.json").unlink()
        (manifested / "MANIFEST.json").symlink_to(outside / "MANIFEST.json")
    else:
        name = {"traversal": "../outside/payload.json", "absolute": str(payload),
                "file_symlink": "linked.json", "directory_symlink": "linked/payload.json"}[kind]
        if kind == "file_symlink":
            (manifested / name).symlink_to(payload)
        elif kind == "directory_symlink":
            (manifested / "linked").symlink_to(outside, target_is_directory=True)
        seal(manifested, {name: sha256(payload)})
    with pytest.raises(ValueError):
        overnight.verify_manifest(manifested)


def test_driver_claims_output_exclusively(overnight, tmp_path, config, monkeypatch):
    validated = load(overnight, tmp_path, config)
    monkeypatch.setattr(overnight.Driver, "command",
                        lambda *a, **k: pytest.fail("duplicate run issued a command"))
    overnight.Driver(validated)
    out = Path(config["output_root"])
    assert out.is_dir()
    marker = out / "owner.txt"
    marker.write_text("first owner\n")
    with pytest.raises((ValueError, FileExistsError)):
        overnight.Driver(validated)
    assert marker.read_text() == "first owner\n"


def test_run_delegates_to_execute(overnight, config, monkeypatch):
    expected = {"status": "synthetic-endstate"}
    driver = Mock()
    driver.execute.return_value = expected
    factory = Mock(return_value=driver)
    monkeypatch.setattr(overnight, "Driver", factory)
    assert overnight.run(config) == expected
    factory.assert_called_once_with(config)
    driver.execute.assert_called_once_with()


def test_command_uses_mocked_popen(overnight, tmp_path, config, monkeypatch):
    driver = overnight.Driver(load(overnight, tmp_path, config))
    process = Mock(returncode=0, pid=12345)
    process.communicate.return_value = ("synthetic stdout\n", "")
    process.poll.return_value = 0
    process.wait.return_value = 0
    process.__enter__ = Mock(return_value=process)
    process.__exit__ = Mock(return_value=False)
    popen = Mock(return_value=process)
    monkeypatch.setattr(overnight.subprocess, "Popen", popen)
    assert driver.command(["synthetic-command", "argument"], timeout=2,
                          input_text="synthetic stdin\n") == "synthetic stdout\n"
    popen.assert_called_once()
    assert popen.call_args.args[0] == ["synthetic-command", "argument"]
    assert popen.call_args.kwargs.get("shell", False) is False
    process.communicate.assert_called_once()
    assert process.communicate.call_args.kwargs["timeout"] <= 2
    assert process.communicate.call_args.kwargs["input"] == "synthetic stdin\n"
    assert popen.call_args.kwargs["start_new_session"] is True
    receipt = read_json(next(driver.out.glob("receipts/*.json")))
    assert receipt["status"] == "finished" and receipt["returncode"] == 0
    assert driver.child is None


class FakeClock:
    def __init__(self):
        self.now = 0.0
        self.sleeps = []

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        assert 0 <= seconds <= 60
        self.sleeps.append(seconds)
        self.now += max(seconds, 0.1)
        assert len(self.sleeps) < 100, "driver failed to enforce total deadline"


class Commands:
    """Model command outputs and local artifacts, never execute shell text."""

    def __init__(self, root, config, scenario="completed"):
        self.root, self.config, self.scenario = root, config, scenario
        self.calls, self.events = [], []
        self.upload, self.upload_hashes = None, None
        evaluation = root / "eval-data-v1"
        write_json(evaluation / "rooms.jsonl", {"uid": "room-1"})
        seal(evaluation)
        geometry = root / "geometry-v1"
        write_json(geometry / "summary.json", {
            "data_manifest_sha256": sha256(evaluation / "MANIFEST.json")})
        for arm in ("joint", "placement"):
            write_json(geometry / f"{arm}-geometry.jsonl", {"uid": "room-1", "arm": arm})
        seal(geometry)
        seal(root / "main-data-v1", {})
        write_json(root / "receipts/geometry.json", {"status": "completed"})
        if scenario == "hash_mismatch":
            (geometry / "joint-geometry.jsonl").write_bytes(b"corrupted after sealing")

    def __call__(self, argv, timeout=60, input_text=None):
        argv = [str(item) for item in argv]
        self.calls.append((argv, timeout, input_text))
        assert 0 < timeout <= self.config["total_timeout_seconds"]
        assert len(self.calls) < 100, "unbounded polling or command retry"
        command = shlex.join(argv) + "\n" + (input_text or "")
        if "sacct" in command:
            self.events.append("sacct")
            if "launch" in self.events:
                return "4242|COMPLETED|0:0\n"
            state = {"terminal_failure": "FAILED", "dependency_never": "PENDING",
                     "timeout": "RUNNING"}.get(self.scenario, "COMPLETED")
            code = "1:0" if state == "FAILED" else "0:0"
            return f"101|COMPLETED|0:0\n102|COMPLETED|0:0\n103|{state}|{code}\n"
        if "squeue" in command:
            self.events.append("squeue")
            return "103|PENDING|DependencyNeverSatisfied\n" if self.scenario == "dependency_never" else ""
        if Path(argv[0]).name == "rsync":
            return self.rsync(argv)
        if "fastfill.comparison.visual_pipeline" in argv:
            return self.prepare(argv)
        if "fastfill.comparison.fid" in argv:
            self.events.append("fid")
            write_json(Path(argv[argv.index("--out") + 1]), {"status": "completed", "strata": {}, "counts": {"requested": 1, "reference_eligible": 1, "shared_success": 1}})
            return "{}\n"
        if "sbatch" in command:
            self.events.append("launch")
            assert "judge_batch.py" in command or "fastfill.comparison.judge_batch" in command
            assert "job.json" in command and "--gres=none" in command
            assert "--no-requeue" in command and "--deadline=" in command
            assert "nohup" not in command
            if self.scenario == "ambiguous_launch":
                raise subprocess.CalledProcessError(255, argv)
            return "4242\n"
        if argv[0] == "ssh" and "sha256sum -- " in argv[-1]:
            if "judge_batch.py" in argv[-1]:
                wrapper = Path(self.config["local_code_root"]) / "fastfill/comparison/judge_batch.py"
                return (sha256(wrapper) + "  " + self.config["remote_api_code"] + "/fastfill/comparison/judge_batch.py\n"
                        + "c" * 64 + "  " + self.config["api_client_path"] + "\n")
            names = ["main-data-v1/MANIFEST.json", *self.config["receipt_paths"]]
            return "".join(sha256(self.root / name) + "  " + name + "\n" for name in names)
        if "chmod" in command:
            return ""
        if "sha256" in command.lower() or "hashlib" in command:
            self.events.append("verify_upload")
            return "verified\n"
        if "mkdir" in command:
            self.events.append("mkdir")
            return ""
        if argv[0] == "ssh" and "launch" in self.events:
            self.events.append("completion")
            return json.dumps({"status": "completed", "exit_code": 0, "pid": 4242}) + "\n"
        pytest.fail(f"unmodelled command: {argv!r}")

    def rsync(self, argv):
        source, target = argv[-2:]
        if ":" in target:
            self.events.append("upload")
            self.upload = Path(source.rstrip("/"))
            assert self.upload.name == "api-upload"
            self.upload_hashes = {str(p.relative_to(self.upload)): sha256(p)
                                  for p in self.upload.rglob("*") if p.is_file()}
            assert "api-bundle/manifest.json" in self.upload_hashes
            return ""
        assert ":" in source, "expected an SSH rsync source"
        remote = source.split(":", 1)[1].rstrip("/")
        destination = Path(target)
        if "launch" in self.events:
            self.events.append("results")
            destination.mkdir(parents=True, exist_ok=True)
            write_json(destination / "results/summary.json", {"status": "completed", "counts": {"selected": 2, "runnable": 2, "launched": 2, "success": 2, "failed": 0}})
            write_json(destination / "receipts/request.result.json", {"status": "completed"})
            (destination / "tables.md").write_text("synthetic result table\n")
            assert source.endswith("/"), "retrieve entire results directory"
            return ""
        self.events.append("download")
        relative = Path(remote).relative_to(self.config["remote_run_root"])
        origin = self.root / relative
        assert origin.exists(), f"unexpected download: {relative}"
        if origin.is_dir():
            destination.mkdir(parents=True, exist_ok=True)
            shutil.copytree(origin, destination, dirs_exist_ok=True)
        else:
            if target.endswith("/") or destination.is_dir():
                destination.mkdir(parents=True, exist_ok=True)
                destination = destination / origin.name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(origin, destination)
        return ""

    def prepare(self, argv):
        self.events.append("prepare")
        assert "prepare" in argv and "--eval-data" in argv
        out = Path(argv[argv.index("--out") + 1])
        api = out / "api-bundle"
        write_json(api / "image.json", {"fake": "image bytes"})
        (api / "prompt.txt").write_text("synthetic prompt")
        cases = [{"id": arm, "prompt_file": "prompt.txt", "prompt_sha256": sha256(api / "prompt.txt"),
                  "images": [{"path": "image.json", "sha256": sha256(api / "image.json")}]} for arm in ("joint", "placement")]
        write_json(api / "manifest.json", {"schema_version": 1,
                   "expected_model": "gpt-6.1-sol", "cases": cases})
        write_json(out / "api-mapping.json", {arm: {"uid": "room-1", "arm": arm, "stratum": "bedroom", "group": "g"} for arm in ("joint", "placement")})
        write_json(out / "api-failures.json", [])
        write_json(out / "fid-manifest.json", {"cases": []})
        write_json(out / "preparation.json", {"status": "prepared", "counts": {"frozen": 1, "rendered": 1, "failed": 0, "gpt_selected": 1, "api_cases": 2}})
        seal(out)
        return json.dumps({"status": "prepared", "counts": {"frozen": 1, "rendered": 1, "failed": 0, "gpt_selected": 1, "api_cases": 2}}) + "\n"


def simulated_run(overnight, tmp_path, config, monkeypatch, scenario):
    validated = load(overnight, tmp_path, config)
    commands = Commands(tmp_path / "synthetic-remote", validated, scenario)
    clock = FakeClock()
    monkeypatch.setattr(overnight.time, "monotonic", clock.monotonic)
    monkeypatch.setattr(overnight.time, "sleep", clock.sleep)

    def command(driver, argv, timeout=60, input_text=None):
        return commands(argv, timeout=timeout, input_text=input_text)

    monkeypatch.setattr(overnight.Driver, "command", command)
    endstate = overnight.run(validated)
    assert isinstance(endstate, dict)
    return endstate, commands, clock


def test_pipeline_happy_path_order_and_one_remote_launch(overnight, tmp_path, config, monkeypatch):
    endstate, commands, _ = simulated_run(overnight, tmp_path, config, monkeypatch, "completed")
    stages = ["sacct", "download", "prepare", "fid", "mkdir", "upload",
              "verify_upload", "launch", "completion", "results"]
    assert all(stage in commands.events for stage in stages), endstate
    assert endstate["status"] == "completed", endstate
    positions = [commands.events.index(stage) for stage in stages]
    assert positions == sorted(positions)
    assert commands.events.count("launch") == 1
    downloads = "\n".join(shlex.join(argv) for argv, _, _ in commands.calls
                          if Path(argv[0]).name == "rsync" and ":" in argv[-2])
    for name in ("geometry-v1", "eval-data-v1", "main-data-v1/MANIFEST.json",
                 "receipts/geometry.json"):
        assert name in downloads
    assert commands.upload_hashes == {
        str(p.relative_to(commands.upload)): sha256(p)
        for p in commands.upload.rglob("*") if p.is_file()
    }, "the sealed upload changed after transmission"
    out = Path(config["output_root"])
    assert list(out.rglob("request.result.json")), "individual remote results were not retained"
    assert list(out.rglob("tables.md")), "remote aggregate artifacts were not retained"


@pytest.mark.parametrize("scenario", ["terminal_failure", "dependency_never", "timeout", "hash_mismatch"])
def test_pipeline_stops_before_downstream_work(overnight, tmp_path, config, monkeypatch, scenario):
    endstate, commands, clock = simulated_run(overnight, tmp_path, config, monkeypatch, scenario)
    assert "sacct" in commands.events
    assert not {"prepare", "fid", "upload", "launch", "results"}.intersection(commands.events)
    assert endstate, "failure must produce an endstate receipt"
    if scenario != "hash_mismatch":
        assert "download" not in commands.events
    if scenario == "dependency_never":
        assert "squeue" in commands.events
        assert clock.now < config["total_timeout_seconds"], "dependency failure must fail promptly"
    if scenario == "timeout":
        assert clock.sleeps
        assert config["total_timeout_seconds"] <= clock.now <= config["total_timeout_seconds"] + config["poll_seconds"]


def test_existing_output_run_preserves_files_and_issues_no_commands(overnight, tmp_path, config, monkeypatch):
    validated = load(overnight, tmp_path, config)
    out = Path(config["output_root"])
    write_json(out / "endstate.json", {"status": "existing"})
    original = (out / "endstate.json").read_bytes()
    calls = []

    def forbidden(*args, **kwargs):
        calls.append(args)
        pytest.fail("existing output triggered external activity")

    monkeypatch.setattr(overnight.Driver, "command", forbidden)
    with pytest.raises((ValueError, FileExistsError)):
        overnight.run(validated)
    assert calls == []
    assert (out / "endstate.json").read_bytes() == original


@pytest.mark.parametrize('poll', [0, 1, 59, 301, True, float('nan')])
def test_manual_polling_minimum(overnight, tmp_path, config, poll):
    with pytest.raises(ValueError):
        load(overnight, tmp_path, {**config, 'poll_seconds': poll})


def test_ambiguous_submission_never_retried(overnight, tmp_path, config, monkeypatch):
    result, commands, _ = simulated_run(overnight, tmp_path, config, monkeypatch, 'ambiguous_launch')
    assert result['status'] == 'partial'
    assert result['api_submission_attempted'] is True
    assert commands.events.count('launch') == 1
    assert 'results' in commands.events


def test_frozen_code_mutation_fails_before_stage(overnight, tmp_path, config):
    driver = overnight.Driver(load(overnight, tmp_path, config))
    driver.freeze()
    (Path(config['local_code_root']) / 'fastfill/comparison/fid.py').write_text('changed')
    with pytest.raises(ValueError, match='frozen tree hash mismatch'):
        driver.advance('render')


def test_command_timeout_kills_only_owned_process_group(overnight, tmp_path, config, monkeypatch):
    driver = overnight.Driver(load(overnight, tmp_path, config))
    process = Mock(returncode=None, pid=6789)
    process.communicate.side_effect = subprocess.TimeoutExpired('synthetic', 1)
    process.poll.return_value = None
    monkeypatch.setattr(overnight.subprocess, 'Popen', Mock(return_value=process))
    kill = Mock()
    monkeypatch.setattr(overnight.os, 'killpg', kill)
    with pytest.raises(subprocess.TimeoutExpired):
        driver.command(['synthetic'], timeout=1)
    kill.assert_called_once_with(6789, overnight.signal.SIGKILL)
    process.wait.assert_called_once()
    assert driver.child is None
    assert json.loads(next((driver.out / 'receipts').glob('*.json')).read_text())['status'] == 'failed'


def test_interruption_retains_partial_endstate(overnight, tmp_path, config, monkeypatch):
    monkeypatch.setattr(overnight.Driver, 'poll_geometry', Mock(side_effect=KeyboardInterrupt('interrupted')))
    result = overnight.run(load(overnight, tmp_path, config))
    assert result['status'] == 'partial' and result['reason'] == 'interrupted'
    assert result['failures'][0]['stage'] == 'jobs'
    assert read_result(config, 'endstate.json') == result


def read_result(config, name):
    return json.loads((Path(config['output_root']) / name).read_text())


def test_read_only_ssh_retries_bounded(overnight, tmp_path, config, monkeypatch):
    clock = FakeClock()
    monkeypatch.setattr(overnight.time, 'monotonic', clock.monotonic)
    monkeypatch.setattr(overnight.time, 'sleep', clock.sleep)
    driver = overnight.Driver(load(overnight, tmp_path, {**config, 'read_retries': 1}))
    command = Mock(side_effect=[subprocess.CalledProcessError(255, 'ssh'), 'ok'])
    monkeypatch.setattr(driver, 'command', command)
    assert driver.ssh('sacct', read_only=True) == 'ok'
    assert command.call_count == 2 and clock.now == 60


def test_cli_returns_partial_exit_without_launch(overnight, tmp_path, config, monkeypatch, capsys):
    path = tmp_path / 'config.json'
    write_json(path, config)
    monkeypatch.setattr(overnight, 'run', lambda value: {'status': 'partial', 'counts': {}})
    assert overnight.main(['--config', str(path)]) == 1
    assert json.loads(capsys.readouterr().out)['status'] == 'partial'


def test_real_visual_fid_and_judge_accept_driver_upload(overnight, tmp_path, config, monkeypatch):
    """Real producer/consumer contracts; only Blender, features and API IO mocked."""
    import numpy as np
    from fastfill.comparison import visual_pipeline, fid, judge_batch
    from fastfill.comparison.tests import test_visual_pipeline as visuals
    from fastfill.comparison.tests import test_judge_batch as judges

    ref, geo, evaluation, _ = visuals.fixture(tmp_path / 'real-inputs', n=2)
    driver = overnight.Driver(load(overnight, tmp_path, config))
    calls = []
    monkeypatch.setattr(visual_pipeline.subprocess, 'run', visuals.fake_blender(calls))
    prepared = visual_pipeline.prepare(ref, geo, driver.out / 'visual', '/mock/blender', eval_data=evaluation)
    assert prepared['counts']['api_cases'] == 4
    monkeypatch.setattr(fid, 'WEIGHTS_SHA256', sha256(Path(config['weights'])))
    features = lambda images, **kwargs: np.zeros((len(images), fid.FEATURE_DIM))
    measured = fid.run_manifest(driver.out / 'visual/fid-manifest.json', driver.out / 'fid.json',
        vendor=config['vendor'], weights=config['weights'], extractor=features)
    assert measured['counts']['shared_success'] == 2
    upload = driver.bundle()
    manifest = json.loads((upload / 'MANIFEST.json').read_text())
    for name in ('api-mapping.json', 'api-failures.json', 'api-bundle/manifest.json', 'api-bundle/provenance.json'):
        assert manifest['files'][name] == sha256(upload / name)
    calls = []
    monkeypatch.setattr(judge_batch.subprocess, 'run', judges.fake_client(calls))
    measured = judge_batch.evaluate(upload, driver.out / 'judge-integration', judges.CLIENT, expected_total=4)
    assert measured['counts']['success'] == 4 and len(calls) == 4
    assert measured['geometry_manifest_sha256'] == sha256(geo / 'MANIFEST.json')
    assert measured['visual_manifest_sha256'] == sha256(upload / 'MANIFEST.json')
