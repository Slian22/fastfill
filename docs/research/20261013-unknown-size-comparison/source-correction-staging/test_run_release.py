"""Synthetic, local-only tests; never open production datasets or submit jobs."""
import hashlib
import importlib.util
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import Mock, patch

SPEC = importlib.util.spec_from_file_location("release_audit", Path(__file__).with_name("run_release.py"))
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)

SOURCES = ["SpatialLM", "IL3D_3dfront", "InteriorGS", "HSSD200", "Structured3D",
           "InternScenes_arkit", "InternScenes_mp3d", "InternScenes_scannet", "InternScenes_3rscan",
           "Scan2CAD", "MultiScan", "OptiScene_holodeck", "IL3D_synthetic", "SAGE-10k",
           "MansionWorld", "InternScenes_gen", "SpatialGen", "SceneSmith"]
ARGS = dict(ir="/historical/ir", out="/historical/release", sources=SOURCES,
            min_objects=1, max_vertices=0, reject_flagged=[], keep_duplicates=False,
            boundary_types=["polygon", "hull"], anchors=["floor", "object"], oob_tol=.1,
            hidden_max=.3, cap=[], source_anchors={}, dev=.05, test=.05, constraint_frac=.3,
            no_rot90=False, field_dropout=.2, desc_dropout=.5)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def row(uid, source="Scan2CAD", flags=None, message="same"):
    return dict(uid=uid, source=source, flags={} if flags is None else flags,
                messages=[{"role": "assistant", "content": message}])


def release(root, rows, args, ir_hashes, code_hashes):
    root.mkdir()
    files = {}
    for split in ("train", "dev", "test"):
        (root / (split + ".jsonl")).write_text("".join(json.dumps(r) + "\n" for r in rows[split]))
    for name in ("dev_rooms", "test_rooms", "dev_constrained_rooms", "test_constrained_rooms"):
        (root / (name + ".jsonl")).write_text("")
    dump(root / "stats.json", {})
    for p in root.iterdir():
        files[p.name] = {"sha256": digest(p), "lines": len(p.read_bytes().splitlines())}
    dump(root / "MANIFEST.json", dict(args=args, files=files, ir_sha256=ir_hashes,
                                      code_sha256=code_hashes, build_start_code_sha256=code_hashes))


class ReleaseTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.code = self.root / "code-final-version"
        (self.code / "fastfill").mkdir(parents=True)
        for name in ("__init__.py", "build.py", "anchors.py", "scene.py", "split.py", "validate.py"):
            (self.code / "fastfill" / name).write_text("# synthetic pinned module\n")
        self.code_hashes = {p.name: digest(p) for p in (self.code / "fastfill").iterdir()}
        self.code_manifest = self.code / "CORRECTION-MANIFEST.json"
        dump(self.code_manifest, {"files": {"fastfill/" + n: h for n, h in self.code_hashes.items()}})
        self.ir = self.root / "ir-final-version"
        self.ir.mkdir()
        for source in SOURCES:
            (self.ir / (source + ".jsonl")).write_text('{"synthetic":true}\n')
        self.ir_hashes = {p.name: digest(p) for p in self.ir.iterdir()}
        self.ir_manifest = self.root / "IR-MANIFEST.json"
        dump(self.ir_manifest, {"files": {n: {"sha256": h, "rows": 1} for n, h in self.ir_hashes.items()}})
        self.rows = {"train": [row("a")], "dev": [row("b", "IL3D_3dfront")], "test": [row("c")]}
        self.old = self.root / "old-release"
        release(self.old, self.rows, ARGS, self.ir_hashes, self.code_hashes)
        self.out = self.root / "new-output"
        self.config = self.root / "config.json"
        self.cfg = dict(code_root=str(self.code), code_manifest_sha256=digest(self.code_manifest),
                        ir_dir=str(self.ir), ir_manifest=str(self.ir_manifest),
                        ir_manifest_sha256=digest(self.ir_manifest), old_release_dir=str(self.old),
                        old_manifest_sha256=digest(self.old / "MANIFEST.json"), output_root=str(self.out))
        dump(self.config, self.cfg)
        self.runner = Mock(side_effect=self.build)

    def build(self, command, cwd, log_path):
        self.assertEqual(Path(cwd), self.code)
        self.assertIn("-I", command)
        self.assertIn("-B", command)
        self.assertTrue(any(x.startswith("pycache_prefix=") for x in command))
        self.assertEqual(command[command.index("--constraint_frac") + 1], "0.3")
        args = {**ARGS, "ir": str(self.ir), "out": str(self.out / "release")}
        release(self.out / "release", self.rows, args, self.ir_hashes, self.code_hashes)
        log_path.write_text("synthetic build\n")

    def run_audit(self):
        return audit.run(self.config, runner=self.runner,
                         version_reader=lambda *a: {"python": "synthetic", "numpy": "synthetic", "shapely": "synthetic"})

    def fail_before_build(self, pattern):
        with self.assertRaisesRegex(audit.AuditError, pattern):
            self.run_audit()
        self.runner.assert_not_called()
        self.assertFalse(self.out.exists())

    def test_bad_ir_hash_fails_before_build(self):
        (self.ir / "Scan2CAD.jsonl").write_text('{"tampered":true}\n')
        self.fail_before_build("sha256")

    def test_bad_code_hash_fails_before_build(self):
        (self.code / "fastfill/build.py").write_text("# tampered\n")
        self.fail_before_build("sha256")

    def test_unpinned_code_file_fails_before_build(self):
        (self.code / "fastfill/extra.py").write_text("# not pinned\n")
        self.fail_before_build("inventory")

    def test_ir_manifest_pin_and_row_count_enforced(self):
        m = json.loads(self.ir_manifest.read_text())
        m["files"]["Scan2CAD.jsonl"]["rows"] = 2
        dump(self.ir_manifest, m)
        self.fail_before_build("sha256")
        self.cfg["ir_manifest_sha256"] = digest(self.ir_manifest)
        dump(self.config, self.cfg)
        self.fail_before_build("rows")

    def test_existing_output_even_empty_is_protected(self):
        self.out.mkdir()
        with self.assertRaisesRegex(audit.AuditError, "exists"):
            self.run_audit()
        self.runner.assert_not_called()
        self.assertEqual(list(self.out.iterdir()), [])

    def test_exact_args_preserved_and_identity_only_pass(self):
        result = self.run_audit()
        self.assertEqual(result["status"], "candidate_complete_identity_preserved")
        self.assertTrue(result["identity_gate_passed"])
        self.assertFalse(result["refit_ready"])
        self.assertFalse(result["full_component_closure_checked"])
        self.assertFalse(result["scientifically_accepted"])
        self.assertEqual(result["counts"]["common"], 3)
        self.assertEqual(result["by_source"]["Scan2CAD"]["messages_unchanged"], 2)
        self.assertEqual(result["runtime"]["python"], "synthetic")
        self.assertEqual(result["effective_build_args"], {**ARGS, "ir": str(self.ir), "out": str(self.out / "release")})

    def test_equal_split_counts_cannot_hide_dev_test_swap(self):
        self.rows = {**self.rows, "dev": [row("c")], "test": [row("b", "IL3D_3dfront")]}
        result = self.run_audit()
        self.assertEqual(result["status"], "candidate_complete_identity_changed")
        self.assertEqual(result["counts"]["moved"], 2)
        self.assertEqual(result["counts"]["old_heldout_moved"], 2)
        self.assertIn("old_heldout_split_migration", result["readiness_blockers"])
        moves = [json.loads(x) for x in (self.out / "moved.jsonl").read_text().splitlines()]
        self.assertEqual({x["uid"] for x in moves}, {"b", "c"})

    def test_old_heldout_to_train_blocks_readiness(self):
        self.rows = {**self.rows, "train": [row("c")], "test": [row("a")]}
        result = self.run_audit()
        self.assertEqual(result["counts"]["old_heldout_to_train"], 1)
        self.assertFalse(result["identity_gate_passed"])

    def test_added_lost_uid_and_source_counts_with_same_flags(self):
        self.rows = {**self.rows, "train": [row("new", "SpatialGen")]}
        result = self.run_audit()
        self.assertEqual(result["counts"]["added"], 1)
        self.assertEqual(result["counts"]["dropped"], 1)
        self.assertEqual(result["by_source"]["Scan2CAD"]["dropped"], 1)
        self.assertEqual(result["by_source"]["SpatialGen"]["added"], 1)
        self.assertEqual(result["by_source"]["Scan2CAD"]["old_by_split"]["train"], 1)
        self.assertEqual(result["by_source"]["Scan2CAD"]["new_by_split"]["train"], 0)
        self.assertEqual(result["counts"]["flags_changed"], 0)
        self.assertEqual(json.loads((self.out / "dropped.jsonl").read_text())["uid"], "a")

    def test_messages_changed_even_when_flags_same(self):
        self.rows = {**self.rows, "train": [row("a", message="corrected dimensions")]}
        result = self.run_audit()
        self.assertEqual(result["counts"]["messages_changed"], 1)
        self.assertEqual(result["counts"]["flags_changed"], 0)
        self.assertEqual(result["by_source"]["Scan2CAD"]["flags_unchanged"], 2)
        self.assertEqual(result["by_source"]["Scan2CAD"]["messages_changed"], 1)

    def test_flags_changed_count_by_source(self):
        self.rows = {**self.rows, "train": [row("a", flags={"tilted": 1})]}
        result = self.run_audit()
        self.assertEqual(result["by_source"]["Scan2CAD"]["flags_changed"], 1)

    def test_source_identity_change_detected(self):
        self.rows = {**self.rows, "train": [row("a", "SpatialGen")]}
        result = self.run_audit()
        self.assertEqual(result["counts"]["source_changed"], 1)
        self.assertFalse(result["identity_gate_passed"])

    def test_duplicate_uid_rejected_even_across_splits(self):
        self.rows = {**self.rows, "test": [row("a")]}
        with self.assertRaisesRegex(audit.AuditError, "duplicate UID"):
            self.run_audit()
        self.assertEqual(json.loads((self.out / "summary.json").read_text())["status"], "failed")

    def test_code_and_ir_rechecked_after_build(self):
        for target in (self.code / "fastfill/build.py", self.ir / "Scan2CAD.jsonl"):
            with self.subTest(target=target):
                original = target.read_bytes()
                self.runner.side_effect = lambda *a: (self.build(*a), target.write_text("changed during build\n"))
                with self.assertRaisesRegex(audit.AuditError, "sha256"):
                    self.run_audit()
                self.assertEqual(json.loads((self.out / "summary.json").read_text())["status"], "failed")
                target.write_bytes(original)
                self.out = self.root / ("next-" + target.name)
                self.cfg["output_root"] = str(self.out)
                dump(self.config, self.cfg)

    def test_old_release_rechecked_after_build(self):
        self.runner.side_effect = lambda *a: (self.build(*a), (self.old / "test.jsonl").write_text("tamper\n"))
        with self.assertRaisesRegex(audit.AuditError, "sha256"):
            self.run_audit()

    def test_output_hash_and_count_must_match(self):
        def corrupt(*args):
            self.build(*args)
            mp = self.out / "release/MANIFEST.json"
            manifest = json.loads(mp.read_text())
            manifest["files"]["train.jsonl"]["lines"] = 123
            dump(mp, manifest)
        self.runner.side_effect = corrupt
        with self.assertRaisesRegex(audit.AuditError, "lines"):
            self.run_audit()

    def test_args_reject_unexpected_new_builder_options(self):
        mp = self.old / "MANIFEST.json"
        m = json.loads(mp.read_text())
        m["args"]["surprise"] = 1
        dump(mp, m)
        self.cfg["old_manifest_sha256"] = digest(mp)
        dump(self.config, self.cfg)
        self.fail_before_build("args")

    def test_pinned_unrelated_prep_change_still_rejected(self):
        module = self.code / "fastfill/build.py"
        module.write_text("# unrelated prep change\n")
        manifest = json.loads(self.code_manifest.read_text())
        manifest["files"]["fastfill/build.py"] = digest(module)
        dump(self.code_manifest, manifest)
        self.cfg["code_manifest_sha256"] = digest(self.code_manifest)
        dump(self.config, self.cfg)
        self.fail_before_build("frozen implementation")

    def test_other_fifteen_source_hashes_must_stay_historical(self):
        raw = self.ir / "HSSD200.jsonl"
        raw.write_text('{"changed":1}\n')
        manifest = json.loads(self.ir_manifest.read_text())
        manifest["files"][raw.name]["sha256"] = digest(raw)
        dump(self.ir_manifest, manifest)
        self.cfg["ir_manifest_sha256"] = digest(self.ir_manifest)
        dump(self.config, self.cfg)
        self.fail_before_build("unchanged source")

    def test_null_row_count_is_not_a_verification_bypass(self):
        m = json.loads(self.ir_manifest.read_text())
        m["files"]["Scan2CAD.jsonl"]["rows"] = None
        dump(self.ir_manifest, m)
        self.cfg["ir_manifest_sha256"] = digest(self.ir_manifest)
        dump(self.config, self.cfg)
        self.fail_before_build("rows")

    def test_build_failure_preserves_diagnostic_output(self):
        self.runner.side_effect = subprocess.CalledProcessError(42, "synthetic-build")
        with self.assertRaises(subprocess.CalledProcessError):
            self.run_audit()
        result = json.loads((self.out / "summary.json").read_text())
        self.assertEqual(result["status"], "failed")
        self.assertFalse(result["refit_ready"])

    def test_real_subprocess_entry_and_runtime_receipt(self):
        # Tiny pinned builder exercises isolated runpy/argv/log handling, not production build/data.
        module = self.code / "fastfill/build.py"
        module.write_text("""import json,shutil,sys
from pathlib import Path
root=Path(__file__).resolve().parents[2]
out=Path(sys.argv[sys.argv.index('--out')+1])
shutil.copytree(root/'old-release',out)
p=out/'MANIFEST.json'
m=json.loads(p.read_text())
m['args']['out']=str(out)
m['args']['ir']=sys.argv[sys.argv.index('--ir')+1]
p.write_text(json.dumps(m))
print('synthetic isolated builder completed', flush=True)
""")
        self.code_hashes["build.py"] = digest(module)
        code = json.loads(self.code_manifest.read_text())
        code["files"]["fastfill/build.py"] = digest(module)
        dump(self.code_manifest, code)
        old = self.old / "MANIFEST.json"
        manifest = json.loads(old.read_text())
        manifest.update(code_sha256=self.code_hashes, build_start_code_sha256=self.code_hashes)
        dump(old, manifest)
        self.cfg.update(code_manifest_sha256=digest(self.code_manifest), old_manifest_sha256=digest(old))
        dump(self.config, self.cfg)
        result = audit.run(self.config)
        self.assertEqual(result["status"], "candidate_complete_identity_preserved")
        self.assertIn("synthetic isolated builder completed", (self.out / "build.log").read_text())
        self.assertTrue(result["runtime"]["numpy"])
        self.assertTrue(result["runtime"]["shapely"])

    def test_zero_population_sources_are_explicit(self):
        result = self.run_audit()
        self.assertEqual(set(result["by_source"]), set(SOURCES))
        self.assertEqual(result["by_source"]["HSSD200"]["old_by_split"], dict.fromkeys(("train", "dev", "test"), 0))

    def test_cli_identity_change_has_nonzero_exit(self):
        self.rows = {**self.rows, "dev": [row("c")], "test": [row("b", "IL3D_3dfront")]}
        result = self.run_audit()
        with patch.object(audit, "run", return_value=result), patch.object(audit.sys, "argv", ["run_release.py", "--config", str(self.config)]), redirect_stdout(io.StringIO()):
            self.assertEqual(audit.main(), 2)

    def test_unmanifested_directory_symlink_rejected(self):
        (self.code / "shadow").symlink_to(self.old, target_is_directory=True)
        self.fail_before_build("symlink")


if __name__ == "__main__":
    unittest.main(verbosity=2)
