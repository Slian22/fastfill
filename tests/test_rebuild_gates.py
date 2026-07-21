"""Regression tests for the jq gates embedded in scripts/rebuild_research_v15.sh.

``bash -n`` cannot catch a semantically wrong jq program: the InternScenes
snapshot-contribution gate once iterated the TOP level of
``per_source_layer`` (whose keys are train/heldout/test) instead of the
nested ``{split: {"<source>/<layer>": n}}`` maps, so it always summed to 0
and hard-failed every rebuild. These tests run the exact jq program
extracted from the script against a make_snapshot-shaped fixture.
"""

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "rebuild_research_v15.sh"

# make_snapshot.py sidecar shape: per_source_layer is nested by split name
# (make_snapshot.py builds {name: _source_layer_counts(...) for name in files}).
NESTED_SIDECAR = {
    "per_source_layer": {
        "train": {"internscenes/floor": 100, "m3dlayout/floor": 5},
        "heldout": {"internscenes/floor": 20},
        "test": {"internscenes/floor": 3, "scenesmith_scenes/surface": 7},
    }
}


def _intern_gate_program() -> str:
    text = SCRIPT.read_text(encoding="utf-8")
    match = re.search(r"intern_n=\$\(jq '([^']+)'", text)
    assert match is not None, "intern_n jq gate missing from rebuild script"
    return match.group(1)


def _run_jq(program: str, payload: dict) -> str:
    if shutil.which("jq") is None:
        pytest.skip("jq not installed")
    result = subprocess.run(
        ["jq", program],
        input=json.dumps(payload),
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def test_intern_gate_sums_nested_per_source_layer():
    assert _run_jq(_intern_gate_program(), NESTED_SIDECAR) == "123"


def test_intern_gate_zero_when_source_absent():
    sidecar = {
        "per_source_layer": {
            "train": {"m3dlayout/floor": 5},
            "heldout": {},
            "test": {},
        }
    }
    assert _run_jq(_intern_gate_program(), sidecar) == "0"


def test_intern_gate_zero_on_empty_sidecar():
    assert _run_jq(_intern_gate_program(), {}) == "0"
