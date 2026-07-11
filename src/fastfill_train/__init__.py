"""fastfill_train — training workspace for the FastFill layout planner.

Path bootstrap: importing this package makes the vendored FastFill contract
(``scenesmith.growing_world.fastfill``) and the data tools
(``fastfill_data``) importable without installation. Heavy dependencies
(torch/trl/peft/datasets) are imported lazily inside the training
entrypoints so data/eval utilities run anywhere.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
VENDOR = REPO_ROOT / "vendor"

for _p in (VENDOR, VENDOR / "tools"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
