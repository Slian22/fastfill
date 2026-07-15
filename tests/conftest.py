"""pytest configuration for the fastfill test suite.

Adds the tests/ directory to sys.path so test modules can import shared
helpers from each other (e.g. ``from test_injectors import make_clean_sample``)
regardless of how pytest is invoked (from repo root, from tests/, via tox, etc).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
