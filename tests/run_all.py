#!/usr/bin/env python3
"""
tests/run_all.py — Self-contained FLU test runner.
Runs all test modules without requiring pytest.

Usage:
    PYTHONPATH=src python3 tests/run_all.py
    PYTHONPATH=src python3 tests/run_all.py --verbose
    PYTHONPATH=src python3 tests/run_all.py --fail-fast
"""

from __future__ import annotations

import sys
import os
import time
import argparse
import importlib.util
import unittest
import traceback
from pathlib import Path

# Ensure src/ and the repo root are on the path.
_tests_dir = Path(__file__).parent
_repo_root  = _tests_dir.parent
sys.path.insert(0, str(_repo_root / "src"))
sys.path.insert(0, str(_repo_root))

# ── Test file discovery ────────────────────────────────────────────────────────

_TEST_FILES = [
    # Core
    _tests_dir / "test_core"      / "test_traversal.py",
    _tests_dir / "test_core"      / "test_parity_neural.py",
    _tests_dir / "test_core"      / "test_apn_seeds.py",
    _tests_dir / "test_core"      / "test_nary_vhdl.py",
    _tests_dir / "test_core"      / "test_fm_dance_properties.py",
    _tests_dir / "test_core"      / "test_od32_iterator.py",
    # Theory
    _tests_dir / "test_theory"    / "test_proofs.py",
    _tests_dir / "test_theory"    / "test_registry.py",
    # Container
    _tests_dir / "test_container" / "test_sparse_export.py",
    # Interfaces
    _tests_dir / "test_interfaces" / "test_interfaces.py",
    # Utils
    _tests_dir / "test_utils"     / "test_viz.py",
]


def _load_module(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _collect_test_functions(module) -> list[tuple[str, callable]]:
    """Return all test_* callables from a module, sorted by name."""
    fns = []
    for name in sorted(dir(module)):
        if name.startswith("test_"):
            obj = getattr(module, name)
            if callable(obj) and not isinstance(obj, type):
                fns.append((name, obj))
    return fns


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--fail-fast", "-x", action="store_true")
    args = parser.parse_args()

    passed = failed = errors = 0
    t0 = time.time()

    for path in _TEST_FILES:
        if not path.exists():
            print(f"  MISSING  {path.name}")
            continue
        try:
            mod = _load_module(path)
        except Exception as e:
            print(f"  ERROR loading {path.name}: {e}")
            errors += 1
            continue

        for name, fn in _collect_test_functions(mod):
            try:
                fn()
                passed += 1
                if args.verbose:
                    print(f"  PASS  {path.name}::{name}")
            except Exception as e:
                failed += 1
                print(f"  FAIL  {path.name}::{name}")
                if args.verbose:
                    traceback.print_exc()
                if args.fail_fast:
                    break
        if args.fail_fast and failed:
            break

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"PASSED {passed}  FAILED {failed}  ERRORS {errors}  ({elapsed:.1f}s)")
    print(f"{'='*60}")
    return 1 if (failed or errors) else 0


if __name__ == "__main__":
    sys.exit(main())
