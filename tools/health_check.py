#!/usr/bin/env python3
"""
FLU Project Health Check Tool
==============================

Validates project health using your existing test infrastructure.
Run: python tools/health_check.py

Exit code: 0 if all critical checks pass, 1 otherwise.
"""

from __future__ import annotations
import sys
import os
import subprocess
from pathlib import Path
from typing import List, Tuple

ROOT = Path(__file__).parent.parent

class HealthCheck:
    def __init__(self):
        self.checks: List[Tuple[str, bool, str]] = []
        self.critical_failures = 0
    
    def check(self, name: str, passed: bool, critical: bool = True, note: str = "") -> None:
        """Record a health check result."""
        icon = "✓" if passed else ("⚠" if not critical else "✗")
        status = "PASS" if passed else ("WARN" if not critical else "FAIL")
        self.checks.append((name, passed, note))
        print(f"  {icon:>3}  {name:<40}  {status:>6}  {note}")
        if not passed and critical:
            self.critical_failures += 1
    
    def summary(self):
        """Get summary stats."""
        total = len(self.checks)
        passed = sum(1 for _, p, _ in self.checks if p)
        return {
            "total": total,
            "passed": passed,
            "failed": total - passed,
            "critical_failures": self.critical_failures,
        }
    
    def report(self) -> str:
        """Generate report."""
        s = self.summary()
        return (
            f"\n{'='*72}\n"
            f"PASSED {s['passed']}/{s['total']}  CRITICAL FAILURES {s['critical_failures']}\n"
            f"{'='*72}\n"
        )

health = HealthCheck()

# ─────────────────────────────────────────────────────────────────────────────

def check_python_version():
    version = sys.version_info
    passed = version >= (3, 10)
    health.check("Python Version", passed, critical=True, 
                 note=f"{version.major}.{version.minor}.{version.micro}")

def check_dependencies():
    required = ["numpy"]
    missing = []
    for pkg in required:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    passed = len(missing) == 0
    note = f"missing: {', '.join(missing)}" if missing else "all installed"
    health.check("Dependencies", passed, critical=True, note=note)

def check_package_imports():
    sys.path.insert(0, str(ROOT / "src"))
    try:
        import flu
        passed = True
        note = f"v{flu.__version__}"
    except Exception as e:
        passed = False
        note = str(e)[:30]
    health.check("FLU Package", passed, critical=True, note=note)

def check_unit_tests():
    try:
        result = subprocess.run(
            ["python", "run_tests.py"],
            cwd=ROOT,
            capture_output=True,
            timeout=120,
            text=True
        )
        passed = result.returncode == 0
        output = result.stdout + result.stderr
        
        # Extract test count
        note = "all pass" if passed else "some failed"
        for line in output.split('\n'):
            if "PASSED" in line and "FAILED" in line:
                note = line.strip()[:40]
                break
        
        health.check("Unit Tests", passed, critical=True, note=note)
    except subprocess.TimeoutExpired:
        health.check("Unit Tests", False, critical=True, note="timeout")
    except Exception as e:
        health.check("Unit Tests", False, critical=True, note=str(e)[:30])

def check_registry_verification():
    try:
        result = subprocess.run(
            ["python", "tools/generate_registry_json.py", "--verify"],
            cwd=ROOT,
            capture_output=True,
            timeout=30,
            text=True
        )
        passed = result.returncode == 0
        note = "current" if passed else "out of sync"
        health.check("Theorem Registry", passed, critical=True, note=note)
    except Exception as e:
        health.check("Theorem Registry", False, critical=True, note=str(e)[:30])

def check_benchmarks():
    try:
        result = subprocess.run(
            ["python", "tests/benchmarks/run_benchmark_suite.py", "--quiet"],
            cwd=ROOT,
            capture_output=True,
            timeout=180,
            text=True
        )
        passed = result.returncode == 0
        note = "all pass" if passed else "some sections failed"
        health.check("Benchmarks", passed, critical=False, note=note)
    except subprocess.TimeoutExpired:
        health.check("Benchmarks", False, critical=False, note="timeout")
    except Exception as e:
        health.check("Benchmarks", False, critical=False, note=str(e)[:30])

# ─────────────────────────────────────────────────────────────────────────────

def main() -> int:
    print("\n" + "="*72)
    print("FLU Project Health Check")
    print("="*72 + "\n")
    
    check_python_version()
    check_dependencies()
    check_package_imports()
    check_unit_tests()
    check_registry_verification()
    check_benchmarks()
    
    print(health.report())
    
    if health.critical_failures > 0:
        print(f"❌ {health.critical_failures} critical failure(s)")
        return 1
    else:
        print("✅ All critical checks passed")
        return 0

if __name__ == "__main__":
    sys.exit(main())
