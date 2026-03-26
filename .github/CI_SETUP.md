# FLU CI/CD Setup — 5 Minutes

## Already in Place ✅

Your repository already has:
- `run_tests.py` — Standalone test runner
- `tools/generate_registry_json.py` — Registry generation + verification
- `tests/benchmarks/run_benchmark_suite.py` — Full benchmark suite
- `pyproject.toml` — v15.3.0 with dependencies

## What's New (3 files to add)

1. `.github/workflows/ci.yml` — Main pipeline
2. `.github/workflows/release.yml` — Release workflow
3. `tools/health_check.py` — Local validation

## Enable GitHub Actions

Go to **Settings** → **Actions** → **General**:
- ✅ Allow all actions and reusable workflows
- ✅ Set **Workflow permissions** to **Read and write**

## Test Locally First

```bash
python tools/health_check.py
# Should output: ✅ All critical checks passed
