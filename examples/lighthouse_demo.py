"""
examples/lighthouse_demo.py
============================
LighthouseBeacon.broadcast() — CBPQC demo (SIMULATION ONLY).

╔══════════════════════════════════════════════════════════╗
║  SIMULATION ONLY — mathematical demonstration.           ║
║  NOT a production cryptographic system.                  ║
╚══════════════════════════════════════════════════════════╝

Run:  PYTHONPATH=src python3 examples/lighthouse_demo.py
"""

from flu import LighthouseBeacon

print("=== LighthouseBeacon CBPQC Demo (SIMULATION ONLY) ===\n")

# ── Basic broadcast ───────────────────────────────────────────────────────────
beacon = LighthouseBeacon(n=3, rounds=2, seed=42)
beacon.broadcast()

# ── Self-consistency check ────────────────────────────────────────────────────
result = beacon.verify()
print(f"\nSelf-check:")
print(f"  deterministic : {result['deterministic']}  (same k_start → same key)")
print(f"  distinct      : {result['distinct']}   (different k_start → different key)")
print(f"  verified      : {result['verified']}")

# ── Key material ──────────────────────────────────────────────────────────────
k0 = beacon.generate_key(k_start=0)
k1 = beacon.generate_key(k_start=1)
print(f"\nKey k_start=0: {k0.hex[:32]}…")
print(f"Key k_start=1: {k1.hex[:32]}…")
print(f"Keys differ:   {k0.digest != k1.digest}")

# ── CLI reminder ─────────────────────────────────────────────────────────────
print("\nCLI equivalent:")
print("  flu-lighthouse --n 3 --rounds 2 --seed 42")
print("\nDone.")
