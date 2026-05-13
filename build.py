#!/usr/bin/env python3
"""Build and verify the NKI/ESBMC proof-of-concept.

Concatenates stubs.py + kernels/<kernel>.py + harness/<harness>.py into
single ESBMC-ready files under build/, then optionally runs ESBMC on each
and compares the verdict against an expected outcome.

Usage:
  python3 build.py build        # generate build/*.py
  python3 build.py verify       # build, then run ESBMC and tally results
  python3 build.py clean        # remove build/

The MANIFEST below is the single source of truth for which kernel pairs
with which harness, the ESBMC flags to use, and the verdict we expect.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent
STUBS = ROOT / "stubs.py"
KERNELS = ROOT / "kernels"
HARNESS = ROOT / "harness"
BUILD = ROOT / "build"


@dataclass(frozen=True)
class Target:
    name: str
    kernel: str           # filename under kernels/
    harness: str          # filename under harness/
    esbmc_args: tuple[str, ...]
    expected: str         # "SUCCESSFUL" or "FAILED"


MANIFEST: list[Target] = [
    Target("tensor_add",           "tensor_add.py",       "tensor_add.py",
           (),                     "SUCCESSFUL"),
    Target("tensor_add_buggy",     "tensor_add_buggy.py", "tensor_add_buggy.py",
           (),                     "FAILED"),
    Target("tensor_add_symbolic",  "tensor_add.py",       "tensor_add_symbolic.py",
           ("--unwind", "6"),      "SUCCESSFUL"),
    Target("transpose2d",          "transpose2d.py",      "transpose2d.py",
           (),                     "SUCCESSFUL"),
    Target("transpose2d_buggy",    "transpose2d_buggy.py","transpose2d_buggy.py",
           (),                     "FAILED"),
    Target("matmul",               "matmul.py",           "matmul.py",
           (),                     "SUCCESSFUL"),
    Target("matmul_big",           "matmul.py",           "matmul_big.py",
           (),                     "SUCCESSFUL"),
    Target("matmul_buggy",         "matmul_buggy.py",     "matmul_buggy.py",
           (),                     "FAILED"),
    Target("maxpooling",           "maxpooling.py",       "maxpooling.py",
           (),                     "SUCCESSFUL"),
    Target("maxpooling_buggy",     "maxpooling_buggy.py", "maxpooling_buggy.py",
           (),                     "FAILED"),
    Target("interpolate_bilinear", "interpolate_bilinear.py",
                                                          "interpolate_bilinear.py",
           (),                     "SUCCESSFUL"),
    Target("interpolate_bilinear_buggy",
                                   "interpolate_bilinear_buggy.py",
                                                          "interpolate_bilinear_buggy.py",
           (),                     "FAILED"),
    Target("interpolate_trilinear",
                                   "interpolate_trilinear.py",
                                                          "interpolate_trilinear.py",
           (),                     "SUCCESSFUL"),
    Target("interpolate_trilinear_buggy",
                                   "interpolate_trilinear_buggy.py",
                                                          "interpolate_trilinear_buggy.py",
           (),                     "FAILED"),
]


def build_one(t: Target) -> Path:
    out = BUILD / f"{t.name}.py"
    parts: list[str] = []
    parts.append(f"# AUTO-GENERATED — DO NOT EDIT.\n")
    parts.append(f"# stubs:   stubs.py\n")
    parts.append(f"# kernel:  kernels/{t.kernel}\n")
    parts.append(f"# harness: harness/{t.harness}\n")
    parts.append(f"# Regenerate with: python3 build.py build\n")
    parts.append("\n")
    for src in (STUBS, KERNELS / t.kernel, HARNESS / t.harness):
        parts.append(f"# ====== {src.relative_to(ROOT)} ======\n")
        parts.append(src.read_text())
        parts.append("\n")
    out.write_text("".join(parts))
    return out


def build_all() -> None:
    BUILD.mkdir(exist_ok=True)
    for t in MANIFEST:
        build_one(t)
    print(f"Built {len(MANIFEST)} target(s) under {BUILD.relative_to(ROOT)}/.")


def run_esbmc(t: Target) -> tuple[str, str]:
    """Return (actual_verdict, raw_output_tail)."""
    cmd = ["esbmc", *t.esbmc_args, str(BUILD / f"{t.name}.py")]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    output = proc.stdout + proc.stderr
    if "VERIFICATION SUCCESSFUL" in output:
        verdict = "SUCCESSFUL"
    elif "VERIFICATION FAILED" in output:
        verdict = "FAILED"
    else:
        verdict = "ERROR"
    return verdict, output[-400:]


def verify_all() -> int:
    build_all()
    print()
    print(f"{'Target':<25} {'Expected':<12} {'Actual':<12} {'Result'}")
    print("-" * 70)
    failures: list[tuple[Target, str, str]] = []
    for t in MANIFEST:
        actual, tail = run_esbmc(t)
        ok = actual == t.expected
        mark = "PASS" if ok else "MISMATCH"
        print(f"{t.name:<25} {t.expected:<12} {actual:<12} {mark}")
        if not ok:
            failures.append((t, actual, tail))
    print("-" * 70)
    if failures:
        print(f"\n{len(failures)} mismatch(es):\n")
        for t, actual, tail in failures:
            print(f"=== {t.name}: expected {t.expected}, got {actual} ===")
            print(tail)
            print()
        return 1
    print(f"\nAll {len(MANIFEST)} target(s) match expected verdict.")
    return 0


def clean() -> None:
    if BUILD.exists():
        shutil.rmtree(BUILD)
        print(f"Removed {BUILD.relative_to(ROOT)}/.")


def main(argv: list[str]) -> int:
    if len(argv) != 2 or argv[1] not in ("build", "verify", "clean"):
        print(__doc__)
        return 2
    if argv[1] == "build":
        build_all()
        return 0
    if argv[1] == "verify":
        return verify_all()
    if argv[1] == "clean":
        clean()
        return 0
    return 2  # unreachable


if __name__ == "__main__":
    sys.exit(main(sys.argv))
