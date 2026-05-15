#!/usr/bin/env python3
"""Run the NKI/ESBMC proof-of-concept regression suite.

Each target is an entry script under `harness/<name>.py` that imports
`stubs` and the appropriate `kernels.<module>`, sets up concrete (or
nondet) input shapes, and asserts the kernel's output contract. ESBMC is
invoked directly on each entry script.

Usage:
  python3 verify.py              # run every target, tally results
  python3 verify.py <name>       # run a single target by manifest name

The MANIFEST below is the single source of truth for ESBMC flags and
expected verdicts.
"""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent
HARNESS = ROOT / "harness"

# ANSI colours, applied only when stdout is a TTY so log files stay clean.
_USE_COLOR = sys.stdout.isatty()
_GREEN  = "\033[32m" if _USE_COLOR else ""
_RED    = "\033[31m" if _USE_COLOR else ""
_RESET  = "\033[0m"  if _USE_COLOR else ""


def _paint(text: str, width: int = 0) -> str:
    """Left-pad to `width` (visible characters) and wrap in ANSI colour."""
    padded = text.ljust(width)
    if text in ("SUCCESSFUL", "PASS"):
        return f"{_GREEN}{padded}{_RESET}"
    if text in ("FAILED", "MISMATCH", "ERROR"):
        return f"{_RED}{padded}{_RESET}"
    return padded


@dataclass(frozen=True)
class Target:
    name: str
    entry: str                  # filename under harness/
    esbmc_args: tuple[str, ...]
    expected: str               # "SUCCESSFUL" or "FAILED"


MANIFEST: list[Target] = [
    Target("tensor_add",                 "tensor_add.py",                 (),                "SUCCESSFUL"),
    Target("tensor_add_buggy",           "tensor_add_buggy.py",           (),                "FAILED"),
    Target("tensor_add_symbolic",        "tensor_add_symbolic.py",        ("--unwind", "6"), "SUCCESSFUL"),
    Target("transpose2d",                "transpose2d.py",                (),                "SUCCESSFUL"),
    Target("transpose2d_buggy",          "transpose2d_buggy.py",          (),                "FAILED"),
    Target("matmul",                     "matmul.py",                     (),                "SUCCESSFUL"),
    Target("matmul_big",                 "matmul_big.py",                 (),                "SUCCESSFUL"),
    Target("matmul_buggy",               "matmul_buggy.py",               (),                "FAILED"),
    Target("maxpooling",                 "maxpooling.py",                 (),                "SUCCESSFUL"),
    Target("maxpooling_buggy",           "maxpooling_buggy.py",           (),                "FAILED"),
    Target("interpolate_bilinear",       "interpolate_bilinear.py",       (),                "SUCCESSFUL"),
    Target("interpolate_bilinear_buggy", "interpolate_bilinear_buggy.py", (),                "FAILED"),
    Target("interpolate_trilinear",      "interpolate_trilinear.py",      (),                "SUCCESSFUL"),
    Target("interpolate_trilinear_buggy","interpolate_trilinear_buggy.py",(),                "FAILED"),
    Target("interpolate_bilinear_chunk1","interpolate_bilinear_chunk1.py",("--unwind", "1"), "FAILED"),
    Target("interpolate_trilinear_chunk1","interpolate_trilinear_chunk1.py",("--unwind", "1"),"FAILED"),
    Target("matmul_basic",               "matmul_basic.py",               (),                "SUCCESSFUL"),
    Target("matmul_basic_buggy",         "matmul_basic_buggy.py",         (),                "FAILED"),
    Target("mamba_v1",                   "mamba_v1.py",                   (),                "SUCCESSFUL"),
    Target("mamba_v1_buggy",             "mamba_v1_buggy.py",             (),                "FAILED"),
    Target("transpose2d_symbolic",       "transpose2d_symbolic.py",       ("--unwind", "5"), "SUCCESSFUL"),
    Target("maxpooling_symbolic",        "maxpooling_symbolic.py",        ("--unwind", "5"), "SUCCESSFUL"),
    Target("mamba_v1_symbolic",          "mamba_v1_symbolic.py",          ("--unwind", "5"), "SUCCESSFUL"),
    Target("interpolate_bilinear_symbolic", "interpolate_bilinear_symbolic.py",
                                                                                 ("--unwind", "5"), "SUCCESSFUL"),
    Target("interpolate_trilinear_symbolic", "interpolate_trilinear_symbolic.py",
                                                                                 ("--unwind", "5"), "SUCCESSFUL"),
    Target("matmul_tiled",               "matmul_tiled.py",                      (),                "SUCCESSFUL"),
    Target("matmul_tiled_buggy",         "matmul_tiled_buggy.py",                (),                "FAILED"),
    Target("matmul_tiled_symbolic",      "matmul_tiled_symbolic.py",             ("--unwind", "4"), "SUCCESSFUL"),
    Target("matmul_hoist_load",          "matmul_hoist_load.py",                 (),                "SUCCESSFUL"),
    Target("matmul_hoist_load_buggy",    "matmul_hoist_load_buggy.py",           (),                "FAILED"),
    Target("matmul_hoist_load_historical",
                                         "matmul_hoist_load_historical.py",      (),                "FAILED"),
    Target("matmul_block_free",          "matmul_block_free.py",                 (),                "SUCCESSFUL"),
    Target("matmul_block_free_buggy",    "matmul_block_free_buggy.py",           (),                "FAILED"),
    Target("matmul_fully_optimized",     "matmul_fully_optimized.py",            (),                "SUCCESSFUL"),
    Target("matmul_fully_optimized_buggy",
                                         "matmul_fully_optimized_buggy.py",     (),                "FAILED"),
    Target("mamba_v2",                   "mamba_v2.py",                          (),                "SUCCESSFUL"),
    Target("mamba_v2_buggy",             "mamba_v2_buggy.py",                    (),                "FAILED"),
    Target("mamba_v3",                   "mamba_v3.py",                          (),                "SUCCESSFUL"),
    Target("mamba_v3_buggy",             "mamba_v3_buggy.py",                    (),                "FAILED"),
    Target("mamba_v3_symbolic",          "mamba_v3_symbolic.py",                 ("--unwind", "5"), "SUCCESSFUL"),
    Target("avgpool",                    "avgpool.py",                           (),                "SUCCESSFUL"),
    Target("avgpool_buggy",              "avgpool_buggy.py",                     (),                "FAILED"),
    Target("avgpool_symbolic",           "avgpool_symbolic.py",                  ("--unwind", "5"), "SUCCESSFUL"),
    Target("attn_fwd_v1",                "attn_fwd_v1.py",                       (),                "SUCCESSFUL"),
    Target("attn_fwd_v1_buggy",          "attn_fwd_v1_buggy.py",                 (),                "FAILED"),
    Target("attn_fwd_v2",                "attn_fwd_v2.py",                       (),                "SUCCESSFUL"),
    Target("attn_fwd_v2_buggy",          "attn_fwd_v2_buggy.py",                 (),                "FAILED"),
    Target("attn_fwd_v3",                "attn_fwd_v3.py",                       ("--unwind", "5"), "SUCCESSFUL"),
    Target("attn_fwd_v3_buggy",          "attn_fwd_v3_buggy.py",                 ("--unwind", "5"), "FAILED"),
    Target("attn_fwd_v3_symbolic",       "attn_fwd_v3_symbolic.py",              ("--unwind", "9"), "SUCCESSFUL"),
    Target("pipelined_attention",        "pipelined_attention.py",               (),                "SUCCESSFUL"),
]


def run(t: Target) -> tuple[str, str]:
    cmd = ["esbmc", *t.esbmc_args, str(HARNESS / t.entry)]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300, cwd=ROOT)
    out = proc.stdout + proc.stderr
    if "VERIFICATION SUCCESSFUL" in out:
        return "SUCCESSFUL", out[-400:]
    if "VERIFICATION FAILED" in out:
        return "FAILED", out[-400:]
    return "ERROR", out[-400:]


def main(argv: list[str]) -> int:
    targets = MANIFEST
    if len(argv) == 2:
        targets = [t for t in MANIFEST if t.name == argv[1]]
        if not targets:
            print(f"unknown target: {argv[1]}", file=sys.stderr)
            return 2

    print(f"{'Target':<28} {'Expected':<12} {'Actual':<12} {'Result'}")
    print("-" * 72)
    failures: list[tuple[Target, str, str]] = []
    for t in targets:
        actual, tail = run(t)
        ok = actual == t.expected
        result = "PASS" if ok else "MISMATCH"
        print(f"{t.name:<28} "
              f"{_paint(t.expected, 12)} "
              f"{_paint(actual,    12)} "
              f"{_paint(result)}")
        if not ok:
            failures.append((t, actual, tail))
    print("-" * 72)
    if failures:
        print(f"\n{len(failures)} mismatch(es):\n")
        for t, actual, tail in failures:
            print(f"=== {t.name}: expected {t.expected}, got {actual} ===")
            print(tail)
            print()
        return 1
    print(f"\nAll {len(targets)} target(s) match expected verdict.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
