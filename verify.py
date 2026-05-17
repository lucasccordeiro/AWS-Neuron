#!/usr/bin/env python3
"""Run the NKI/ESBMC proof-of-concept regression suite.

Each target is an entry script under `harness/<name>.py` that imports
`stubs` and the appropriate `kernels.<module>`, sets up concrete (or
nondet) input shapes, and asserts the kernel's output contract. ESBMC is
invoked directly on each entry script.

Two phases:
  phase-1 (default flags) — shape and bounds contracts via stub asserts.
  phase-2 (--overflow-check, default div-by-zero) — safety properties
    (signed-integer overflow, integer division by zero) on host-side
    index arithmetic.  Concrete-shape good kernels opt into phase-2 via
    `safety_args` / `safety_expected`; the AUDIT-15 host-arithmetic
    reproducer is phase-2-only and rediscovers the upstream
    ZeroDivisionError without the port-time precondition.

Usage:
  python3 verify.py              # run both phases for every target
  python3 verify.py <name>       # run a single target by manifest name
  python3 verify.py --phase=1    # phase-1 only
  python3 verify.py --phase=2    # phase-2 only

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
    entry: str                       # filename under harness/
    esbmc_args: tuple[str, ...]
    expected: str | None             # phase-1 verdict; None to skip phase-1
    safety_args: tuple[str, ...] = ()    # appended to esbmc_args under phase-2
    safety_expected: str | None = None   # phase-2 verdict; None to skip phase-2


# Phase-2 flags: `--overflow-check` enables signed integer over/underflow
# checks; integer division-by-zero is on by default in ESBMC.  Concrete-
# shape and symbolic-shape good kernels opt in (the symbolic shapes'
# existing `__ESBMC_assume` bounds are tight enough that signed-int
# overflow is unreachable across the nondet shape space — no extra
# preconditions needed); buggy variants stay phase-1 only (the shape
# bug already fails phase-1, phase-2 adds no signal); historical-bug
# stays phase-1 only (same reason).
_SAFETY: tuple[str, ...] = ("--overflow-check",)

# Symbolic-shape `--unwind N` values below are k-induction-certified
# completeness bounds.  An offline `esbmc --k-induction` run on each
# symbolic target reports "Solution found by the forward condition;
# all states are reachable (k = N)", meaning the loop genuinely
# terminates within N unwindings on every nondet input — so plain
# BMC at `--unwind N` is exhaustive on the symbolic shape family,
# not merely bounded.  The two exceptions are `mamba_v3_symbolic` and
# `attn_fwd_v3_symbolic`, where k-induction timed out at 240s; their
# `--unwind` values remain heuristic and the bound is the soundness
# caveat documented in REPORT.md.


MANIFEST: list[Target] = [
    Target("tensor_add",                 "tensor_add.py",                 (),                "SUCCESSFUL", _SAFETY, "SUCCESSFUL"),
    Target("tensor_add_buggy",           "tensor_add_buggy.py",           (),                "FAILED"),
    Target("tensor_add_symbolic",        "tensor_add_symbolic.py",        ("--unwind", "5"), "SUCCESSFUL", _SAFETY, "SUCCESSFUL"),
    Target("transpose2d",                "transpose2d.py",                (),                "SUCCESSFUL", _SAFETY, "SUCCESSFUL"),
    Target("transpose2d_buggy",          "transpose2d_buggy.py",          (),                "FAILED"),
    Target("matmul",                     "matmul.py",                     (),                "SUCCESSFUL", _SAFETY, "SUCCESSFUL"),
    Target("matmul_big",                 "matmul_big.py",                 (),                "SUCCESSFUL", _SAFETY, "SUCCESSFUL"),
    Target("matmul_buggy",               "matmul_buggy.py",               (),                "FAILED"),
    Target("maxpooling",                 "maxpooling.py",                 (),                "SUCCESSFUL", _SAFETY, "SUCCESSFUL"),
    Target("maxpooling_buggy",           "maxpooling_buggy.py",           (),                "FAILED"),
    Target("interpolate_bilinear",       "interpolate_bilinear.py",       (),                "SUCCESSFUL", _SAFETY, "SUCCESSFUL"),
    Target("interpolate_bilinear_buggy", "interpolate_bilinear_buggy.py", (),                "FAILED"),
    Target("interpolate_trilinear",      "interpolate_trilinear.py",      (),                "SUCCESSFUL", _SAFETY, "SUCCESSFUL"),
    Target("interpolate_trilinear_buggy","interpolate_trilinear_buggy.py",(),                "FAILED"),
    Target("interpolate_bilinear_chunk1","interpolate_bilinear_chunk1.py",("--unwind", "1"), "FAILED"),
    Target("interpolate_trilinear_chunk1","interpolate_trilinear_chunk1.py",("--unwind", "1"),"FAILED"),
    # AUDIT-15 host-arithmetic reproducer: phase-2 only.  Rediscovers the
    # upstream ZeroDivisionError on chunk_size=1 via ESBMC's div-by-zero
    # check on the integer floor-div, without relying on our port-time
    # `assert step_size > 0` precondition.
    Target("audit15_hostarith_unguarded","audit15_hostarith_unguarded.py",(),                None,         _SAFETY, "FAILED"),
    Target("matmul_basic",               "matmul_basic.py",               (),                "SUCCESSFUL", _SAFETY, "SUCCESSFUL"),
    Target("matmul_basic_buggy",         "matmul_basic_buggy.py",         (),                "FAILED"),
    Target("mamba_v1",                   "mamba_v1.py",                   (),                "SUCCESSFUL", _SAFETY, "SUCCESSFUL"),
    Target("mamba_v1_buggy",             "mamba_v1_buggy.py",             (),                "FAILED"),
    Target("transpose2d_symbolic",       "transpose2d_symbolic.py",       ("--unwind", "5"), "SUCCESSFUL", _SAFETY, "SUCCESSFUL"),
    Target("maxpooling_symbolic",        "maxpooling_symbolic.py",        ("--unwind", "5"), "SUCCESSFUL", _SAFETY, "SUCCESSFUL"),
    Target("mamba_v1_symbolic",          "mamba_v1_symbolic.py",          ("--unwind", "5"), "SUCCESSFUL", _SAFETY, "SUCCESSFUL"),
    Target("interpolate_bilinear_symbolic", "interpolate_bilinear_symbolic.py",
                                                                                 ("--unwind", "4"), "SUCCESSFUL", _SAFETY, "SUCCESSFUL"),
    Target("interpolate_trilinear_symbolic", "interpolate_trilinear_symbolic.py",
                                                                                 ("--unwind", "3"), "SUCCESSFUL", _SAFETY, "SUCCESSFUL"),
    Target("matmul_tiled",               "matmul_tiled.py",                      (),                "SUCCESSFUL", _SAFETY, "SUCCESSFUL"),
    Target("matmul_tiled_buggy",         "matmul_tiled_buggy.py",                (),                "FAILED"),
    Target("matmul_tiled_symbolic",      "matmul_tiled_symbolic.py",             ("--unwind", "4"), "SUCCESSFUL", _SAFETY, "SUCCESSFUL"),
    Target("matmul_hoist_load",          "matmul_hoist_load.py",                 (),                "SUCCESSFUL", _SAFETY, "SUCCESSFUL"),
    Target("matmul_hoist_load_buggy",    "matmul_hoist_load_buggy.py",           (),                "FAILED"),
    Target("matmul_hoist_load_symbolic", "matmul_hoist_load_symbolic.py",        ("--unwind", "3"), "SUCCESSFUL", _SAFETY, "SUCCESSFUL"),
    Target("matmul_hoist_load_historical",
                                         "matmul_hoist_load_historical.py",      (),                "FAILED"),
    Target("matmul_block_free",          "matmul_block_free.py",                 (),                "SUCCESSFUL", _SAFETY, "SUCCESSFUL"),
    Target("matmul_block_free_buggy",    "matmul_block_free_buggy.py",           (),                "FAILED"),
    Target("matmul_block_free_symbolic", "matmul_block_free_symbolic.py",        ("--unwind", "3"), "SUCCESSFUL", _SAFETY, "SUCCESSFUL"),
    Target("matmul_fully_optimized",     "matmul_fully_optimized.py",            (),                "SUCCESSFUL", _SAFETY, "SUCCESSFUL"),
    Target("matmul_fully_optimized_buggy",
                                         "matmul_fully_optimized_buggy.py",     (),                "FAILED"),
    Target("matmul_fully_optimized_symbolic",
                                         "matmul_fully_optimized_symbolic.py",  ("--unwind", "3"), "SUCCESSFUL", _SAFETY, "SUCCESSFUL"),
    Target("mamba_v2",                   "mamba_v2.py",                          (),                "SUCCESSFUL", _SAFETY, "SUCCESSFUL"),
    Target("mamba_v2_buggy",             "mamba_v2_buggy.py",                    (),                "FAILED"),
    Target("mamba_v3",                   "mamba_v3.py",                          (),                "SUCCESSFUL", _SAFETY, "SUCCESSFUL"),
    Target("mamba_v3_buggy",             "mamba_v3_buggy.py",                    (),                "FAILED"),
    Target("mamba_v3_symbolic",          "mamba_v3_symbolic.py",                 ("--unwind", "5"), "SUCCESSFUL", _SAFETY, "SUCCESSFUL"),
    Target("avgpool",                    "avgpool.py",                           (),                "SUCCESSFUL", _SAFETY, "SUCCESSFUL"),
    Target("avgpool_buggy",              "avgpool_buggy.py",                     (),                "FAILED"),
    Target("avgpool_symbolic",           "avgpool_symbolic.py",                  ("--unwind", "1"), "SUCCESSFUL", _SAFETY, "SUCCESSFUL"),
    Target("attn_fwd_v1",                "attn_fwd_v1.py",                       (),                "SUCCESSFUL", _SAFETY, "SUCCESSFUL"),
    Target("attn_fwd_v1_buggy",          "attn_fwd_v1_buggy.py",                 (),                "FAILED"),
    Target("attn_fwd_v2",                "attn_fwd_v2.py",                       (),                "SUCCESSFUL", _SAFETY, "SUCCESSFUL"),
    Target("attn_fwd_v2_buggy",          "attn_fwd_v2_buggy.py",                 (),                "FAILED"),
    Target("attn_fwd_v3",                "attn_fwd_v3.py",                       ("--unwind", "5"), "SUCCESSFUL", _SAFETY, "SUCCESSFUL"),
    Target("attn_fwd_v3_buggy",          "attn_fwd_v3_buggy.py",                 ("--unwind", "5"), "FAILED"),
    Target("attn_fwd_v3_symbolic",       "attn_fwd_v3_symbolic.py",              ("--unwind", "9"), "SUCCESSFUL", _SAFETY, "SUCCESSFUL"),
    Target("pipelined_attention",        "pipelined_attention.py",               (),                "SUCCESSFUL", _SAFETY, "SUCCESSFUL"),
    Target("pipelined_attention_load_q", "pipelined_attention_load_q.py",        (),                "SUCCESSFUL", _SAFETY, "SUCCESSFUL"),
]


def run(t: Target, extra_args: tuple[str, ...] = ()) -> tuple[str, str]:
    cmd = ["esbmc", *t.esbmc_args, *extra_args, str(HARNESS / t.entry)]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300, cwd=ROOT)
    out = proc.stdout + proc.stderr
    if "VERIFICATION SUCCESSFUL" in out:
        return "SUCCESSFUL", out[-400:]
    if "VERIFICATION FAILED" in out:
        return "FAILED", out[-400:]
    return "ERROR", out[-400:]


def _phase_runs(t: Target, phases: set[int]) -> list[tuple[int, tuple[str, ...], str]]:
    """Phase-tagged runs to perform for `t`, filtered by requested phases."""
    out: list[tuple[int, tuple[str, ...], str]] = []
    if 1 in phases and t.expected is not None:
        out.append((1, (), t.expected))
    if 2 in phases and t.safety_expected is not None:
        out.append((2, t.safety_args, t.safety_expected))
    return out


def main(argv: list[str]) -> int:
    phases: set[int] = {1, 2}
    args = argv[1:]
    if args and args[0].startswith("--phase="):
        spec = args[0].split("=", 1)[1]
        phases = {int(p) for p in spec.split(",") if p}
        args = args[1:]

    targets = MANIFEST
    if len(args) == 1:
        targets = [t for t in MANIFEST if t.name == args[0]]
        if not targets:
            print(f"unknown target: {args[0]}", file=sys.stderr)
            return 2

    print(f"{'Target':<32} {'Phase':<6} {'Expected':<12} {'Actual':<12} {'Result'}")
    print("-" * 78)
    failures: list[tuple[Target, int, str, str, str]] = []
    runs_count = 0
    for t in targets:
        for phase, extra, expected in _phase_runs(t, phases):
            actual, tail = run(t, extra)
            ok = actual == expected
            result = "PASS" if ok else "MISMATCH"
            print(f"{t.name:<32} "
                  f"{phase:<6} "
                  f"{_paint(expected, 12)} "
                  f"{_paint(actual,   12)} "
                  f"{_paint(result)}")
            runs_count += 1
            if not ok:
                failures.append((t, phase, expected, actual, tail))
    print("-" * 78)
    if failures:
        print(f"\n{len(failures)} mismatch(es):\n")
        for t, phase, expected, actual, tail in failures:
            print(f"=== {t.name} (phase {phase}): expected {expected}, got {actual} ===")
            print(tail)
            print()
        return 1
    print(f"\nAll {runs_count} run(s) match expected verdict.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
