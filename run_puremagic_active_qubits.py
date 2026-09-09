#!/usr/bin/env python3
"""Run PureMagic benchmarks with Part-B defaults and sum logged qubit use over cycles.

    python3 run_puremagic_active_qubits.py
    python3 run_puremagic_active_qubits.py --benchmarks ftcb_adder_10q
    python3 run_puremagic_active_qubits.py --magic-state-lambda 100 --no-t-failures

Uses the Part-B settings: Pauli-product width 1, lambda 0.5,
T failures enabled, seed 29, and one ancilla row. Counts are the raw trace numerators,
not distinct qubits over the whole run or a normalized occupied-cube metric.
Plotting is disabled; the reported counters are summed as-is.
The text table is saved in this repository's results directory. Failed and timed-out
benchmarks are marked explicitly; the table is updated after every benchmark.

Logging requires debug assertions. Build a separate optimized binary with these
enabled, without editing PureMagic sources or replacing its normal binaries.
"""

from __future__ import annotations

import argparse
import math
import os
import re
import shutil
import subprocess
import time
from pathlib import Path

import run_puremagic_repo_benchmarks as common

PUREMAGIC_ROOT = Path(__file__).resolve().parent
BUILD_DIR = PUREMAGIC_ROOT / "target" / "active-qubits"
RAW_DIR = PUREMAGIC_ROOT / "results" / "benchmarking" / "raw_puremagic_active_qubits"
DEFAULT_RESULTS = PUREMAGIC_ROOT / "results" / "puremagic_active_qubits.txt"
CYCLE = re.compile(r"\bINFO\s+lcycle (\d+):")
QUBITS = re.compile(r"\bINFO\s+qubits:\s+(\d+)/(\d+)\s+\(")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--benchmarks", nargs="+", help="QASM stems; default: all benchmarks.")
    parser.add_argument("--benchmark-dir", type=Path, default=common.DEFAULT_BENCHMARK_DIR)
    parser.add_argument("--results-file", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--max-gates", type=int, default=25_000, help="Gate cutoff; 0 disables.")
    parser.add_argument("--seed", type=int, default=29)
    parser.add_argument("--ancilla-rows", type=int, default=1)
    parser.add_argument("--magic-state-lambda", type=float, default=0.5,
                        help="Magic-state production rate per logical cycle (default: 0.5).")
    parser.add_argument("--no-t-failures", action="store_true",
                        help="Disable T-injection failures (enabled by default).")
    parser.add_argument("--timeout-s", type=float, default=common.DEFAULT_TIMEOUT_S,
                        help="Timeout for each transpiler+scheduler pipeline (default: 7200).")
    parser.add_argument("--build", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--transpiler", type=Path, default=BUILD_DIR / "release" / "transpile")
    parser.add_argument("--scheduler", type=Path, default=BUILD_DIR / "release" / "puremagic",
                        help="Must have debug assertions enabled, even with --no-build.")
    args = parser.parse_args()
    if args.max_gates < 0 or args.ancilla_rows < 1 or args.timeout_s <= 0:
        parser.error("Require max-gates >= 0, ancilla-rows >= 1, and timeout-s > 0.")
    if not math.isfinite(args.magic_state_lambda) or args.magic_state_lambda <= 0:
        parser.error("--magic-state-lambda must be finite and positive.")
    for name in ("benchmark_dir", "results_file", "transpiler", "scheduler"):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    return args


def build_binaries() -> None:
    env = dict(os.environ, CARGO_PROFILE_RELEASE_DEBUG_ASSERTIONS="true")
    subprocess.run(
        ["cargo", "build", "--locked", "--release", "--target-dir", str(BUILD_DIR),
         "--bin", "transpile", "--bin", "puremagic"],
        cwd=PUREMAGIC_ROOT, env=env, check=True,
    )


def sum_active_qubits(trace: Path, expected_cycles: int) -> int:
    """Sum one numerator per cycle, rejecting missing, duplicate, or partial logs."""
    if not trace.is_file():
        raise RuntimeError(f"Missing scheduler trace: {trace}")
    total = completed_cycles = 0
    pending_cycle = None
    with trace.open(encoding="utf-8") as source:
        for raw_line in source:
            line = common.ANSI_ESCAPE.sub("", raw_line)
            cycle = CYCLE.search(line)
            if cycle:
                if pending_cycle is not None or int(cycle[1]) != completed_cycles + 1:
                    raise RuntimeError(f"Missing or duplicate logical cycle in {trace}")
                pending_cycle = int(cycle[1])
            used = QUBITS.search(line)
            if used:
                if pending_cycle is None:
                    raise RuntimeError(f"Qubit count without a unique logical cycle in {trace}")
                total += int(used[1])
                completed_cycles += 1
                pending_cycle = None
    if pending_cycle is not None or completed_cycles != expected_cycles:
        raise RuntimeError(
            f"Trace has {completed_cycles} complete cycles; expected {expected_cycles}. "
            "Use a logging-enabled binary with --log-scheduler info."
        )
    return total


def run_one(qasm: Path, args: argparse.Namespace) -> int:
    workdir = RAW_DIR / qasm.stem
    workdir.mkdir(parents=True, exist_ok=True)
    copied_qasm = workdir / f"{qasm.stem}.cliffordt.qasm"
    trans = workdir / f"{qasm.stem}.trans"
    trace = workdir / f"{qasm.stem}.sched_trace"
    trace.unlink(missing_ok=True)
    shutil.copy2(qasm, copied_qasm)
    commands = [
        ("transpile", [str(args.transpiler), "--input_file", str(copied_qasm),
                       "--output_file", str(trans), "--max_width", str(common.MAX_PAULI_PRODUCT_WEIGHT)]),
        ("puremagic", [str(args.scheduler), "--circuit", str(trans), "--use-magic-routing",
                       "--magic-state-lambda", str(args.magic_state_lambda),
                       "--rseed", str(args.seed), "--ancilla-rows", str(args.ancilla_rows),
                       "--log-scheduler", "info",
                       *(["--no-t-failures"] if args.no_t_failures else [])]),
    ]
    deadline = time.perf_counter() + args.timeout_s
    for stage, command in commands:
        log_path = workdir / f"{stage}.log"
        with log_path.open("w", encoding="utf-8") as log:
            result = subprocess.run(
                command, cwd=workdir, stdout=log, stderr=subprocess.STDOUT,
                timeout=common.remaining_timeout(deadline, args.timeout_s, stage),
            )
        if result.returncode:
            raise RuntimeError(f"{stage} exited with {result.returncode}; see {log_path}")
    output = (workdir / "puremagic.log").read_text(encoding="utf-8")
    scheduled = common.parse_required(
        r"Scheduled\s+\d+\s+in\s+(\d+)\s+logical cycles, volume\s+\d+",
        output, "logical cycle count",
    )
    return sum_active_qubits(trace, int(scheduled[1]))


def write_table(args: argparse.Namespace, rows: dict[str, int | str]) -> None:
    width = max(len("Benchmark"), *(len(stem) for stem in rows))
    lines = [
        "PureMagic active qubits",
        "Metric: sum of the logged used-qubit numerator across all logical cycles.",
        "Raw scheduler counts with plotting disabled; summed as reported.",
        f"Settings: lambda={args.magic_state_lambda:g}, "
        f"T failures={'off' if args.no_t_failures else 'on'}, seed={args.seed}, "
        f"ancilla rows={args.ancilla_rows}, max gates={args.max_gates}, timeout={args.timeout_s:g}s.",
        "",
        f"| {'Benchmark':<{width}} | {'Total active qubits':>19} |",
        f"| {'-' * width} | {'-' * 19} |",
        *(f"| {stem:<{width}} | {str(total):>19} |" for stem, total in rows.items()),
        "",
        f"Logs and traces: {RAW_DIR}",
    ]
    args.results_file.parent.mkdir(parents=True, exist_ok=True)
    args.results_file.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    paths = common.benchmark_paths(args.benchmark_dir, args.benchmarks)
    selected = []
    rows: dict[str, int | str] = {}
    for qasm in paths:
        metadata = common.qasm_metadata(qasm)
        if args.max_gates and metadata["gate_count"] > args.max_gates:
            rows[qasm.stem] = "SKIPPED (gate limit)"
        elif metadata["unsupported_gate_counts"]:
            rows[qasm.stem] = "UNSUPPORTED GATES"
        else:
            selected.append(qasm)
            rows[qasm.stem] = "PENDING"
    if selected:
        if args.build:
            print("Building separate logging-enabled PureMagic binaries...", flush=True)
            build_binaries()
        for binary in (args.transpiler, args.scheduler):
            if not binary.is_file():
                raise FileNotFoundError(binary)
    write_table(args, rows)
    for qasm in selected:
        print(f"Running {qasm.stem}...", flush=True)
        try:
            rows[qasm.stem] = run_one(qasm, args)
        except (TimeoutError, subprocess.TimeoutExpired):
            rows[qasm.stem] = "TIMEOUT"
            print(f"Timed out after {args.timeout_s:g}s; see {RAW_DIR / qasm.stem}")
        except (OSError, RuntimeError) as exc:
            rows[qasm.stem] = "FAILED"
            print(f"Failed: {exc}")
        write_table(args, rows)
        print(f"{qasm.stem}: {rows[qasm.stem]}", flush=True)
    print(f"Saved table to {args.results_file}")


if __name__ == "__main__":
    main()
