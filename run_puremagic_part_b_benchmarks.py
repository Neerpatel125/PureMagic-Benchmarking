#!/usr/bin/env python3
"""Benchmark PureMagic against the myTopoLS Part-B supply assumptions.

This runner is separate from the ready-magic Part-A comparison.  It uses
``--magic-state-lambda 0.5`` by default (mean two logical cycles), leaves
stochastic T-injection failures enabled, and records critical-path magic-state
delay emitted by the instrumented scheduler.

Example:
    python3 run_puremagic_part_b_benchmarks.py --max-gates 10000

The summary JSON is written to LS-Benchmarking-Results, while raw logs and
artifacts stay in PureMagic/results/benchmarking.

Each complete transpiler+scheduler benchmark has a two-hour wall-time limit by
default. Timed-out runs are recorded and the runner continues with the next
benchmark. They are skipped by ``--resume`` and rerun only when starting a
fresh run.
"""

from __future__ import annotations

import argparse
import json
import selectors
import shutil
import statistics
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import run_puremagic_repo_benchmarks as common


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_BENCHMARK_DIR = common.DEFAULT_BENCHMARK_DIR
DEFAULT_RESULTS = (
    REPO_ROOT.parent
    / "LS-Benchmarking-Results"
    / "part_b_comparison"
    / "results"
    / "puremagic_part_b_results.json"
)
DEFAULT_RAW_DIR = REPO_ROOT / "results" / "benchmarking" / "raw_puremagic_part_b"
DEFAULT_TRANSPILE = REPO_ROOT / "target" / "release" / "transpile"
DEFAULT_SCHEDULER = REPO_ROOT / "target" / "release" / "puremagic"
METHOD_NAME = "puremagic_part_b_supply"
METHOD_LABEL = "PureMagic (Part-B supply)"
MAX_PAULI_PRODUCT_WEIGHT = 1
DEFAULT_MAGIC_STATE_LAMBDA = 0.5
CODE_DISTANCE = 11


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run PureMagic with the myTopoLS Part-B magic-state supply settings."
    )
    parser.add_argument(
        "--benchmarks",
        nargs="+",
        help="QASM stems to consider (default: every QASM in --benchmark-dir).",
    )
    parser.add_argument("--benchmark-dir", type=Path, default=DEFAULT_BENCHMARK_DIR)
    parser.add_argument(
        "--max-gates",
        type=int,
        default=10_000,
        help="Skip circuits above this gate count (default: 10000; 0 disables).",
    )
    parser.add_argument("--results-file", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--transpiler", type=Path, default=DEFAULT_TRANSPILE)
    parser.add_argument("--scheduler", type=Path, default=DEFAULT_SCHEDULER)
    parser.add_argument(
        "--magic-state-lambda", type=float, default=DEFAULT_MAGIC_STATE_LAMBDA,
        help="Exponential production rate per PureMagic lcycle (default: 0.5).",
    )
    parser.add_argument(
        "--seeds", nargs="+", type=int, default=[29],
        help="Cultivation/T-injection seeds (default: 29). Use several seeds for paper results.",
    )
    parser.add_argument("--ancilla-rows", type=int, default=1)
    parser.add_argument(
        "--timeout-s", type=float, default=7200.0,
        help="Maximum wall time for each complete transpiler+scheduler benchmark (default: 2 hours).",
    )
    parser.add_argument(
        "--build", action=argparse.BooleanOptionalAction, default=True,
        help="Build release binaries before benchmarking (default: true).",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Skip completed or timed-out entries already present in the result JSON.",
    )
    return parser.parse_args()


def tee_process(
    command: list[str], *, cwd: Path, log_path: Path, timeout_s: float
) -> tuple[str, float]:
    start = time.perf_counter()
    lines: list[str] = []
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        selector = selectors.DefaultSelector()
        selector.register(process.stdout, selectors.EVENT_READ)
        try:
            while process.poll() is None:
                if time.perf_counter() - start > timeout_s:
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                    raise TimeoutError(
                        f"Timed out after {timeout_s:.0f}s: {' '.join(command)}; see {log_path}"
                    )
                for key, _ in selector.select(timeout=0.25):
                    line = key.fileobj.readline()
                    if line:
                        print(line, end="")
                        log.write(line)
                        lines.append(line)
            for line in process.stdout:
                print(line, end="")
                log.write(line)
                lines.append(line)
        finally:
            selector.close()
        elapsed = time.perf_counter() - start
        if process.returncode:
            raise RuntimeError(
                f"Command failed with exit code {process.returncode}; see {log_path}"
            )
    return "".join(lines), elapsed


def remaining_timeout(deadline: float, total_timeout_s: float, stage: str) -> float:
    remaining = deadline - time.perf_counter()
    if remaining <= 0:
        raise TimeoutError(
            f"Timed out after {total_timeout_s:.0f}s before the {stage} stage could start."
        )
    return remaining


def parse_delay(output: str) -> int:
    match = common.parse_required(
        r"^Magic-state delay:\s*(\d+)\s+logical cycles\s*$",
        output,
        "critical-path magic-state delay",
    )
    return int(match.group(1))


def build_release_binaries() -> None:
    command = ["cargo", "build", "--release", "--bin", "transpile", "--bin", "puremagic"]
    print("Building PureMagic release binaries...")
    subprocess.run(command, cwd=REPO_ROOT, check=True)


def run_transpiler(
    stem: str,
    source_qasm: Path,
    args: argparse.Namespace,
    raw_dir: Path,
    timeout_s: float,
) -> dict:
    clifford_qasm = raw_dir / f"{stem}.cliffordt.qasm"
    trans_file = raw_dir / f"{stem}.trans"
    log_path = raw_dir / f"{stem}__transpile.log"
    shutil.copy2(source_qasm, clifford_qasm)
    command = [
        str(args.transpiler),
        "--input_file", str(clifford_qasm),
        "--output_file", str(trans_file),
        "--max_width", str(MAX_PAULI_PRODUCT_WEIGHT),
    ]
    output, process_wall = tee_process(
        command, cwd=raw_dir, log_path=log_path, timeout_s=timeout_s
    )
    metrics = common.parse_transpiler_output(output)
    metrics["benchmark_wall_time_s"] = common.parse_benchmark_wall_time(output, "transpiler")
    metrics["process_wall_time_s"] = process_wall
    return {
        "metrics": metrics,
        "trans_file": trans_file,
        "clifford_qasm": clifford_qasm,
        "log_path": log_path,
        "command": command,
    }


def run_scheduler_trial(
    stem: str,
    seed: int,
    trans_file: Path,
    transpiler_metrics: dict,
    args: argparse.Namespace,
    raw_dir: Path,
    timeout_s: float,
) -> dict:
    log_path = raw_dir / f"{stem}__seed_{seed}__puremagic.log"
    command = [
        str(args.scheduler),
        "--circuit", str(trans_file),
        "--use-magic-routing",
        "--magic-state-lambda", str(args.magic_state_lambda),
        "--rseed", str(seed),
        "--ancilla-rows", str(args.ancilla_rows),
    ]
    # Deliberately no --no-t-failures: the 50% injection-outcome model is enabled.
    output, process_wall = tee_process(
        command, cwd=raw_dir, log_path=log_path, timeout_s=timeout_s
    )
    scheduler_metrics = common.parse_scheduler_output(output)
    scheduler_wall = common.parse_benchmark_wall_time(output, "scheduler")
    scheduler_metrics["benchmark_wall_time_s"] = scheduler_wall
    scheduler_metrics["process_wall_time_s"] = process_wall
    scheduler_metrics["magic_state_delay_lcycles"] = parse_delay(output)

    space = scheduler_metrics["topology_qubit_count"]
    logical_time = scheduler_metrics["logical_cycles"]
    volume = scheduler_metrics["volume"]
    delay = scheduler_metrics["magic_state_delay_lcycles"]
    if space * logical_time != volume:
        raise RuntimeError(f"{stem}: volume {volume} != {space} * {logical_time}")
    if delay > logical_time:
        raise RuntimeError(f"{stem}: delay {delay} exceeds total lcycles {logical_time}")
    schedule_source = raw_dir / f"{trans_file.stem}.schedule"
    schedule_copy = raw_dir / f"{stem}__seed_{seed}.schedule"
    if schedule_source.exists():
        shutil.copy2(schedule_source, schedule_copy)

    fair_wall = float(transpiler_metrics["benchmark_wall_time_s"] + scheduler_wall)
    full_process_wall = float(transpiler_metrics["process_wall_time_s"] + process_wall)
    return {
        "seed": seed,
        "status": "ok",
        "metrics": {
            "space": float(space),
            "time": float(logical_time),
            "volume": float(volume),
            "delay": float(delay),
            "delay_ecc_cycles": float(delay * CODE_DISTANCE),
            "compilation_time_s": float(scheduler_wall),
            "wall_time_s": fair_wall,
            "process_wall_time_s": full_process_wall,
        },
        "scheduler_metrics": scheduler_metrics,
        "artifacts": {
            "schedule": str(schedule_copy.resolve()) if schedule_copy.exists() else None,
            "scheduler_log": str(log_path.resolve()),
        },
        "command": command,
    }


def aggregate_trials(trials: list[dict]) -> dict:
    keys = (
        "space", "time", "volume", "delay", "delay_ecc_cycles",
        "compilation_time_s", "wall_time_s", "process_wall_time_s",
    )
    aggregate: dict[str, float] = {}
    for key in keys:
        values = [float(trial["metrics"][key]) for trial in trials]
        aggregate[key] = float(statistics.fmean(values))
        aggregate[f"{key}_std"] = float(statistics.pstdev(values)) if len(values) > 1 else 0.0
        aggregate[f"{key}_min"] = min(values)
        aggregate[f"{key}_max"] = max(values)
    aggregate["trial_count"] = len(trials)
    return aggregate


def new_payload(selected: list[str], benchmark_dir: Path, args: argparse.Namespace) -> dict:
    mean_lcycles = 1.0 / args.magic_state_lambda
    return {
        "schema_version": 3,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "description": (
            "PureMagic results for the myTopoLS Part-B comparison: lambda=0.5 by default, "
            "magic routing enabled, and stochastic T-injection failures enabled."
        ),
        "selected_benchmarks": selected,
        "selected_methods": [METHOD_NAME],
        "benchmark_source_dir": str(benchmark_dir),
        "shared_puremagic_part_b_config": {
            "max_gates": args.max_gates,
            "routing": "PureMagic dual-purpose magic/ancilla routing",
            "use_magic_routing": True,
            "max_pauli_product_weight": MAX_PAULI_PRODUCT_WEIGHT,
            "magic_state_lambda": args.magic_state_lambda,
            "mean_cultivation_lcycles": mean_lcycles,
            "code_distance_for_unit_conversion": CODE_DISTANCE,
            "mean_cultivation_ecc_cycles": mean_lcycles * CODE_DISTANCE,
            "t_injection_failures": True,
            "t_injection_success_probability": 0.5,
            "seeds": args.seeds,
            "ancilla_rows": args.ancilla_rows,
            "timeout_s_per_benchmark": args.timeout_s,
            "metric_definition": {
                "space": "PureMagic topology nodes: data + bus + magic patches.",
                "time": "PureMagic scheduler lcycles; one lcycle is one logical cube layer.",
                "volume": "fixed topology-node count * all scheduled lcycles.",
                "delay": (
                    "Critical-path lcycles in which dependency-ready T products existed, every "
                    "magic node was still cultivating, and the scheduler made no useful progress. "
                    "T waits overlapped with useful work are not charged."
                ),
            },
            "runtime_definition": {
                "wall_time_s": (
                    "Internal Rust QASM-to-final-schedule time: transpiler timer + scheduler timer; "
                    "process startup and final schedule serialization are excluded."
                ),
                "compilation_time_s": "Scheduler-only internal timer; diagnostic.",
                "process_wall_time_s": "Full transpiler plus scheduler subprocess elapsed time.",
            },
        },
        "benchmarks": [],
    }


def write_payload(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def find_entry(payload: dict, stem: str) -> dict | None:
    return next((entry for entry in payload.get("benchmarks", []) if entry.get("stem") == stem), None)


def completed(entry: dict, seeds: list[int]) -> bool:
    for run in entry.get("runs", []):
        if run.get("method") != METHOD_NAME:
            continue
        if run.get("status") == "timeout":
            return True
        if run.get("status") == "ok":
            return {trial.get("seed") for trial in run.get("trials", [])} == set(seeds)
    return False


def timeout_run(timeout_s: float, error: Exception) -> dict:
    return {
        "method": METHOD_NAME,
        "label": METHOD_LABEL,
        "status": "timeout",
        "timeout_s": timeout_s,
        "error": str(error),
    }


def main() -> None:
    args = parse_args()
    args.benchmark_dir = args.benchmark_dir.expanduser().resolve()
    args.results_file = args.results_file.expanduser().resolve()
    args.transpiler = args.transpiler.expanduser().resolve()
    args.scheduler = args.scheduler.expanduser().resolve()
    if args.magic_state_lambda <= 0:
        raise ValueError("--magic-state-lambda must be positive")
    if args.ancilla_rows < 1:
        raise ValueError("--ancilla-rows must be at least 1")
    if args.timeout_s <= 0:
        raise ValueError("--timeout-s must be positive")
    if args.max_gates < 0:
        raise ValueError("--max-gates must be >= 0")
    if not args.seeds or len(set(args.seeds)) != len(args.seeds):
        raise ValueError("--seeds must be a non-empty list of unique integers")
    if not args.benchmark_dir.is_dir():
        raise FileNotFoundError(args.benchmark_dir)

    metadata_by_stem: dict[str, dict] = {}
    paths = common.benchmark_paths(args.benchmark_dir, args.benchmarks)
    print("Source:    ", args.benchmark_dir)
    print("Max gates: ", args.max_gates or "disabled")
    print("\n" + "=" * 84)
    print("INPUT CIRCUIT SUMMARY")
    print("=" * 84)
    for qasm in paths:
        stem = qasm.stem
        metadata = common.qasm_metadata(qasm)
        if args.max_gates and metadata["gate_count"] > args.max_gates:
            print(
                f"SKIP {stem:<36} gates={metadata['gate_count']:<7} "
                f"(limit {args.max_gates})"
            )
            continue
        if metadata["unsupported_gate_counts"]:
            raise RuntimeError(f"{stem}: unsupported gates {metadata['unsupported_gate_counts']}")
        metadata_by_stem[stem] = metadata
        print(
            f"RUN  {stem:<36} qubits={metadata['num_qubits']:<3} "
            f"gates={metadata['gate_count']:<5} depth={metadata['depth']:<5} "
            f"T={metadata['t_count']:<5}"
        )
    print("=" * 84)

    selected = list(metadata_by_stem)
    if not selected:
        print("\nNo circuits are within the gate limit; nothing to run.")
        return

    if args.build:
        build_release_binaries()
    for binary in (args.transpiler, args.scheduler):
        if not binary.exists():
            raise FileNotFoundError(binary)

    raw_dir = DEFAULT_RAW_DIR
    raw_dir.mkdir(parents=True, exist_ok=True)
    if args.resume and args.results_file.exists():
        payload = json.loads(args.results_file.read_text(encoding="utf-8"))
        config = payload.get("shared_puremagic_part_b_config", {})
        if config.get("seeds") is not None and config["seeds"] != args.seeds:
            raise ValueError(
                f"Cannot --resume with seeds {args.seeds}; JSON was created with {config['seeds']}"
            )
        if (
            config.get("magic_state_lambda") is not None
            and float(config["magic_state_lambda"]) != args.magic_state_lambda
        ):
            raise ValueError("Cannot --resume with a different --magic-state-lambda")
        if (
            config.get("ancilla_rows") is not None
            and int(config["ancilla_rows"]) != args.ancilla_rows
        ):
            raise ValueError("Cannot --resume with a different --ancilla-rows")
        if config.get("max_gates") not in (None, args.max_gates):
            raise ValueError("Cannot --resume with a different --max-gates")
        payload["selected_benchmarks"] = list(
            dict.fromkeys(payload.get("selected_benchmarks", []) + selected)
        )
        write_payload(args.results_file, payload)
    else:
        payload = new_payload(selected, args.benchmark_dir, args)

    print("Benchmarks:", ", ".join(selected))
    print("Seeds:     ", ", ".join(map(str, args.seeds)))
    print("Lambda:    ", args.magic_state_lambda)
    print("T failures: enabled")
    print("Timeout:   ", f"{args.timeout_s:.0f}s per benchmark pipeline")
    for stem in selected:
        qasm = args.benchmark_dir / f"{stem}.qasm"
        metadata = metadata_by_stem[stem]
        entry = find_entry(payload, stem)
        if entry is None:
            entry = {
                "stem": stem,
                "display_name": stem,
                **metadata,
                "runs": [],
            }
            payload["benchmarks"].append(entry)
            write_payload(args.results_file, payload)
        elif entry.get("qasm_sha256") != metadata["qasm_sha256"]:
            raise RuntimeError(
                f"{stem}: QASM changed since this JSON was created; use a new --results-file"
            )
        if args.resume and completed(entry, args.seeds):
            print(f"Skipping completed/timed-out {stem}")
            continue

        print(f"\n--- {stem} | PureMagic Part-B supply ---")
        deadline = time.perf_counter() + args.timeout_s
        try:
            transpiler = run_transpiler(
                stem,
                qasm,
                args,
                raw_dir,
                remaining_timeout(deadline, args.timeout_s, "transpiler"),
            )
            trials = []
            for seed in args.seeds:
                trials.append(
                    run_scheduler_trial(
                        stem,
                        seed,
                        transpiler["trans_file"],
                        transpiler["metrics"],
                        args,
                        raw_dir,
                        remaining_timeout(
                            deadline, args.timeout_s, f"scheduler seed {seed}"
                        ),
                    )
                )
            run = {
                "method": METHOD_NAME,
                "label": METHOD_LABEL,
                "status": "ok",
                "metrics": aggregate_trials(trials),
                "transpiler_metrics": transpiler["metrics"],
                "trials": trials,
                "artifacts": {
                    "clifford_t_qasm": str(transpiler["clifford_qasm"].resolve()),
                    "transpiled_circuit": str(transpiler["trans_file"].resolve()),
                    "transpiler_log": str(transpiler["log_path"].resolve()),
                },
                "transpile_command": transpiler["command"],
            }
        except TimeoutError as exc:
            print(f"TIMEOUT: {exc}")
            print("Continuing to the next benchmark.")
            run = timeout_run(args.timeout_s, exc)
        entry["runs"] = [r for r in entry.get("runs", []) if r.get("method") != METHOD_NAME] + [run]
        write_payload(args.results_file, payload)
        if run["status"] == "ok":
            print(
                f"{stem}: volume={run['metrics']['volume']:.3f}, "
                f"delay={run['metrics']['delay']:.3f} lcycles, "
                f"fair wall={run['metrics']['wall_time_s']:.6f}s"
            )

    print(f"\nSaved Part-B comparison JSON to {args.results_file}")
    print(f"Saved raw PureMagic logs and artifacts to {raw_dir}")


if __name__ == "__main__":
    main()
