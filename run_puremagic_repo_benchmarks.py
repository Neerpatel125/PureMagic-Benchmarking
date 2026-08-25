#!/usr/bin/env python3
"""Benchmark PureMagic on the shared LS-Benchmarking QASM circuits.

Run from the PureMagic repository root:

    python3 run_puremagic_repo_benchmarks.py

The JSON result is written directly to the sibling LS-Benchmarking-Results
repository by default.

Circuits above 10,000 gates are skipped by default. Change the cutoff with
``--max-gates``; use 0 to disable it.

The default configuration uses PureMagic routing, lightweight PBC (omega=1),
disabled stochastic T-injection failures, and the repository's high-production
setting (-m 100).  This is the readily-available-magic configuration used to
match static compilers in the PureMagic paper's page-7 comparison.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_BENCHMARK_DIR = (
    REPO_ROOT.parent / "LS-Benchmarking-Results" / "Benchmarks" / "QASM"
)
DEFAULT_RESULTS = (
    REPO_ROOT.parent
    / "LS-Benchmarking-Results"
    / "results"
    / "puremagic_repo_results.json"
)
RAW_DIR = REPO_ROOT / "results" / "benchmarking" / "raw_puremagic"
DEFAULT_TRANSPILE = REPO_ROOT / "target" / "release" / "transpile"
DEFAULT_SCHEDULER = REPO_ROOT / "target" / "release" / "puremagic"

METHOD_NAME = "puremagic_ready_magic"
METHOD_LABEL = "PureMagic (ready magic)"
MAX_PAULI_PRODUCT_WEIGHT = 1
MAGIC_STATE_LAMBDA = 100.0
SUPPORTED_GATES = {"cx", "cz", "swap", "h", "s", "sdg", "sx", "sxdg", "x", "y", "z", "t", "tdg"}
DISPLAY_GATE_NAMES = {"cx": "CNOT", "h": "HAD", "t": "T", "s": "S"}
ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;]*m")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run PureMagic on the shared LS-Benchmarking QASM circuits."
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
    parser.add_argument("--seed", type=int, default=29)
    parser.add_argument("--ancilla-rows", type=int, default=1)
    parser.add_argument(
        "--build", action=argparse.BooleanOptionalAction, default=True,
        help="Build release binaries before benchmarking (default: true).",
    )
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def qasm_metadata(path: Path) -> dict:
    qreg_pattern = re.compile(r"^qreg\s+\w+\[(\d+)\]\s*;")
    gate_pattern = re.compile(r"^([A-Za-z][A-Za-z0-9_]*)\s+(.+);$")
    qubit_pattern = re.compile(r"\[(\d+)\]")
    num_qubits = 0
    gate_counts: dict[str, int] = {}
    last_layer: list[int] = []
    unsupported: dict[str, int] = {}

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("//", 1)[0].strip()
        if not line or line.startswith(("OPENQASM", "include", "creg", "barrier", "measure")):
            continue
        qreg_match = qreg_pattern.match(line)
        if qreg_match:
            num_qubits = int(qreg_match.group(1))
            last_layer = [0] * num_qubits
            continue
        gate_match = gate_pattern.match(line)
        if not gate_match:
            continue
        gate = gate_match.group(1).lower()
        qubits = [int(value) for value in qubit_pattern.findall(gate_match.group(2))]
        if not qubits:
            continue
        display_gate = DISPLAY_GATE_NAMES.get(gate, gate.upper())
        gate_counts[display_gate] = gate_counts.get(display_gate, 0) + 1
        if gate not in SUPPORTED_GATES:
            unsupported[gate] = unsupported.get(gate, 0) + 1
        if last_layer:
            layer = max(last_layer[q] for q in qubits) + 1
            for qubit in qubits:
                last_layer[qubit] = layer

    return {
        "qasm_path": str(path.resolve()),
        "qasm_sha256": sha256_file(path),
        "num_qubits": num_qubits,
        "gate_count": sum(gate_counts.values()),
        "depth": max(last_layer, default=0),
        "gate_counts": gate_counts,
        "t_count": gate_counts.get("T", 0) + gate_counts.get("TDG", 0),
        "unsupported_gate_counts": unsupported,
    }


def benchmark_paths(benchmark_dir: Path, requested: list[str] | None) -> list[Path]:
    if requested:
        paths = [benchmark_dir / f"{stem}.qasm" for stem in requested]
        missing = [path for path in paths if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"Missing benchmark: {missing[0]}")
        return paths
    paths = sorted(benchmark_dir.glob("*.qasm"))
    if not paths:
        raise RuntimeError(f"No QASM files found in {benchmark_dir}")
    return paths


def write_payload(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def tee_process(command: list[str], *, cwd: Path, log_path: Path) -> tuple[str, float]:
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
        for line in process.stdout:
            print(line, end="")
            log.write(line)
            lines.append(line)
        returncode = process.wait()
    elapsed = time.perf_counter() - start
    output = "".join(lines)
    if returncode != 0:
        raise RuntimeError(
            f"Command failed with exit code {returncode}. See {log_path}\n{' '.join(command)}"
        )
    return output, elapsed


def parse_required(pattern: str, output: str, description: str) -> re.Match[str]:
    match = re.search(pattern, ANSI_ESCAPE.sub("", output), re.MULTILINE)
    if not match:
        raise RuntimeError(f"Could not parse {description} from PureMagic output.")
    return match


def parse_benchmark_wall_time(output: str, stage: str) -> float:
    match = parse_required(
        r"^Benchmark wall time:\s*([0-9.eE+-]+)\s*$",
        output,
        f"{stage} benchmark wall time",
    )
    return float(match.group(1))


def parse_transpiler_output(output: str) -> dict:
    circuit = parse_required(
        r"Circuit has\s+(\d+)\s+gates on\s+(\d+)\s+qubits", output, "input circuit size"
    )
    post = parse_required(
        r"Circuit length:\s+(\d+)\s+\(before\)\s+->\s+(\d+)\s+\(after transpilation\)",
        output,
        "transpiled operation count",
    )
    written = parse_required(r"Wrote\s+(\d+)\s+T gates and\s+(\d+)\s+Cliffords", output, "output gate counts")
    average = parse_required(r"Average Pauli product weight:\s*([0-9.eE+-]+)", output, "average Pauli weight")
    return {
        "input_gate_count": int(circuit.group(1)),
        "num_qubits": int(circuit.group(2)),
        "transpiled_operation_count": int(post.group(2)),
        "transpiled_t_count": int(written.group(1)),
        "transpiled_clifford_count": int(written.group(2)),
        "average_pauli_product_weight": float(average.group(1)),
    }


def parse_scheduler_output(output: str) -> dict:
    scheduled = parse_required(
        r"Scheduled\s+(\d+)\s+in\s+(\d+)\s+logical cycles, volume\s+(\d+)",
        output,
        "scheduled volume",
    )
    total = parse_required(r"^\s*total:\s+(\d+)\s*$", output, "topology qubit count")
    data = parse_required(r"^\s*data:\s+(\d+)\s+", output, "data-patch count")
    bus = parse_required(r"^\s*bus:\s+(\d+)\s+", output, "bus-patch count")
    magic = parse_required(r"^\s*magic:\s+(\d+)\s+", output, "magic-patch count")
    loaded = parse_required(r"Loaded circuit with\s+(\d+)\s+products and\s+(\d+)\s+qubits", output, "loaded circuit")
    parallelism = parse_required(r"Parallelism:\s*([0-9.eE+-]+)x", output, "parallelism")
    efficiency = parse_required(r"Normalized scheduling efficiency:\s*([0-9.eE+-]+)", output, "efficiency")
    failures = parse_required(r"T gate failures:\s*(\d+)/(\d+)", output, "T gate failures")
    cultivation = re.search(
        r"Magic state cultivation time:.*?^\s*average:\s*([0-9.eE+-]+)",
        ANSI_ESCAPE.sub("", output),
        re.MULTILINE | re.DOTALL,
    )
    return {
        "scheduled_product_count": int(scheduled.group(1)),
        "logical_cycles": int(scheduled.group(2)),
        "volume": int(scheduled.group(3)),
        "topology_qubit_count": int(total.group(1)),
        "data_patch_count": int(data.group(1)),
        "bus_patch_count": int(bus.group(1)),
        "magic_patch_count": int(magic.group(1)),
        "transpiled_product_count": int(loaded.group(1)),
        "circuit_qubit_count": int(loaded.group(2)),
        "parallelism": float(parallelism.group(1)),
        "normalized_scheduling_efficiency": float(efficiency.group(1)),
        "t_gate_failure_count": int(failures.group(1)),
        "t_gate_count": int(failures.group(2)),
        "average_cultivation_time": float(cultivation.group(1)) if cultivation else None,
    }


def build_release_binaries() -> None:
    command = ["cargo", "build", "--release", "--bin", "transpile", "--bin", "puremagic"]
    print("Building PureMagic release binaries...")
    subprocess.run(command, cwd=REPO_ROOT, check=True)


def run_one(stem: str, source_qasm: Path, args: argparse.Namespace) -> dict:
    display = stem
    clifford_qasm = RAW_DIR / f"{stem}.cliffordt.qasm"
    trans_file = RAW_DIR / f"{stem}.trans"
    transpile_log = RAW_DIR / f"{stem}__transpile.log"
    scheduler_log = RAW_DIR / f"{stem}__puremagic.log"
    schedule_file = RAW_DIR / f"{stem}.schedule"
    shutil.copy2(source_qasm, clifford_qasm)

    transpile_command = [
        str(args.transpiler.resolve()),
        "--input_file", str(clifford_qasm),
        "--output_file", str(trans_file),
        "--max_width", str(MAX_PAULI_PRODUCT_WEIGHT),
    ]
    scheduler_command = [
        str(args.scheduler.resolve()),
        "--circuit", str(trans_file),
        "--use-magic-routing",
        "--magic-state-lambda", str(MAGIC_STATE_LAMBDA),
        "--no-t-failures",
        "--rseed", str(args.seed),
        "--ancilla-rows", str(args.ancilla_rows),
    ]

    print(f"\n--- {display} | {METHOD_LABEL} ---")
    process_wall_start = time.perf_counter()
    transpile_stdout, transpile_process_wall_time_s = tee_process(
        transpile_command, cwd=RAW_DIR, log_path=transpile_log
    )
    scheduler_stdout, scheduler_process_wall_time_s = tee_process(
        scheduler_command, cwd=RAW_DIR, log_path=scheduler_log
    )
    process_wall_time_s = time.perf_counter() - process_wall_start
    transpiler_metrics = parse_transpiler_output(transpile_stdout)
    scheduler_metrics = parse_scheduler_output(scheduler_stdout)
    transpiler_wall_time_s = parse_benchmark_wall_time(transpile_stdout, "transpiler")
    scheduler_wall_time_s = parse_benchmark_wall_time(scheduler_stdout, "scheduler")
    transpiler_metrics["benchmark_wall_time_s"] = transpiler_wall_time_s
    transpiler_metrics["process_wall_time_s"] = transpile_process_wall_time_s
    scheduler_metrics["benchmark_wall_time_s"] = scheduler_wall_time_s
    scheduler_metrics["process_wall_time_s"] = scheduler_process_wall_time_s

    space = scheduler_metrics["topology_qubit_count"]
    logical_time = scheduler_metrics["logical_cycles"]
    volume = scheduler_metrics["volume"]
    if space * logical_time != volume:
        raise RuntimeError(f"{stem}: expected volume {space} * {logical_time}, got {volume}")
    patch_breakdown = sum(
        scheduler_metrics[key]
        for key in ("data_patch_count", "bus_patch_count", "magic_patch_count")
    )
    if patch_breakdown != space:
        raise RuntimeError(
            f"{stem}: topology breakdown sums to {patch_breakdown}, expected {space}"
        )
    if scheduler_metrics["t_gate_failure_count"] != 0:
        raise RuntimeError(f"{stem}: T gate failures were not disabled")

    pipeline_wall_time_s = transpiler_wall_time_s + scheduler_wall_time_s
    metrics = {
        "space": float(space),
        "time": float(logical_time),
        "volume": float(volume),
        "compilation_time_s": float(scheduler_wall_time_s),
        "wall_time_s": float(pipeline_wall_time_s),
        "process_wall_time_s": float(process_wall_time_s),
    }
    print(
        f"space={space}, time={logical_time}, space-time={volume}, "
        f"transpile={transpiler_wall_time_s:.3f}s, schedule={scheduler_wall_time_s:.3f}s, "
        f"fair-wall={pipeline_wall_time_s:.3f}s, process-wall={process_wall_time_s:.3f}s"
    )

    return {
        "method": METHOD_NAME,
        "label": METHOD_LABEL,
        "status": "ok",
        "metrics": metrics,
        "transpiler_metrics": transpiler_metrics,
        "scheduler_metrics": scheduler_metrics,
        "artifacts": {
            "clifford_t_qasm": str(clifford_qasm),
            "transpiled_circuit": str(trans_file),
            "schedule": str(schedule_file),
            "transpiler_log": str(transpile_log),
            "scheduler_log": str(scheduler_log),
        },
        "transpile_command": transpile_command,
        "command": scheduler_command,
    }


def new_payload(selected: list[str], benchmark_dir: Path, args: argparse.Namespace) -> dict:
    return {
        "schema_version": 2,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "description": (
            "PureMagic results on the shared LS-Benchmarking circuits with readily "
            "available magic states and deterministic T injection."
        ),
        "selected_benchmarks": selected,
        "selected_methods": [METHOD_NAME],
        "benchmark_source_dir": str(benchmark_dir),
        "shared_puremagic_config": {
            "max_gates": args.max_gates,
            "routing": "PureMagic dual-purpose ancilla routing",
            "use_magic_routing": True,
            "max_pauli_product_weight": MAX_PAULI_PRODUCT_WEIGHT,
            "magic_state_lambda": MAGIC_STATE_LAMBDA,
            "magic_state_assumption": (
                "Readily available magic states via -m 100, the high-production setting "
                "used to approximate the paper's one-cycle cultivation comparison."
            ),
            "t_injection_failures": False,
            "random_seed": args.seed,
            "ancilla_rows": args.ancilla_rows,
            "metric_definition": {
                "space": (
                    "Number of nodes in the generated logical-patch topology: data patches "
                    "+ bus patches + magic patches. With --use-magic-routing, magic patches "
                    "also serve as routing ancillas."
                ),
                "time": (
                    "Number of PureMagic scheduling lcycles. Each lcycle packs mutually "
                    "compatible routed Pauli products; this is a compiler scheduling step, "
                    "not classical wall-clock time."
                ),
                "volume": (
                    "space * time, i.e. the fixed topology-node count multiplied by all "
                    "scheduled lcycles. Magic-state factory hardware outside the modeled "
                    "magic patches is not included."
                ),
            },
            "runtime_definition": {
                "wall_time_s": (
                    "Fair QASM-to-final-in-memory-schedule wall time: internal Rust transpiler "
                    "timer plus internal Rust scheduler timer. Starts before QASM parsing, "
                    "includes the required intermediate .trans write/read, and excludes CLI "
                    "startup and final .schedule serialization."
                ),
                "compilation_time_s": (
                    "Internal PureMagic scheduler timer after transpilation; diagnostic only."
                ),
                "process_wall_time_s": (
                    "Full outer two-stage pipeline interval, including both process startups, "
                    "the inter-process handoff, and artifact serialization."
                ),
            },
        },
        "benchmarks": [],
    }


def find_entry(payload: dict, stem: str) -> dict | None:
    return next((entry for entry in payload.get("benchmarks", []) if entry.get("stem") == stem), None)


def completed(entry: dict) -> bool:
    return any(
        run.get("method") == METHOD_NAME and run.get("status", "ok") == "ok"
        for run in entry.get("runs", [])
    )


def main() -> None:
    args = parse_args()
    benchmark_dir = args.benchmark_dir.expanduser().resolve()
    results_file = args.results_file.expanduser().resolve()
    args.transpiler = args.transpiler.expanduser().resolve()
    args.scheduler = args.scheduler.expanduser().resolve()

    if args.ancilla_rows < 1:
        raise ValueError("--ancilla-rows must be at least 1")
    if args.max_gates < 0:
        raise ValueError("--max-gates must be >= 0")
    if not benchmark_dir.is_dir():
        raise FileNotFoundError(f"Benchmark directory not found: {benchmark_dir}")

    metadata: dict[str, dict] = {}
    paths = benchmark_paths(benchmark_dir, args.benchmarks)
    print("\nSource:    ", benchmark_dir)
    print("Max gates: ", args.max_gates or "disabled")
    print("\n" + "=" * 84)
    print("INPUT CIRCUIT SUMMARY")
    print("=" * 84)
    for qasm in paths:
        stem = qasm.stem
        meta = qasm_metadata(qasm)
        if args.max_gates and meta["gate_count"] > args.max_gates:
            print(
                f"SKIP {stem:<36} gates={meta['gate_count']:<7} "
                f"(limit {args.max_gates})"
            )
            continue
        if meta["unsupported_gate_counts"]:
            raise RuntimeError(f"{stem} has unsupported gates: {meta['unsupported_gate_counts']}")
        metadata[stem] = meta
        print(
            f"RUN  {stem:<36} qubits={meta['num_qubits']:<3} "
            f"gates={meta['gate_count']:<5} depth={meta['depth']:<5} T={meta['t_count']:<5} "
            f"sha256={meta['qasm_sha256'][:12]}..."
        )
    print("=" * 84)

    selected = list(metadata)
    if not selected:
        print("\nNo circuits are within the gate limit; nothing to run.")
        return

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    if args.build:
        build_release_binaries()
    for binary in (args.transpiler, args.scheduler):
        if not binary.is_file():
            raise FileNotFoundError(f"PureMagic binary not found: {binary}")

    if args.resume and results_file.exists():
        payload = json.loads(results_file.read_text(encoding="utf-8"))
        print(f"Resuming from {results_file}")
    else:
        payload = new_payload(selected, benchmark_dir, args)

    print("\nBenchmarks:", ", ".join(selected))
    print("Method:    ", METHOD_LABEL)

    for stem in selected:
        entry = find_entry(payload, stem)
        if entry is None:
            entry = {
                "stem": stem,
                "display_name": stem,
                **metadata[stem],
                "runs": [],
            }
            payload["benchmarks"].append(entry)
            write_payload(results_file, payload)
        if args.resume and completed(entry):
            print(f"\nSkipping completed {stem} | {METHOD_LABEL}")
            continue
        run = run_one(stem, benchmark_dir / f"{stem}.qasm", args)
        entry["runs"] = [item for item in entry.get("runs", []) if item.get("method") != METHOD_NAME]
        entry["runs"].append(run)
        write_payload(results_file, payload)

    print(f"\nSaved JSON results to: {results_file}")
    print(f"Raw PureMagic artifacts/logs: {RAW_DIR}")


if __name__ == "__main__":
    main()
