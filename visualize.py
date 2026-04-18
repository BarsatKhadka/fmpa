"""
Visualize the initial.plc placement for a benchmark.

Usage:
    uv run python visualize.py -ibm1
    uv run python visualize.py -ibm01
    uv run python visualize.py -ibm12 --save
"""

import sys
import os
import argparse

# Normalize names like "ibm1" -> "ibm01", "ibm18" -> "ibm18"
IBM_NAMES = [
    "ibm01", "ibm02", "ibm03", "ibm04", "ibm06",
    "ibm07", "ibm08", "ibm09", "ibm10", "ibm11",
    "ibm12", "ibm13", "ibm14", "ibm15", "ibm16",
    "ibm17", "ibm18",
]

ICCAD_ROOT = os.path.join(
    os.path.dirname(__file__),
    "external", "MacroPlacement", "Testcases", "ICCAD04",
)


def resolve_name(raw: str) -> str:
    """Accept ibm1/ibm01/IBM1/IBM01 and return canonical ibm0X."""
    raw = raw.lstrip("-").lower()
    if not raw.startswith("ibm"):
        raise ValueError(f"Unknown benchmark: {raw!r}")
    num = raw[3:]  # e.g. "1", "01", "12"
    padded = f"ibm{int(num):02d}"
    if padded not in IBM_NAMES:
        raise ValueError(f"No benchmark named {padded!r}. Available: {IBM_NAMES}")
    return padded


def main():
    # Pull out the benchmark name first (handles leading dash like -ibm1)
    raw_args = sys.argv[1:]
    bench_raw = None
    remaining = []
    for a in raw_args:
        lower = a.lstrip("-").lower()
        if lower.startswith("ibm") and bench_raw is None:
            bench_raw = a
        else:
            remaining.append(a)

    if bench_raw is None:
        print("Usage: python visualize.py -ibm1 [--save] [--out path]", file=sys.stderr)
        sys.exit(1)

    parser = argparse.ArgumentParser()
    parser.add_argument("--save", action="store_true")
    parser.add_argument("--out", default=None)
    args = parser.parse_args(remaining)

    name = resolve_name(bench_raw)
    bench_dir = os.path.join(ICCAD_ROOT, name).replace("\\", "/")

    if not os.path.isdir(bench_dir):
        print(f"Error: benchmark directory not found: {bench_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading {name} from {bench_dir} ...")
    from macro_place.loader import load_benchmark_from_dir
    from macro_place.utils import visualize_placement

    benchmark, plc = load_benchmark_from_dir(bench_dir)

    save_path = None
    if args.save:
        save_path = args.out or f"{name}_placement.png"

    print(f"Visualizing {name} ({benchmark.num_hard_macros} hard macros, "
          f"{benchmark.num_soft_macros} soft macros) ...")
    visualize_placement(benchmark.macro_positions, benchmark, save_path=save_path, plc=plc)


if __name__ == "__main__":
    main()
