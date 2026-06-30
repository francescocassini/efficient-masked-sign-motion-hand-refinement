#!/usr/bin/env python3
"""List the end-to-end efficiency benchmark entrypoints for paper Table 3."""

from pathlib import Path


SCRIPTS = [
    "scripts/t0a_efficiency_benchmark.py",
    "scripts/t0a_efficiency_benchmark_csl.py",
]


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    print("Paper Table 3 efficiency benchmark scripts:")
    for script in SCRIPTS:
        path = root / script
        status = "OK" if path.exists() else "MISSING"
        print(f"- {script} [{status}]")
    print("\nRun the Phoenix and CSL scripts under the documented CUDA environment.")


if __name__ == "__main__":
    main()
