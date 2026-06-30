#!/usr/bin/env python3
"""List the sanitized configs used for paper Table 2.

The native benchmark still runs through the repository's standard ``test.py``
entrypoint. This helper prints the paper-facing configs so a fresh checkout has
one stable command surface for the table.
"""

from pathlib import Path


CONFIGS = [
    "configs/paper/table2_soke_ar.yaml",
    "configs/paper/table2_masked_nar_direct.yaml",
    "configs/paper/table2_masked_nar_handpolish_base.yaml",
    "configs/paper/table2_handpolish.yaml",
]


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    print("Paper Table 2 native pose benchmark configs:")
    for config in CONFIGS:
        path = root / config
        status = "OK" if path.exists() else "MISSING"
        print(f"- {config} [{status}]")
    print("\nRun each config with:")
    print("  python -u -m test --cfg <config> --task t2m --nodebug")


if __name__ == "__main__":
    main()
