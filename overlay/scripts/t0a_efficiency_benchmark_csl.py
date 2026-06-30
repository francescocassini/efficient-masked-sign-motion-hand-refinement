#!/usr/bin/env python3
"""Run the isolated three-repetition T0-A benchmark on CSL-200."""

import t0a_efficiency_benchmark as benchmark


benchmark.OUTPUT = benchmark.ROOT / "artifacts/t0a_efficiency_results_csl"
benchmark.DATASET = "csl"
benchmark.RUNS = {
    "soke_ar": "configs/paper/table3_soke_ar_csl200.yaml",
    "p3": "configs/paper/table3_masked_nar_csl200.yaml",
    "p5": "configs/paper/table3_handpolish_csl200.yaml",
}


if __name__ == "__main__":
    benchmark.main()
