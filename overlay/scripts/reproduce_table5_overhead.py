#!/usr/bin/env python3
"""Run the post-cache overhead benchmark for paper Table 5."""

from runpy import run_module


if __name__ == "__main__":
    run_module("scripts.benchmark_p6k_t0a_style", run_name="__main__")
