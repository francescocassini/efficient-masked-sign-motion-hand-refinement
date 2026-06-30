#!/usr/bin/env python3
"""Aggregate matched-cache post-refinement results for paper Table 4."""

from runpy import run_module


if __name__ == "__main__":
    run_module("scripts.a3_p6k_r5_aggregate", run_name="__main__")
