#!/usr/bin/env python3
"""Paper-facing wrapper for the GainEdit baseline evaluation."""

from runpy import run_module


if __name__ == "__main__":
    run_module("scripts.p6_hand_gain_regressor_paired_eval", run_name="__main__")
