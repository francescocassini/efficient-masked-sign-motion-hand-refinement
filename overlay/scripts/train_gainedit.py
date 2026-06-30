#!/usr/bin/env python3
"""Paper-facing wrapper for the GainEdit baseline training."""

from runpy import run_module


if __name__ == "__main__":
    run_module("scripts.p6_hand_gain_regressor_train", run_name="__main__")
