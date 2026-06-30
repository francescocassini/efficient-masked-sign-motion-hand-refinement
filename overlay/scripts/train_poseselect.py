#!/usr/bin/env python3
"""Paper-facing wrapper for PoseSelect training."""

from runpy import run_module


if __name__ == "__main__":
    run_module("scripts.p6_pose_aware_candidate_selector_train", run_name="__main__")
