#!/usr/bin/env python3
"""Paper-facing wrapper for PoseSelect post-cache evaluation."""

from runpy import run_module


if __name__ == "__main__":
    run_module("scripts.p6_pose_aware_candidate_selector_paired_eval", run_name="__main__")
