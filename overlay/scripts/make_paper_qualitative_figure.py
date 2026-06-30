#!/usr/bin/env python3
"""Regenerate the selected paper qualitative figure."""

from runpy import run_module


if __name__ == "__main__":
    run_module("scripts.make_paper_contrastive_with_maskednar_figure", run_name="__main__")
