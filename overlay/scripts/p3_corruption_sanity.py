#!/usr/bin/env python3
"""Monte Carlo sanity check for P3 BERT-style corruption probabilities."""
import argparse

import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=1_000_000)
    parser.add_argument("--mask-prob", type=float, default=0.8)
    parser.add_argument("--random-prob", type=float, default=0.1)
    args = parser.parse_args()

    draws = torch.rand(args.samples)
    mask = draws < args.mask_prob
    random = (draws >= args.mask_prob) & (
        draws < args.mask_prob + args.random_prob
    )
    keep = ~(mask | random)
    print(f"mask={mask.float().mean():.5f}")
    print(f"random={random.float().mean():.5f}")
    print(f"keep={keep.float().mean():.5f}")


if __name__ == "__main__":
    main()
