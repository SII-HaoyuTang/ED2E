#!/usr/bin/env python3
"""
Pack Stage 3 cache into a mmap-friendly store for training.

Examples
--------
python scripts/pack_stage3_cache.py \
    --stage3-source data/ed_energy_5w/cache_stage3 \
    --packed-dir data/ed_energy_5w/cache_stage3_packed

python scripts/pack_stage3_cache.py \
    --stage3-source data/ed_energy_5w/cache_stage3/all_stage3_lk12_ck8_a8.zip \
    --packed-dir data/ed_energy_5w/cache_stage3_packed
"""
from __future__ import annotations

import argparse
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

from ed2e.data.stage3_packed import pack_stage3_cache  # noqa: E402


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Pack Stage 3 pickle/bundle cache into a mmap-friendly store.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--stage3-source", required=True, help="Stage 3 cache dir, bundle zip, or single-sample pkl.")
    p.add_argument("--packed-dir", required=True, help="Output directory for the packed Stage 3 store.")
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--quiet", action="store_true")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    stats = pack_stage3_cache(
        args.stage3_source,
        args.packed_dir,
        max_samples=args.max_samples,
        overwrite=args.overwrite,
        show_progress=(not args.quiet),
    )
    print(
        f"Packed Stage 3 cache -> {args.packed_dir} | "
        f"samples={stats['num_samples']} nodes={stats['total_nodes']} "
        f"charts={stats['total_charts']} membership={stats['total_membership']}"
    )


if __name__ == "__main__":
    main()
