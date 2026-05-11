#!/usr/bin/env python
"""Quick experiment campaign launcher (stub).

Orchestrates multiple `run_sim`-equivalent invocations across configs and seeds
for a small ablation matrix, writes a manifest, and supports --dry-run.

Stub version: parses arguments, prints the planned matrix, writes a manifest.
Full implementation follows in subsequent commits on this branch.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Quick campaign launcher (stub).")
    p.add_argument("--campaign", default="configs/campaigns/quick_20min.yaml")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--max-runs", type=int, default=None)
    p.add_argument("--seeds", type=int, nargs="+", default=None)
    p.add_argument("--generations", type=int, default=None)
    p.add_argument("--out-dir", default="outputs/quick_campaign")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[stub] quick campaign launcher (dry_run={args.dry_run})")
    print(f"[stub] would read {args.campaign}, write manifest to {out_dir}")


if __name__ == "__main__":
    main()
