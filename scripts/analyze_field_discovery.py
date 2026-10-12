#!/usr/bin/env python
"""Aggregate schema-v4 hybrid-alife field-discovery pilot artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from hybrid_alife.experiments.field_discovery import summarize_results


def collect_result_paths(inputs: list[Path]) -> list[Path]:
    paths = set()
    for item in inputs:
        if item.is_dir():
            paths.update(item.rglob("*.json"))
        elif item.is_file():
            paths.add(item)
        else:
            raise FileNotFoundError(item)
    return sorted(paths)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, default=None, help="JSON path; stdout if omitted")
    args = parser.parse_args()
    paths = collect_result_paths(args.inputs)
    if not paths:
        parser.error("no JSON result files found")
    results = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    payload = json.dumps(summarize_results(results), indent=2) + "\n"
    if args.output is None:
        print(payload, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
        print(f"Field-discovery summary written to {args.output}")


if __name__ == "__main__":
    main()
