"""Export, bundle and serve the WebGL2 flow viewer.

    python scripts/run_viewer.py export --scene obstacle --out outputs/viewer/replay.hvr
    python scripts/run_viewer.py serve --replay outputs/viewer/replay.hvr
    python scripts/run_viewer.py standalone --replay outputs/viewer/replay.hvr --out viewer.html
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hybrid_alife.viewer.bundle import serve, standalone  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    commands = parser.add_subparsers(dest="command", required=True)
    export_args = commands.add_parser("export", help="run the simulation and record a replay")
    export_args.add_argument("--scene", choices=("obstacle", "periodic"), default="obstacle")
    export_args.add_argument("--steps", type=int, default=240)
    export_args.add_argument("--seed", type=int, default=3)
    export_args.add_argument("--grid", type=int, default=256)
    export_args.add_argument("--out", type=Path, default=Path("outputs/viewer/replay.hvr"))
    for name, text in (("serve", "serve the viewer over HTTP"),
                       ("standalone", "write one double-clickable HTML file")):
        sub = commands.add_parser(name, help=text)
        sub.add_argument("--replay", type=Path, default=Path("outputs/viewer/replay.hvr"))
        if name == "serve":
            sub.add_argument("--port", type=int, default=8765)
        else:
            sub.add_argument("--out", type=Path, default=Path("outputs/viewer/viewer.html"))
    args = parser.parse_args()
    if args.command == "export":
        from hybrid_alife.viewer.export import export

        path = export(args.out, args.scene, steps=args.steps, seed=args.seed, grid=args.grid)
        print(f"wrote {path} ({path.stat().st_size / 1e6:.1f} MB)")
    elif args.command == "serve":
        serve(args.replay, args.port)
    else:
        print(f"wrote {standalone(args.replay, args.out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
