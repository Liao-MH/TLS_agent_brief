from __future__ import annotations

import argparse

from .utils import ensure_dir
from .agent import run_agent
from .presets import get_presets

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("tls_toolkit")
    sub = p.add_subparsers(dest="cmd", required=True)

    pa = sub.add_parser("agent", help="Run agent: auto preset, full pipeline, optional rerun, report.")
    pa.add_argument("--wsi", required=True, help="Path to WSI")
    pa.add_argument("--out_root", required=True, help="Root output directory")
    pa.add_argument("--device", default="cuda", help="cuda or cpu")
    pa.add_argument("--config_file", required=True, help="mmseg config .py")
    pa.add_argument("--checkpoint", required=True, help="model checkpoint .pth")
    pa.add_argument("--max_reruns", type=int, default=1, help="Max reruns after QC")
    pa.add_argument(
        "--preset",
        default=None,
        choices=list(get_presets().keys()),
        help="Force a specific preset (disables fallback by default).",
    )
    pa.add_argument(
        "--allow_preset_fallback",
        action="store_true",
        default=False,
        help="When --preset is set, allow fallback to other presets if --max_reruns > 0.",
    )
    return p


def main():
    args = build_parser().parse_args()

    if args.cmd == "agent":
        out_root = ensure_dir(args.out_root)
        final_run_dir = run_agent(
            wsi_path=args.wsi,
            out_root=out_root,
            device=args.device,
            config_file=args.config_file,
            checkpoint=args.checkpoint,
            max_reruns=int(args.max_reruns),
            preset=args.preset,
            allow_preset_fallback=bool(args.allow_preset_fallback),
        )
        print(final_run_dir)


if __name__ == "__main__":
    main()
