from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from tracking.sonic_reference import (
    DEFAULT_ROOT_Z,
    VALID_JOINT_ORDERS,
    iter_sonic_motion_dirs,
    save_sonic_motion_npz,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert SONIC/GROOT reference motion CSV folders into "
            "Humanoid-GPT-compatible npz trajectories."
        )
    )
    parser.add_argument(
        "source",
        type=Path,
        help="A SONIC motion folder or a folder containing motion subfolders.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("storage/sonic_reference"),
        help="Directory used when converting one or more motion folders.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help="Exact .npz path when converting a single motion folder.",
    )
    parser.add_argument(
        "--frequency",
        type=float,
        default=50.0,
        help="Source motion frequency in Hz.",
    )
    parser.add_argument(
        "--joint-order",
        choices=VALID_JOINT_ORDERS,
        default="isaaclab",
        help=(
            "Input CSV joint order. SONIC reference CSVs are normally "
            "'isaaclab'. Use 'mujoco' or 'humanoid-gpt' if already remapped."
        ),
    )
    parser.add_argument(
        "--default-root-z",
        type=float,
        default=DEFAULT_ROOT_Z,
        help="Root height used when body_pos root z is all zero.",
    )
    parser.add_argument(
        "--preserve-zero-root-height",
        action="store_true",
        help="Keep zero root z values instead of filling --default-root-z.",
    )
    parser.add_argument(
        "--keep-root-xy",
        action="store_true",
        help="Do not rebase root x/y so the first frame starts at the origin.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively discover SONIC motion folders below source.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    motion_dirs = iter_sonic_motion_dirs(args.source, recursive=args.recursive)
    if not motion_dirs:
        raise SystemExit(f"No SONIC motion folders found under {args.source}")

    if args.output_path is not None and len(motion_dirs) != 1:
        raise SystemExit("--output-path can only be used with one motion folder")

    print(f"Found {len(motion_dirs)} SONIC motion(s).")
    for motion_dir in motion_dirs:
        output_path = (
            args.output_path
            if args.output_path is not None
            else args.output_dir / f"{motion_dir.name}.npz"
        )
        data = save_sonic_motion_npz(
            motion_dir,
            output_path,
            frequency=args.frequency,
            joint_order=args.joint_order,
            rebase_root_xy=not args.keep_root_xy,
            fill_zero_root_height=not args.preserve_zero_root_height,
            default_root_z=args.default_root_z,
        )
        qpos = data["qpos"]
        duration = len(qpos) / float(np.asarray(data["frequency"]))
        root_min = qpos[:, :3].min(axis=0)
        root_max = qpos[:, :3].max(axis=0)
        print(
            f"Saved {output_path} "
            f"frames={len(qpos)} duration={duration:.2f}s "
            f"root_min={root_min.round(4).tolist()} "
            f"root_max={root_max.round(4).tolist()} "
            f"height_filled={bool(data['root_height_filled'])}"
        )


if __name__ == "__main__":
    main()
