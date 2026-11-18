# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Licensed under the license in the root directory of this source tree.

"""Reconstruct RGB images, view the scene, and export a point cloud."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from unit.inference.cli import build_parser, run_prediction
from unit.utils.runtime import create_output_dir, list_image_paths


def main(argv: Sequence[str] | None = None) -> None:
    """Show and/or export a reconstruction of the selected frames."""
    project_root = Path(__file__).resolve().parent
    parser = build_parser(project_root=project_root)
    args = parser.parse_args(argv)
    if not args.viewer and not args.save_ply:
        parser.error("--no-viewer and --no-save-ply cannot be used together")
    image_paths = list_image_paths(args.images, stride=args.stride)
    predictions = run_prediction(args, image_paths)

    if args.viewer:
        from unit.visualization import visualize

        visualize(
            predictions,
            port=args.port,
            init_conf_threshold=args.conf_threshold,
            max_depth=args.max_depth,
            playback_group_size=args.playback_group_size,
            mask_sky=args.mask_sky,
            skyseg_path=args.skyseg_path,
            save_ply=args.save_ply,
            image_paths=image_paths,
            output_dir=args.output_dir,
        )
    else:
        from unit.visualization.export import export_prediction_ply

        ply_path = create_output_dir(args.output_dir) / "scenes.ply"
        export_prediction_ply(
            predictions, ply_path, args.conf_threshold, max_depth=args.max_depth,
        )
        print(f"Saved point cloud to {ply_path}")


if __name__ == "__main__":
    main()
