# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Licensed under the license in the root directory of this source tree.

"""Reconstruct RGB images and save predictions, with an optional Viser viewer."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from unit.inference.cli import build_parser, run_prediction
from unit.inference.io import save_predictions
from unit.utils.runtime import create_output_dir, list_image_paths


def main(argv: Sequence[str] | None = None) -> Path:
    """Reconstruct the selected frames and return the new output directory."""
    project_root = Path(__file__).resolve().parent
    args = build_parser(project_root=project_root).parse_args(argv)
    if args.save_ply is None:
        args.save_ply = args.viewer
    image_paths = list_image_paths(args.images, stride=args.stride)
    predictions, settings = run_prediction(args, image_paths)

    # Save the complete prediction before applying any display or PLY filters.
    output_dir = create_output_dir(args.output_dir)
    save_predictions(predictions, output_dir, settings=settings)
    print(f"Saved predictions for {len(image_paths)} frames to {output_dir}")

    if args.viewer:
        from unit.visualization.viewer import visualize_scene

        visualize_scene(
            predictions,
            port=args.port,
            init_conf_threshold=args.conf_threshold,
            max_depth=args.max_depth,
            playback_group_size=args.playback_group_size,
            mask_sky=args.mask_sky,
            skyseg_path=args.skyseg_path,
            save_ply=args.save_ply,
            image_paths=image_paths,
            run_dir=output_dir,
        )
    elif args.save_ply:
        from unit.visualization.export import export_prediction_ply

        export_prediction_ply(
            predictions, output_dir / "scenes.ply", args.conf_threshold,
            max_depth=args.max_depth,
        )
    return output_dir


if __name__ == "__main__":
    main()
