# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Licensed under the license in the root directory of this source tree.

"""Run the public UniT inference API and open the optional Viser viewer."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from unit.inference.cli import build_parser, run_prediction
from unit.visualization import visualize


def main(argv: Sequence[str] | None = None):
    """Return the Viser server when --background-mode is enabled."""
    args = build_parser(viewer=True, project_root=Path(__file__).resolve().parent).parse_args(argv)
    result, image_paths, settings = run_prediction(args, viewer=True)
    return visualize(
        result, port=args.port, init_conf_threshold=args.conf_threshold,
        background_mode=args.background_mode, mask_sky=args.mask_sky,
        image_folder=args.images, playback_group_size=args.playback_group_size,
        save_ply=args.save_ply, image_paths=image_paths, output_dir=args.output_dir,
        skyseg_path=args.skyseg_path, prediction_settings=settings,
    )


if __name__ == "__main__":
    main()
