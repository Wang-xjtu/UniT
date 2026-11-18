"""Arguments and model loading for the RGB reconstruction demo."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch

from unit.inference.scheduling import make_schedule
from unit.utils.runtime import ensure_output_path, preflight_optional_features
from unit.visualization.export import DEFAULT_MAX_DEPTH, validate_max_depth


DEFAULT_REPOSITORY = "Haotian-sx/UniT"


def positive_int(value: str) -> int:
    """Parse an integer greater than zero, with a useful argparse error."""
    try:
        result = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a positive integer") from exc
    if result < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return result


def build_parser(
    *, project_root: str | Path | None = None,
) -> argparse.ArgumentParser:
    """Use application-provided asset defaults, or the caller's working directory."""
    root = Path.cwd() if project_root is None else Path(project_root).expanduser().resolve()
    parser = argparse.ArgumentParser(
        description="Reconstruct RGB images, view the scene, and export a point cloud.",
        allow_abbrev=False,
        epilog="For depth or camera observations, pass a dict to UniT.predict() in Python.",
    )
    parser.set_defaults(protected_examples=root / "examples")
    inputs = parser.add_argument_group("Input and weights")
    inputs.add_argument(
        "--images",
        default=str(root / "examples/HKUST_Toy/images"),
        help="Directory of images, sorted by filename (default: the bundled HKUST_Toy example)",
    )
    source = inputs.add_mutually_exclusive_group()
    source.add_argument(
        "--checkpoint", default=None,
        help="Local UniT checkpoint instead of downloading Hub weights",
    )
    source.add_argument(
        "--repo-id", default=DEFAULT_REPOSITORY,
        help=f"Hugging Face model repository (default: {DEFAULT_REPOSITORY})",
    )

    inference = parser.add_argument_group("Inference")
    inference.add_argument(
        "--group-size", type=positive_int,
        default=None, help="Frames per group: omit for offline; 1 for online",
    )
    inference.add_argument(
        "--queue-capacity", type=positive_int,
        default=None, help="KV-cache capacity in frames (default: selected frame count after stride)",
    )
    inference.add_argument(
        "--hybrid", action=argparse.BooleanOptionalAction,
        default=False, help="Hybrid offline-online inference: first group offline, then one frame at a time",
    )
    inference.add_argument(
        "--target-size", type=positive_int,
        default=518, help="Image resolution; a multiple of 14 (default: 518)",
    )
    inference.add_argument(
        "--preprocess-mode", choices=["crop", "pad"], default="crop",
        help="Crop tall images or pad images to a square after resizing (default: crop)",
    )
    inference.add_argument("--stride", type=positive_int, default=1, help="Keep every Nth sorted image")
    inference.add_argument("--device", default="auto", help="auto, cpu, cuda, or a device such as cuda:1")

    output = parser.add_argument_group("Output and point filtering")
    output.add_argument(
        "--output-dir",
        help="New directory for exported files (default: a fresh folder under ./outputs/)",
    )
    output.add_argument(
        "--save-ply", action=argparse.BooleanOptionalAction,
        default=True,
        help="Export a colored point cloud to scenes.ply (default: enabled)",
    )
    output.add_argument(
        "--conf-threshold", type=float,
        default=25.0, help="Percentage of lowest-confidence points to exclude from PLY/viewer (default: 25)",
    )
    output.add_argument(
        "--max-depth", type=float, default=DEFAULT_MAX_DEPTH,
        help=f"Maximum camera-z depth in meters for PLY/viewer (default: {DEFAULT_MAX_DEPTH:g}; 0 disables)",
    )

    viewer = parser.add_argument_group("Viewer")
    viewer.add_argument(
        "--no-viewer", dest="viewer", action="store_false", default=True,
        help="Export the point cloud without starting Viser",
    )
    viewer.add_argument("--port", type=positive_int, default=8080, help="Viewer port (default: 8080)")
    viewer.add_argument(
        "--mask-sky", action=argparse.BooleanOptionalAction,
        default=False, help="Filter sky in the viewer and its PLY export using an optional ONNX model",
    )
    viewer.add_argument(
        "--skyseg-path", default=str(root / "skyseg.onnx"),
        help="Existing sky-segmentation ONNX model",
    )
    viewer.add_argument(
        "--playback-group-size", type=positive_int, default=1,
        help="Frames displayed per playback step; independent of inference grouping",
    )
    return parser


def run_prediction(
    args: argparse.Namespace, image_paths: Sequence[str],
) -> dict[str, Any]:
    """Return predictions for the ordered image paths.

    Check inputs before loading weights. This function owns and releases the
    model before returning, so the viewer only needs the CPU predictions.
    """
    if args.target_size % 14:
        raise ValueError("target_size must be a positive multiple of 14")
    if not 0 <= args.conf_threshold <= 100:
        raise ValueError("conf_threshold must be between 0 and 100")
    validate_max_depth(args.max_depth)
    if args.mask_sky and not args.viewer:
        raise ValueError("Sky filtering requires the viewer. Remove --mask-sky or omit --no-viewer.")
    schedule = make_schedule(len(image_paths), args.group_size, args.queue_capacity, args.hybrid)
    if args.save_ply or args.mask_sky:
        ensure_output_path(args.output_dir, image_paths, protected_roots=[args.protected_examples])
        if args.output_dir is not None and Path(args.output_dir).expanduser().exists():
            raise FileExistsError(f"Output directory already exists: {args.output_dir}")
    preflight_optional_features(
        viewer=args.viewer, save_ply=args.save_ply, mask_sky=args.mask_sky,
        skyseg_path=args.skyseg_path,
    )
    from unit import UniT

    if args.checkpoint is not None:
        model = UniT.from_checkpoint(args.checkpoint, device=args.device)
    else:
        model = UniT.from_pretrained(args.repo_id, device=args.device)
    model_device = next(model.parameters()).device
    prediction_finished = False
    try:
        result = model.predict(
            {"images": image_paths}, group_size=args.group_size,
            queue_capacity=schedule.queue_capacity, hybrid=args.hybrid,
            target_size=args.target_size, preprocess_mode=args.preprocess_mode,
        )
        prediction_finished = True
    finally:
        # This entry point owns the model. Release it before starting a viewer
        # that may keep running for hours, and return unused CUDA memory.
        del model
        if model_device.type == "cuda":
            try:
                with torch.cuda.device(model_device):
                    torch.cuda.empty_cache()
            except Exception:
                # Preserve a prediction failure if CUDA cleanup fails as well.
                if prediction_finished:
                    raise
    return result
