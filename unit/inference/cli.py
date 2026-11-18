"""Shared command-line interface for prediction and the optional Viser viewer."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Sequence

import torch

from unit.inference.io import ensure_output_path, save_predictions
from unit.inference.scheduling import make_schedule
from unit.utils.runtime import create_output_dir, list_image_paths, preflight_optional_features


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
    *, viewer: bool = False, project_root: str | Path | None = None,
) -> argparse.ArgumentParser:
    """Use application-provided asset defaults, or the caller's working directory."""
    root = Path.cwd() if project_root is None else Path(project_root).expanduser().resolve()
    parser = argparse.ArgumentParser(
        description="UniT RGB reconstruction" + (" with an interactive Viser viewer" if viewer else ""),
        allow_abbrev=False,
        epilog="For depth or camera observations, pass a dict to UniT.predict() in Python.",
    )
    parser.set_defaults(
        default_checkpoint=str(root / "checkpoints/checkpoint.pt"),
        protected_examples=root / "examples",
    )
    parser.add_argument(
        "--images", dest="images",
        default=str(root / "examples/HKUST_Toy/images"),
        help="Directory of images, sorted by filename",
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--checkpoint", default=None,
        help="Local UniT checkpoint (default: checkpoints/checkpoint.pt)",
    )
    source.add_argument(
        "--repo-id", help="Load UniT weights from a Hugging Face model repository",
    )
    parser.add_argument(
        "--group-size", dest="group_size", type=positive_int,
        default=None, help="Frames per group: omit for offline; 1 for online",
    )
    parser.add_argument(
        "--queue-capacity", dest="queue_capacity", type=positive_int,
        default=None, help="KV-cache capacity in frames (default: selected frame count after stride)",
    )
    parser.add_argument(
        "--hybrid", dest="hybrid", action=argparse.BooleanOptionalAction,
        default=False, help="Hybrid offline-online inference: first group offline, then one frame at a time",
    )
    parser.add_argument(
        "--target-size", dest="target_size", type=positive_int,
        default=518, help="Image resolution; a multiple of 14 (default: 518)",
    )
    parser.add_argument("--preprocess-mode", choices=["crop", "pad"], default="crop")
    parser.add_argument("--stride", type=positive_int, default=1, help="Keep every Nth sorted image")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or a device such as cuda:1")
    parser.add_argument(
        "--output-dir", dest="output_dir",
        help="New results directory; existing directories are never reused",
    )
    parser.add_argument(
        "--save-ply", dest="save_ply", action=argparse.BooleanOptionalAction,
        default=viewer, help="Export a point cloud (requires the demo extra)",
    )
    parser.add_argument(
        "--conf-threshold", dest="conf_threshold", type=float,
        default=25.0, help="Percentage of lowest-confidence points to exclude from PLY/viewer",
    )
    if viewer:
        parser.add_argument("--port", type=positive_int, default=8080)
        parser.add_argument(
            "--background-mode", dest="background_mode",
            action=argparse.BooleanOptionalAction, default=False,
            help="Return the server to Python callers without blocking",
        )
        parser.add_argument(
            "--mask-sky", dest="mask_sky", action=argparse.BooleanOptionalAction,
            default=False, help="Filter sky points using the optional ONNX model",
        )
        parser.add_argument(
            "--skyseg-path", default=str(root / "skyseg.onnx"),
            help="Existing sky-segmentation ONNX model",
        )
        parser.add_argument(
            "--playback-group-size", type=positive_int, default=1,
            help="Frames displayed per playback step; independent of inference grouping",
        )
    return parser


def run_prediction(args: argparse.Namespace, *, viewer: bool = False) -> tuple[dict[str, Any], list[str], dict[str, Any]]:
    """Select RGB frames and call the public sequence-dict inference API."""
    if args.target_size % 14:
        raise ValueError("target_size must be a positive multiple of 14")
    if not 0 <= args.conf_threshold <= 100:
        raise ValueError("conf_threshold must be between 0 and 100")
    image_paths = list_image_paths(args.images, stride=args.stride)
    schedule = make_schedule(len(image_paths), args.group_size, args.queue_capacity, args.hybrid)
    ensure_output_path(args.output_dir, image_paths, protected_roots=[args.protected_examples])
    if args.output_dir is not None:
        if Path(args.output_dir).expanduser().exists():
            raise FileExistsError(f"Output directory already exists: {args.output_dir}")
    preflight_optional_features(
        viewer=viewer, save_ply=args.save_ply, mask_sky=getattr(args, "mask_sky", False),
        skyseg_path=getattr(args, "skyseg_path", None),
    )
    from unit import UniT

    if args.repo_id is not None:
        model = UniT.from_pretrained(args.repo_id, device=args.device)
        source = {"repo_id": args.repo_id}
    else:
        checkpoint = args.checkpoint
        if checkpoint is None:
            checkpoint = args.default_checkpoint
        model = UniT.from_checkpoint(checkpoint, device=args.device)
        source = {"checkpoint": str(Path(checkpoint).expanduser().resolve())}
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
                if prediction_finished:
                    raise
    settings = {
        **source,
        "group_size": args.group_size, "queue_capacity": schedule.queue_capacity,
        "hybrid": args.hybrid, "target_size": args.target_size,
        "preprocess_mode": args.preprocess_mode,
        "stride": args.stride, "device": args.device,
    }
    return result, image_paths, settings


def main(
    argv: Sequence[str] | None = None, *, project_root: str | Path | None = None,
) -> Path:
    """Predict and save results without starting a visualization server."""
    args = build_parser(project_root=project_root).parse_args(argv)
    result, image_paths, settings = run_prediction(args)
    output_dir = create_output_dir(args.output_dir)
    save_predictions(result, output_dir, settings=settings)
    if args.save_ply:
        from unit.visualization.export import export_prediction_ply

        export_prediction_ply(result, output_dir / "scenes.ply", args.conf_threshold)
    print(f"Saved {len(image_paths)} frames to {output_dir}")
    return output_dir
