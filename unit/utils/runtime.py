"""Runtime helpers for inference, resource checks, and safe output paths."""

from __future__ import annotations

from datetime import datetime
import importlib
from pathlib import Path
import uuid

import torch


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def list_image_paths(image_folder: str | Path, stride: int = 1) -> list[str]:
    """Return supported image files in filename order, sampled once by stride."""
    if isinstance(stride, bool) or not isinstance(stride, int) or stride < 1:
        raise ValueError("stride must be a positive integer")
    folder = Path(image_folder).expanduser()
    if not folder.is_dir():
        raise FileNotFoundError(f"Image folder does not exist: {folder}")
    paths = sorted(
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )[::stride]
    if not paths:
        raise ValueError(f"No supported images found in {folder}")
    return [str(p.resolve()) for p in paths]


def inference_precision(device: str | torch.device) -> torch.dtype:
    """Use BF16 where supported, FP16 on other CUDA devices, and FP32 on CPU."""
    device = torch.device(device)
    if device.type == "cuda":
        with torch.cuda.device(device):
            return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    if device.type == "cpu":
        return torch.float32
    raise ValueError("The inference examples support CUDA and CPU devices")


def resolve_output_path(path: str | Path | None = None) -> Path:
    """Resolve an explicit destination or the calling project's output root."""
    return (Path.cwd() / "outputs" if path is None else Path(path).expanduser()).resolve()


def resolve_sky_model_path(path: str | Path | None = None) -> Path:
    """Resolve the sky model in the calling project, never in the installed package."""
    return (Path.cwd() / "skyseg.onnx" if path is None else Path(path).expanduser()).resolve()


def preflight_optional_features(
    *, viewer: bool = False, save_ply: bool = False, mask_sky: bool = False,
    skyseg_path: str | Path | None = None,
) -> None:
    """Check requested dependencies/resources before loading or writing results."""
    requirements = []
    if viewer:
        requirements.append(("viser", "demo"))
    if save_ply:
        requirements.append(("trimesh", "demo"))
    if mask_sky:
        model_path = resolve_sky_model_path(skyseg_path)
        if not model_path.is_file():
            raise FileNotFoundError(
                f"Sky segmentation model not found: {model_path}. "
                "Supply skyseg_path (CLI: --skyseg-path)."
            )
        requirements.extend((("cv2", "sky"), ("onnxruntime", "sky")))
    for name, extra in requirements:
        try:
            importlib.import_module(name)
        except (ImportError, OSError) as error:
            raise ImportError(
                f"Requested feature requires {name}: pip install -e '.[{extra}]'"
            ) from error


def create_output_dir(path: str | Path | None = None, prefix: str = "inference") -> Path:
    """Create a fresh output directory; never reuse or overwrite an existing run."""
    if path is None:
        run_name = f"{prefix}-{datetime.now():%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:8]}"
        destination = resolve_output_path() / run_name
    else:
        destination = resolve_output_path(path)
    destination.mkdir(parents=True, exist_ok=False)
    return destination
