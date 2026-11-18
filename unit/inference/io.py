"""Save inference results without overwriting inputs or previous outputs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from unit.utils.runtime import resolve_output_path


def ensure_output_path(
    output_dir: str | Path | None, image_paths: Sequence[str | Path],
    *, protected_roots: Sequence[str | Path] = (),
) -> Path | None:
    """Reject output paths inside input-image directories or application examples.

    Call this before creating the output directory.
    ``protected_roots`` lets an application protect examples outside the working
    directory. ``create_output_dir`` rejects existing destinations; result files
    also use exclusive creation so repeated saves cannot overwrite a result.
    """
    destination = resolve_output_path(output_dir)
    protected = {(Path.cwd() / "examples").resolve()}
    protected.update(Path(folder).expanduser().resolve() for folder in protected_roots)
    protected.update(Path(image).expanduser().resolve().parent for image in image_paths)
    if any(destination.is_relative_to(folder) for folder in protected):
        raise ValueError("output_dir must be outside input image directories and example directories")
    return None if output_dir is None else destination


def _json_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    raise TypeError(f"Cannot encode {type(value).__name__} in result metadata")


def save_predictions(
    result: Mapping[str, Any],
    output_dir: str | Path,
    settings: Mapping[str, Any] | None = None,
) -> Path:
    """Save the full prediction from ``UniT.predict`` and return its directory.

    Create ``output_dir`` outside the input image directories before calling.
    ``predictions.npz`` contains float32 arrays; the JSON files record the pixel
    transforms, frame order, and optional run settings. Existing result files
    are never overwritten.
    """
    metadata = result.get("metadata", {})
    destination = Path(output_dir).expanduser().resolve()
    ensure_output_path(destination, metadata.get("image_paths") or [])
    if not destination.is_dir():
        raise FileNotFoundError(f"Create a fresh output directory before saving: {destination}")
    arrays = {
        key: value.detach().float().cpu().numpy()
        for key, value in result.items()
        if isinstance(value, torch.Tensor)
    }
    if not arrays or not all(np.isfinite(array).all() for array in arrays.values()):
        raise ValueError("Inference returned empty or non-finite predictions")
    targets = [
        destination / name for name in ("predictions.npz", "metadata.json", "frames.json", "run.json")
    ]
    if any(path.exists() for path in targets):
        raise FileExistsError(f"Refusing to overwrite predictions in {destination}")
    metadata_json = json.dumps(metadata, indent=2, default=_json_value, allow_nan=False) + "\n"
    settings_json = json.dumps(dict(settings or {}), indent=2, default=_json_value, allow_nan=False) + "\n"
    frames_json = json.dumps(metadata.get("frame_names", []), indent=2) + "\n"
    with targets[0].open("xb") as stream:
        np.savez_compressed(stream, **arrays)
    for path, content in zip(targets[1:], (metadata_json, frames_json, settings_json)):
        with path.open("x", encoding="utf-8") as stream:
            stream.write(content)
    return destination
