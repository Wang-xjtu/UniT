"""Point-cloud export helpers with optional dependencies loaded on demand."""

from pathlib import Path
from typing import Any

import numpy as np

from unit.utils.runtime import ensure_output_path

DEFAULT_MAX_DEPTH = 80.0


def validate_max_depth(max_depth: float) -> None:
    """Validate the optional camera-depth cutoff before loading or writing data."""
    if not np.isfinite(max_depth) or max_depth < 0:
        raise ValueError("max_depth must be finite and non-negative; use 0 to disable depth filtering")


def as_numpy(value: Any) -> np.ndarray:
    """Convert a NumPy array or CPU/GPU tensor without changing the input."""
    if hasattr(value, "detach"):
        return value.detach().float().cpu().numpy()
    return np.asarray(value)


def confidence_mask(
    confidence: np.ndarray, percentile: float, valid: np.ndarray | None = None,
) -> np.ndarray:
    """Keep valid points at or above the requested confidence percentile.

    Equal scores at the cutoff are kept. Near-zero scores are always dropped,
    including the zero scores assigned by sky filtering.
    """
    if not 0 <= percentile <= 100:
        raise ValueError("confidence percentile must be between 0 and 100")
    confidence = np.asarray(confidence)
    eligible = np.isfinite(confidence)
    if valid is not None:
        if valid.shape != confidence.shape:
            raise ValueError("Point validity and confidence shapes must match")
        eligible &= valid
    if not np.any(eligible):
        raise ValueError("No finite points with valid confidence are available")
    threshold = np.percentile(confidence[eligible], percentile)
    mask = eligible & (confidence >= threshold) & (confidence > 1e-5)
    if not np.any(mask):
        raise ValueError("No points remain above the confidence validity threshold")
    return mask


def point_validity(
    points: np.ndarray,
    confidence: np.ndarray,
    *,
    camera_depth: np.ndarray | None = None,
    max_depth: float = DEFAULT_MAX_DEPTH,
) -> np.ndarray:
    """Select finite geometry within an optional camera-Z limit, in meters.

    The cutoff includes its boundary. Zero disables depth filtering and retains
    the finite-point behavior without requiring local camera point maps.
    """
    validate_max_depth(max_depth)
    if points.shape != confidence.shape + (3,):
        raise ValueError("Point maps must match confidence dimensions with a final XYZ axis")
    valid = np.isfinite(points).all(axis=-1)
    if max_depth > 0:
        if camera_depth is None or camera_depth.shape != confidence.shape:
            raise ValueError("Depth filtering requires cam_points with the same shape as world_points")
        valid &= np.isfinite(camera_depth) & (camera_depth <= max_depth)
        if not np.any(valid):
            raise ValueError(
                f"No finite points remain within max_depth={max_depth:g} meters. "
                "Increase max_depth or use max_depth=0 to disable depth filtering."
            )
    return valid


def point_mask(
    points: np.ndarray,
    confidence: np.ndarray,
    percentile: float,
    *,
    camera_depth: np.ndarray | None = None,
    max_depth: float = DEFAULT_MAX_DEPTH,
) -> np.ndarray:
    """Apply the same depth, finite-value, and confidence filters for display/export."""
    valid = point_validity(points, confidence, camera_depth=camera_depth, max_depth=max_depth)
    return confidence_mask(confidence, percentile, valid)


def save_ply_fn(
    points: np.ndarray, colors: np.ndarray, masks: np.ndarray, filepath: str | Path,
) -> None:
    """Write a new PLY in the point maps' original coordinate system.

    Use points of shape ``[S, H, W, 3]``, RGB of shape ``[S, 3, H, W]``,
    and a boolean ``[S, H, W]`` mask with True for each point to export.
    """
    try:
        import trimesh
    except ImportError as exc:
        raise ImportError("PLY export requires: pip install -e '.[demo]'") from exc

    masks = masks.astype(bool)
    colors = np.transpose(colors, (0, 2, 3, 1))

    points_f = points[masks]
    colors_f = colors[masks]

    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    if colors_f.size > 0:
        if colors_f.max() <= 1.0:
            colors_u8 = (colors_f * 255.0).astype(np.uint8)
        else:
            colors_u8 = colors_f.astype(np.uint8)
    else:
        colors_u8 = colors_f.astype(np.uint8)

    cloud = trimesh.PointCloud(vertices=points_f, colors=colors_u8)
    with filepath.open("xb") as stream:
        stream.write(cloud.export(file_type="ply"))


def export_prediction_ply(
    result: dict[str, Any], path: str | Path, confidence_percent: float = 25.0,
    *, max_depth: float = DEFAULT_MAX_DEPTH,
) -> None:
    """Export filtered points in their original coordinates without changing results.

    ``max_depth`` limits predicted camera-Z depth in meters; 0 disables it.
    Destinations inside input-image directories and local examples are rejected.
    """
    destination = Path(path).expanduser().resolve()
    ensure_output_path(destination.parent, result.get("metadata", {}).get("image_paths") or [])
    validate_max_depth(max_depth)
    if not 0 <= confidence_percent <= 100:
        raise ValueError("confidence_percent must be between 0 and 100")
    points = as_numpy(result["world_points"])
    images = as_numpy(result["images"])
    confidence = as_numpy(result["cam_points_conf"])
    camera_depth = as_numpy(result["cam_points"])[..., 2] if max_depth > 0 else None
    mask = point_mask(
        points, confidence, confidence_percent, camera_depth=camera_depth, max_depth=max_depth,
    )
    save_ply_fn(points, images, mask, destination)
