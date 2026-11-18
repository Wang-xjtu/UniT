"""Point-cloud export helpers with optional dependencies loaded on demand."""

from pathlib import Path
from typing import Any

import numpy as np

from unit.inference.io import ensure_output_path


def as_numpy(value: Any) -> np.ndarray:
    """Convert a NumPy array or CPU/GPU tensor without changing the input."""
    if hasattr(value, "detach"):
        return value.detach().float().cpu().numpy()
    return np.asarray(value)


def confidence_mask(
    confidence: np.ndarray, percentile: float, valid: np.ndarray | None = None,
) -> np.ndarray:
    """Filter finite confidences by percentile and the established validity floor.

    An optional validity mask excludes unusable geometry from the percentile.
    For entirely finite inputs this retains the original cutoff and tie behavior.
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


def point_mask(points: np.ndarray, confidence: np.ndarray, percentile: float) -> np.ndarray:
    """Select usable geometry consistently for the viewer and PLY export."""
    if points.shape != confidence.shape + (3,):
        raise ValueError("Point maps must match confidence dimensions with a final XYZ axis")
    return confidence_mask(confidence, percentile, np.isfinite(points).all(axis=-1))


def save_ply_fn(
    points: np.ndarray, colors: np.ndarray, masks: np.ndarray, filepath: str | Path,
) -> None:
    """Export a point cloud without replacing an existing file."""
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



def export_prediction_ply(result: dict[str, Any], path: str | Path, confidence_percent: float = 25.0) -> None:
    """Export to a new PLY outside input directories and local examples."""
    destination = Path(path).expanduser().resolve()
    ensure_output_path(destination.parent, result.get("metadata", {}).get("image_paths") or [])
    if not 0 <= confidence_percent <= 100:
        raise ValueError("confidence_percent must be between 0 and 100")
    points = as_numpy(result["world_points"])
    images = as_numpy(result["images"])
    confidence = as_numpy(result["cam_points_conf"])
    mask = point_mask(points, confidence, confidence_percent)
    save_ply_fn(points, images, mask, destination)
