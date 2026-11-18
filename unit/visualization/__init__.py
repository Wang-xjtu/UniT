"""Optional visualization tools; imported only when a viewer or export is requested."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    import viser

__all__ = ["visualize"]


def visualize(
    predictions: dict[str, Any],
    *,
    port: int = 8080,
    init_conf_threshold: float = 50.0,
    background_mode: bool = False,
    mask_sky: bool = False,
    image_folder: str | Path | None = None,
    playback_group_size: int = 1,
    save_ply: bool = False,
    image_paths: Sequence[str | Path] | None = None,
    output_dir: str | Path | None = None,
    skyseg_path: str | Path | None = None,
    prediction_settings: dict[str, Any] | None = None,
) -> viser.ViserServer:
    """Explore a prediction dict returned by :meth:`UniT.predict` in Viser.

    Install the ``demo`` extra to use this function. Images, camera poses, and
    points are read directly from ``predictions``; no intermediate file is
    required. Sky filtering uses the original paths and transforms recorded in
    the prediction metadata and also requires the ``sky`` extra.

    ``init_conf_threshold`` discards this percentage of the lowest-confidence
    points. ``playback_group_size`` controls playback only, not inference grouping.
    With ``save_ply=True`` a point cloud is saved in a fresh output directory.
    The default output root is ``outputs/`` in the working directory.
    Existing outputs and input images are never overwritten.
    The default sky model is ``skyseg.onnx`` in the working directory.

    By default, this call keeps the viewer running. Set ``background_mode=True``
    to receive the server immediately, and call ``server.stop()`` when finished.
    """
    from unit.visualization.viewer import visualize_scene

    return visualize_scene(
        predictions,
        port=port,
        init_conf_threshold=init_conf_threshold,
        background_mode=background_mode,
        mask_sky=mask_sky,
        image_folder=image_folder,
        playback_group_size=playback_group_size,
        save_ply=save_ply,
        image_paths=image_paths,
        output_dir=output_dir,
        skyseg_path=skyseg_path,
        prediction_settings=prediction_settings,
    )
