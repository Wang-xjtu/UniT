# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE.txt file in the root directory of this source tree.
# Modified for UniT.

"""Interactive reconstruction viewer with camera controls and temporal playback."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
import logging
import threading
import time
from typing import Any

import numpy as np
try:
    import viser
    import viser.transforms as viser_tf
except ImportError as exc:
    raise ImportError("The viewer requires: pip install -e '.[demo]'") from exc

from unit.inference.io import ensure_output_path, save_predictions
from unit.utils.geometry import closed_form_inverse_se3
from unit.utils.runtime import create_output_dir, preflight_optional_features
from unit.visualization.camera import quat_to_rot
from unit.visualization.export import as_numpy, point_mask, save_ply_fn
from unit.visualization.sky import apply_sky_segmentation

logger = logging.getLogger(__name__)


def _scene_arrays(
    images: np.ndarray, world_points: np.ndarray, extrinsics: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Flatten colored points and center points/cameras without altering inputs."""
    frames, height, width, _ = world_points.shape
    points = world_points.reshape(-1, 3)
    colors = (images.transpose(0, 2, 3, 1).reshape(-1, 3) * 255).astype(np.uint8)
    camera_to_world = closed_form_inverse_se3(extrinsics)[:, :3, :].copy()
    finite = np.isfinite(points).all(axis=-1)
    if not np.any(finite):
        raise ValueError("No finite points are available for visualization")
    center = np.mean(points[finite], axis=0)
    camera_to_world[..., -1] -= center
    frame_indices = np.repeat(np.arange(frames), height * width)
    return points - center, colors, camera_to_world, frame_indices


def _frame_visibility(
    frame_groups: np.ndarray, *, playing: bool, current_only: bool,
    current_group: int, selected_frame: str,
) -> np.ndarray:
    """Choose all, one, or the current/cumulative playback groups of frames."""
    if playing:
        if current_only:
            return frame_groups == current_group
        return frame_groups <= current_group
    if selected_frame == "All":
        return np.ones(len(frame_groups), dtype=bool)
    return np.arange(len(frame_groups)) == int(selected_frame)


def _representative_frame(
    frame_count: int, group_size: int, *, playing: bool,
    current_group: int, selected_frame: str,
) -> int:
    """Choose the middle frame of the current group for camera following."""
    if playing:
        start = current_group * group_size
        end = min(frame_count - 1, (current_group + 1) * group_size - 1)
        index = (start + end) // 2
    else:
        index = 0 if selected_frame == "All" else int(selected_frame)
    return max(0, min(frame_count - 1, index))


def _follow_pose(
    frame_position: np.ndarray, frame_rotation: np.ndarray,
    position: np.ndarray | None, rotation: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Smooth a trailing viewpoint using the established translation/SLERP rates."""
    offset = np.array([-0.6, -0.5, -1.8], dtype=np.float32)
    target_position = frame_position + quat_to_rot(frame_rotation) @ offset
    target_rotation = frame_rotation.copy()
    if position is None:
        position = target_position.copy()
        rotation = target_rotation.copy()
    translation_smoothing = 0.05
    rotation_smoothing = 0.08
    position = ((1.0 - translation_smoothing) * position
                + translation_smoothing * target_position)

    dot = float(np.dot(rotation, target_rotation))
    if dot < 0.0:
        target_rotation = -target_rotation
        dot = -dot
    if dot > 0.9995:
        rotation = ((1.0 - rotation_smoothing) * rotation
                    + rotation_smoothing * target_rotation)
        rotation = rotation / np.linalg.norm(rotation)
    else:
        theta = np.arccos(np.clip(dot, -1.0, 1.0))
        sin_theta = np.sin(theta)
        source_weight = np.sin((1.0 - rotation_smoothing) * theta) / sin_theta
        target_weight = np.sin(rotation_smoothing * theta) / sin_theta
        rotation = source_weight * rotation + target_weight * target_rotation
    return position, rotation


def _add_camera_frames(
    server: viser.ViserServer, camera_to_world: np.ndarray, images: np.ndarray,
) -> tuple[list[viser.FrameHandle], list[viser.CameraFrustumHandle]]:
    """Create camera axes and clickable frusta with approximate display FoV."""
    frames, frustums = [], []

    def attach_camera(frustum, frame):
        @frustum.on_click
        def on_click(_) -> None:
            for client in server.get_clients().values():
                client.camera.wxyz = frame.wxyz
                client.camera.position = frame.position

    for index, (pose, image) in enumerate(zip(camera_to_world, images)):
        matrix = np.eye(4, dtype=camera_to_world.dtype)
        matrix[:3, :] = pose
        transform = viser_tf.SE3.from_matrix(matrix)
        frame = server.scene.add_frame(
            f"frame_{index}", wxyz=transform.rotation().wxyz,
            position=transform.translation(), axes_length=0.05,
            axes_radius=0.002, origin_radius=0.002,
        )
        display_image = (image.transpose(1, 2, 0) * 255).astype(np.uint8)
        height, width = display_image.shape[:2]
        frustum = server.scene.add_camera_frustum(
            f"frame_{index}/frustum", fov=2 * np.arctan2(height / 2, 1.1 * height),
            aspect=width / height, scale=0.1, image=display_image, line_width=1.0,
        )
        frames.append(frame)
        frustums.append(frustum)
        attach_camera(frustum, frame)
    return frames, frustums


def _stop_after_failure(server: viser.ViserServer) -> None:
    """Release a running server without masking the exception being handled."""
    try:
        server.stop()
    except Exception:
        logger.exception("Failed to stop Viser while handling a viewer error")


def _run_viewer_loop(
    server: viser.ViserServer, tick: Callable[[], None], background: bool,
) -> None:
    """Run playback and close the service on normal or exceptional termination."""
    stop_event = threading.Event()
    stop_lock = threading.Lock()
    original_stop = server.stop
    thread = None

    def stop_server() -> None:
        with stop_lock:
            if stop_event.is_set():
                return
            stop_event.set()
        try:
            if thread is not None and thread is not threading.current_thread() and thread.is_alive():
                thread.join(timeout=1.0)
        finally:
            original_stop()

    server.stop = stop_server
    if background:
        def server_loop() -> None:
            try:
                while not stop_event.is_set():
                    tick()
                    stop_event.wait(0.005)
            except Exception:
                logger.exception("Viser playback failed; stopping the server")
            finally:
                _stop_after_failure(server)

        thread = threading.Thread(target=server_loop, daemon=True)
        thread.start()
    else:
        try:
            while not stop_event.is_set():
                tick()
                stop_event.wait(0.01)
        except KeyboardInterrupt:
            stop_server()
        except BaseException:
            _stop_after_failure(server)
            raise
        else:
            stop_server()


def visualize_scene(
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
    """Build a viewer for a prediction dict; public callers use ``visualize``.

    Display recentering affects viewer arrays only. Optional exports retain the
    original prediction coordinates and are written into a new output directory.
    ``playback_group_size`` controls display grouping, independently of inference.
    """
    if playback_group_size <= 0:
        raise ValueError("playback_group_size must be positive")
    if not 0 <= init_conf_threshold <= 100:
        raise ValueError("init_conf_threshold must be between 0 and 100")
    metadata = predictions.get("metadata", {})
    image_paths = metadata.get("image_paths") if image_paths is None else image_paths
    images = as_numpy(predictions["images"])
    world_points = as_numpy(predictions["world_points"])
    confidence = as_numpy(predictions["cam_points_conf"])
    extrinsics = as_numpy(predictions["extrinsics"])
    frame_count = len(images)
    if frame_count == 0:
        raise ValueError("At least one predicted frame is required")
    if mask_sky and image_paths is None:
        raise ValueError("Sky filtering requires the exact image_paths used for inference")
    if image_paths is not None and len(image_paths) != frame_count:
        raise ValueError("image_paths must contain one path per predicted frame")
    transforms = metadata.get("transforms")
    if mask_sky and (transforms is None or len(transforms) != frame_count):
        raise ValueError("Sky filtering requires preprocessing metadata from UniT.predict()")

    # Reject unusable predictions before allocating a service or creating output files.
    point_mask(world_points, confidence, init_conf_threshold)

    preflight_optional_features(save_ply=save_ply, mask_sky=mask_sky, skyseg_path=skyseg_path)

    if save_ply or mask_sky or prediction_settings is not None:
        protected_paths = list(image_paths or [])
        if image_folder is not None:
            protected_paths.append(str(Path(image_folder) / "_input_frame_"))
        ensure_output_path(output_dir, protected_paths)
        run_dir = create_output_dir(output_dir, prefix="viser")
        if prediction_settings is not None:
            save_predictions(predictions, run_dir, settings=prediction_settings)
        print(f"Writing generated files to {run_dir}")
        if mask_sky:
            confidence = apply_sky_segmentation(
                confidence, image_paths, run_dir / "sky_masks",
                skyseg_path=skyseg_path, transforms=transforms,
            )
        if save_ply:
            save_ply_fn(
                world_points, images, point_mask(world_points, confidence, init_conf_threshold),
                run_dir / "scenes.ply",
            )

    points, colors, camera_to_world, frame_indices = _scene_arrays(
        images, world_points, extrinsics,
    )
    confidence = confidence.reshape(-1)
    # Frame membership is fixed for this viewer; no per-tick point-group allocation.
    frame_groups = np.arange(frame_count) // playback_group_size
    group_count = int(np.ceil(frame_count / playback_group_size))
    anim_state = {"last_tick": time.time(), "current_group": 0}
    follow_state = {"position": None, "rotation": None}

    print(f"Starting viser server on port {port}")
    server = viser.ViserServer(host="0.0.0.0", port=port)
    try:
        server.gui.configure_theme(titlebar_content=None, control_layout="collapsible")
        show_cameras = server.gui.add_checkbox("Show Cameras", initial_value=True)
        confidence_percent = server.gui.add_slider(
            "Confidence Percent", min=0, max=100, step=0.1,
            initial_value=init_conf_threshold,
        )
        camera_size = server.gui.add_slider(
            "Camera Size", min=0.01, max=1.0, step=0.01, initial_value=0.1,
        )
        point_size = server.gui.add_slider(
            "Point Size", min=0.001, max=0.05, step=0.0001, initial_value=0.005,
        )
        frame_selector = server.gui.add_dropdown(
            "Show Points from Frames",
            options=["All"] + [str(i) for i in range(frame_count)], initial_value="All",
        )
        playing = server.gui.add_checkbox("Playing", initial_value=False)
        current_only = server.gui.add_checkbox("Current View", initial_value=False)
        fps = server.gui.add_slider("FPS", min=1, max=60, step=1, initial_value=10)
        follow_camera = server.gui.add_checkbox("Follow Camera", initial_value=False)

        initial_mask = point_mask(points, confidence, init_conf_threshold)
        point_cloud = server.scene.add_point_cloud(
            name="viser_pcd", points=points[initial_mask], colors=colors[initial_mask],
            point_size=0.005, point_shape="circle",
        )
        frames, frustums = _add_camera_frames(server, camera_to_world, images)

        def update_point_cloud() -> None:
            visible_frames = _frame_visibility(
                frame_groups, playing=playing.value, current_only=current_only.value,
                current_group=int(anim_state["current_group"]), selected_frame=frame_selector.value,
            )
            mask = point_mask(points, confidence, confidence_percent.value) & visible_frames[frame_indices]
            point_cloud.points = points[mask]
            point_cloud.colors = colors[mask]
            for index, (frame, frustum) in enumerate(zip(frames, frustums)):
                frame.visible = frustum.visible = bool(visible_frames[index]) and show_cameras.value

        def update_camera_follow() -> None:
            if not follow_camera.value or not frames:
                follow_state["position"] = follow_state["rotation"] = None
                return
            index = _representative_frame(
                frame_count, playback_group_size, playing=playing.value,
                current_group=int(anim_state["current_group"]), selected_frame=frame_selector.value,
            )
            frame = frames[index]
            position, rotation = _follow_pose(
                frame.position, frame.wxyz, follow_state["position"], follow_state["rotation"],
            )
            follow_state["position"], follow_state["rotation"] = position, rotation
            for client in server.get_clients().values():
                client.camera.position = position
                client.camera.wxyz = rotation

        for control in (confidence_percent, frame_selector, show_cameras, current_only):
            control.on_update(lambda _: update_point_cloud())

        @camera_size.on_update
        def update_camera_size(_) -> None:
            for frustum in frustums:
                frustum.scale = camera_size.value

        @point_size.on_update
        def update_point_size(_) -> None:
            point_cloud.point_size = point_size.value

        @fps.on_update
        def update_fps(_) -> None:
            anim_state["last_tick"] = time.time()

        @playing.on_update
        def update_playing(_) -> None:
            anim_state["current_group"] = 0
            update_point_cloud()

        def tick() -> None:
            if not playing.value:
                return
            now = time.time()
            if now - anim_state["last_tick"] >= 1.0 / max(1, int(fps.value)):
                anim_state["current_group"] = (anim_state["current_group"] + 1) % max(1, group_count)
                anim_state["last_tick"] = now
                update_point_cloud()
            update_camera_follow()

        _run_viewer_loop(server, tick, background_mode)
    except BaseException:
        _stop_after_failure(server)
        raise
    return server
