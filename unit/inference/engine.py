"""Grouped prediction from prepared observations to common-coordinate geometry."""

from __future__ import annotations

from collections.abc import Mapping

import torch

from unit.inference._cache import release_caches
from unit.inference.scheduling import GroupSchedule
from unit.inference.postprocess import align_predictions_to_observations
from unit.utils.geometry import (
    intrinsics_to_camera_rays,
    express_predictions_in_reference_frame,
    to_hmat,
    closed_form_inverse_se3,
)


def _encode_group_observations(
    group: dict[str, torch.Tensor],
    reference_camera_to_world: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode rays/depth and poses relative to the first observed camera.

    Missing modalities use zero tensors. The group's pose observations are
    expressed in the same reference coordinates as the model inputs so optional
    postprocessing can use them directly.
    """
    images = group["images"]
    batch_size, frames, _, height, width = images.shape
    intrinsics = group.get("intrinsic")
    if intrinsics is not None:
        rays = intrinsics_to_camera_rays(
            intrinsics.reshape(batch_size * frames, 3, 3), (height, width)
        ).view(batch_size, frames, height, width, 3)
        rays = rays.to(images.dtype)
    else:
        rays = torch.zeros(
            batch_size, frames, height, width, 3,
            device=images.device, dtype=images.dtype,
        )
    depths = group.get("depth")
    if depths is None:
        depths = torch.zeros(
            batch_size, frames, height, width,
            device=images.device, dtype=images.dtype,
        )
    ray_depth = torch.cat([rays[..., :2], depths.unsqueeze(-1)], dim=-1)

    extrinsics = group.get("extrinsic")
    if extrinsics is not None:
        extrinsics = torch.matmul(
            to_hmat(extrinsics), reference_camera_to_world.unsqueeze(1)
        )[:, :, :3]
        group["extrinsic"] = extrinsics
    else:
        extrinsics = torch.zeros(
            batch_size, frames, 3, 4, device=images.device, dtype=images.dtype,
        )
    return ray_depth, extrinsics


def _align_group_predictions(
    predictions: dict[str, torch.Tensor],
    observations: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Apply optional observation-based corrections after network prediction."""
    geometry = [observations.get(name) for name in ("depth", "intrinsic", "extrinsic")]
    geometry = [value if value is not None and torch.any(value != 0) else None
                for value in geometry]
    predictions["cam_points"], predictions["extrinsics"] = align_predictions_to_observations(
        predictions["cam_points"], predictions["extrinsics"],
        *geometry, predictions["cam_points_conf"],
    )
    return predictions


def _inference_cache_dtype(
    model: torch.nn.Module, device: torch.device, fallback_dtype: torch.dtype,
) -> torch.dtype:
    """Match backbone caches to the dtype used by attention projections."""
    device_type = device.type
    if torch.is_autocast_enabled(device_type):
        return torch.get_autocast_dtype(device_type)
    for parameter in model.parameters():
        if parameter.is_floating_point():
            return parameter.dtype
    return fallback_dtype


def _collect_cpu_predictions(
    outputs: dict[str, list[torch.Tensor]], predictions: dict[str, torch.Tensor],
) -> None:
    """Collect completed groups without keeping model-device tensor references."""
    for name, value in predictions.items():
        outputs.setdefault(name, []).append(value.detach().cpu())


def _validate_prepared_batch(
    batch: Mapping[str, torch.Tensor | None], schedule: GroupSchedule,
) -> tuple[int, int, int, int]:
    """Check prepared tensor shapes and scheduling before allocating caches."""
    if not isinstance(batch, Mapping):
        raise ValueError("Prepared inputs must be a dictionary of tensors")
    unknown = set(batch) - {"images", "depth", "intrinsic", "extrinsic"}
    if unknown:
        names = ", ".join(sorted(repr(name) for name in unknown))
        raise ValueError(f"Unknown prepared input fields: {names}")
    images = batch.get("images")
    if not isinstance(images, torch.Tensor):
        raise ValueError("images must be a Tensor with shape [B, S, 3, H, W]")
    if images.ndim != 5 or images.shape[2] != 3:
        raise ValueError("images must have shape [B, S, 3, H, W]")
    batch_size, frames, _, height, width = images.shape
    if min(batch_size, frames, height, width) <= 0:
        raise ValueError("images must contain a non-empty batch and sequence of images")
    if schedule.frames != frames:
        raise ValueError(
            f"schedule.frames ({schedule.frames}) must match the images frame count ({frames})"
        )
    shapes = {
        "depth": ((batch_size, frames, height, width),),
        "intrinsic": ((batch_size, frames, 3, 3),),
        "extrinsic": ((batch_size, frames, 3, 4), (batch_size, frames, 4, 4)),
    }
    for name, expected in shapes.items():
        value = batch.get(name)
        if value is None:
            continue
        if not isinstance(value, torch.Tensor):
            raise ValueError(f"{name} must be a prepared Tensor or None")
        if tuple(value.shape) not in expected:
            dimensions = " or ".join(str(shape) for shape in expected)
            raise ValueError(f"{name} must have shape {dimensions}, got {tuple(value.shape)}")
    return batch_size, frames, height, width


@torch.inference_mode()
def run_inference(
    batch: dict[str, torch.Tensor | None],
    model: torch.nn.Module,
    schedule: GroupSchedule,
    *,
    patch_size: int = 14,
    postprocess: bool = False,
    show_progress: bool = False,
) -> dict[str, torch.Tensor]:
    """Infer a prepared batch without changing the caller's model mode/precision.

    Optional prepared geometry has shapes depth [B, S, H, W], intrinsic
    [B, S, 3, 3], and extrinsic [B, S, 3, 4] or [B, S, 4, 4]. The schedule
    must describe the same number of frames as the batch.

    Prepared inputs remain on CPU. Each iteration transfers one group to the
    model device, predicts and aligns geometry there, then collects completed
    outputs on CPU. Only caches and reference poses persist on the model device.
    Caches are released even after partial initialization or a failed forward.
    Returning the complete sequence still requires CPU memory proportional to
    its length; offline mode processes the full sequence as one group.
    """
    batch_size, frames, height, width = _validate_prepared_batch(batch, schedule)
    images = batch["images"]
    if not any(name in batch for name in ("depth", "intrinsic", "extrinsic")):
        postprocess = False

    unwrapped_model = model.module if hasattr(model, "module") else model
    parameter = next(unwrapped_model.parameters(), None)
    device = parameter.device if parameter is not None else images.device
    dtype = parameter.dtype if parameter is not None else images.dtype
    observed_reference = None
    if batch.get("extrinsic") is not None:
        first_pose = batch["extrinsic"][:, :1].to(device=device, dtype=dtype)
        observed_reference = closed_form_inverse_se3(to_hmat(first_pose)[:, 0])
        del first_pose
    inference_finished = False
    try:
        if schedule.use_cache:
            unwrapped_model.setup_caches(
                batch_size, height, width, patch_size,
                _inference_cache_dtype(unwrapped_model, device, dtype), device,
                schedule.queue_capacity,
            )
        outputs: dict[str, list[torch.Tensor]] = {}
        predicted_reference = None
        start = step = 0
        while start < frames:
            end = min(start + schedule.group_at(step), frames)
            group = {
                name: value[:, start:end].to(device=device, dtype=dtype)
                for name, value in batch.items() if value is not None
            }
            ray_depth, extrinsics = _encode_group_observations(group, observed_reference)
            predictions = model(
                images=group["images"], ray_depth=ray_depth,
                extrinsics=extrinsics, group_size=end - start,
            )
            with torch.autocast(device_type=device.type, enabled=False):
                if postprocess:
                    predictions = _align_group_predictions(predictions, group)
                predictions, predicted_reference = express_predictions_in_reference_frame(
                    predictions, predicted_reference,
                )
            # Finish the device-to-host copies before processing the next group.
            _collect_cpu_predictions(outputs, predictions)
            del predictions, group, ray_depth, extrinsics
            if show_progress:
                print(
                    f"Progress: ({end / frames * 100:.1f}%), Current Group Size: {end - start}",
                    flush=True,
                )
            start = end
            step += 1

        predictions = {name: torch.cat(values, dim=1) for name, values in outputs.items()}
        inference_finished = True
        return predictions
    finally:
        if schedule.use_cache:
            try:
                unwrapped_model.reset_caches()
            except Exception:
                # A reset failure must not hide the original inference error.
                if inference_finished:
                    raise
            finally:
                release_caches(unwrapped_model)
