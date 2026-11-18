# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE.txt file in the root directory of this source tree.
# Modified for UniT.
"""Prepare RGB images and geometric prompts in one shared pixel coordinate system."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from unit.utils.runtime import list_image_paths

ImageInput = str | Path | Sequence[str | Path | Image.Image | np.ndarray]


def _as_numpy(value: Any, name: str) -> np.ndarray:
    try:
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu()
            # NumPy has no BF16 dtype; preserve other supported tensor dtypes.
            if value.dtype == torch.bfloat16:
                value = value.float()
            value = value.numpy()
        array = np.asarray(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a numeric array") from error
    if array.dtype.kind not in "iuf":
        raise ValueError(f"{name} must be a numeric array")
    return array


def _rgb_image(value: str | Path | Image.Image | np.ndarray) -> Image.Image:
    if isinstance(value, (str, Path)):
        with Image.open(Path(value).expanduser()) as opened:
            image = opened.copy()
    elif isinstance(value, Image.Image):
        image = value.copy()
    elif isinstance(value, np.ndarray):
        if value.ndim != 3 or value.shape[-1] != 3:
            raise ValueError("RGB NumPy images must have shape [H, W, 3]")
        if value.dtype == np.uint8:
            image = Image.fromarray(value)
        elif np.issubdtype(value.dtype, np.floating):
            if not np.isfinite(value).all() or value.min() < 0 or value.max() > 1:
                raise ValueError("Floating RGB images must be finite and lie in [0, 1]")
            image = Image.fromarray(np.rint(value * 255).astype(np.uint8))
        else:
            raise ValueError("RGB NumPy images must be uint8 or floating point in [0, 1]")
    else:
        raise TypeError("Each image must be a path, PIL image, or RGB NumPy array")
    if image.mode == "RGBA":
        image = Image.alpha_composite(Image.new("RGBA", image.size, "white"), image)
    if min(image.size) <= 0:
        raise ValueError("Images must have positive width and height")
    return image.convert("RGB")


def _select_images(images: ImageInput) -> tuple[list[Any], list[str], list[str] | None]:
    if isinstance(images, (str, Path)):
        values = list_image_paths(images)
    elif isinstance(images, Sequence) and not isinstance(images, (bytes, np.ndarray)):
        values = list(images)
    else:
        raise TypeError("images must be a directory or a sequence of paths, PIL images, or RGB arrays")
    if not values:
        raise ValueError("At least one image is required")
    are_paths = [isinstance(value, (str, Path)) for value in values]
    if any(are_paths) and not all(are_paths):
        raise ValueError("Use either a sequence of image paths or a sequence of in-memory images")
    if all(are_paths):
        paths = [str(Path(value).expanduser().resolve()) for value in values]
        return values, [Path(path).name for path in paths], paths
    return values, [f"frame_{index:06d}" for index in range(len(values))], None


def _transform_spec(width: int, height: int, target_size: int, mode: str) -> dict[str, Any]:
    if mode == "pad" and height > width:
        new_height = target_size
        new_width = round(width * (new_height / height) / 14) * 14
    else:
        new_width = target_size
        new_height = round(height * (new_width / width) / 14) * 14
    if min(new_width, new_height) < 14:
        raise ValueError("Image aspect ratio is too extreme for the requested target_size")
    crop_top = (new_height - target_size) // 2 if mode == "crop" and new_height > target_size else 0
    output_height = min(new_height, target_size) if mode == "crop" else new_height
    output_width = new_width
    pad_left = pad_top = pad_right = pad_bottom = 0
    if mode == "pad":
        pad_left = (target_size - output_width) // 2
        pad_right = target_size - output_width - pad_left
        pad_top = (target_size - output_height) // 2
        pad_bottom = target_size - output_height - pad_top
        output_width = output_height = target_size
    return {
        "original_size": [height, width],
        "resized_size": [new_height, new_width],
        "crop_top": crop_top,
        "crop_height": min(new_height, target_size) if mode == "crop" else new_height,
        "padding": [pad_left, pad_top, pad_right, pad_bottom],
        "processed_size": [output_height, output_width],
    }


def transform_mask(mask: np.ndarray, transform: dict[str, Any], fill_value: int | float = 0) -> np.ndarray:
    """Apply a frame's nearest-neighbor resize/crop/pad to a depth map or mask.

    ``transform`` is one entry from ``result['metadata']['transforms']``. The
    input must be a two-dimensional map at that frame's original resolution.
    Padding is invalid (zero) by default. The input is never modified.
    """
    array = np.asarray(mask)
    if array.ndim != 2 or list(array.shape) != transform["original_size"]:
        raise ValueError("Mask/depth shape must match the frame's original_size")
    if array.dtype.kind not in "biuf":
        raise ValueError("Mask/depth must be a numeric or boolean array")
    height, width = transform["resized_size"]
    # Pillow's float map avoids quantizing metric depth and uses exactly the
    # same pixel-center convention as the RGB image resize.
    resized = np.asarray(Image.fromarray(array.astype(np.float32)).resize(
        (width, height), Image.Resampling.NEAREST
    ))
    top = transform["crop_top"]
    resized = resized[top:top + transform["crop_height"]]
    left, top_pad, right, bottom = transform["padding"]
    resized = np.pad(resized, ((top_pad, bottom), (left, right)), constant_values=fill_value)
    return resized.astype(array.dtype, copy=True)


def _prepare_depths(depths: Any, original_sizes: list[list[int]]) -> list[np.ndarray]:
    count = len(original_sizes)
    if isinstance(depths, (torch.Tensor, np.ndarray)):
        depths = _as_numpy(depths, "depth")
        if depths.ndim != 3 or len(depths) != count:
            raise ValueError("depth must have shape [S, H, W] or be a list of per-frame arrays")
        frames = list(depths)
    elif isinstance(depths, Sequence):
        frames = list(depths)
        if len(frames) != count:
            raise ValueError("depth must contain one array for each image")
    else:
        raise ValueError("depth must have shape [S, H, W] or be a list of per-frame arrays")
    output = []
    for index, (frame, size) in enumerate(zip(frames, original_sizes)):
        if frame is None:
            raise ValueError(
                f"depth[{index}] cannot be None; use a zero array of shape {tuple(size)} "
                "for a frame without depth observations"
            )
        array = _as_numpy(frame, f"depth[{index}]")
        if array.ndim != 2 or list(array.shape) != size:
            raise ValueError(f"depth[{index}] must match original image size {size}")
        array = array.astype(np.float32, copy=True)
        array[~np.isfinite(array) | (array <= 0)] = 0
        output.append(array)
    return output


def _prepare_intrinsics(intrinsics: Any, count: int) -> np.ndarray:
    matrices = _as_numpy(intrinsics, "intrinsic")
    if matrices.shape == (3, 3):
        matrices = np.broadcast_to(matrices, (count, 3, 3))
    if matrices.shape != (count, 3, 3):
        raise ValueError("intrinsic must have shape [3, 3] or [S, 3, 3]")
    matrices = matrices.astype(np.float32, copy=True)
    if not np.isfinite(matrices).all() or np.any(matrices[:, (0, 1), (0, 1)] <= 0):
        raise ValueError("Every intrinsic matrix must be finite with positive focal lengths")
    if not np.allclose(matrices[:, 2], [0, 0, 1], atol=1e-6) or not np.allclose(matrices[:, (0, 1), (1, 0)], 0, atol=1e-6):
        raise ValueError("intrinsic must use the pinhole form [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]")
    return matrices


def _prepare_extrinsics(extrinsics: Any, count: int) -> np.ndarray:
    bf16_input = isinstance(extrinsics, torch.Tensor) and extrinsics.dtype == torch.bfloat16
    matrices = _as_numpy(extrinsics, "extrinsic")
    if matrices.shape not in ((count, 3, 4), (count, 4, 4)):
        raise ValueError("extrinsic must have shape [S, 3, 4] or [S, 4, 4] (world-to-camera)")
    matrices = matrices.astype(np.float32, copy=True)
    if not np.isfinite(matrices).all():
        raise ValueError("Every extrinsic matrix must be finite; missing camera poses are not supported")
    if matrices.shape[-2] == 4 and not np.allclose(matrices[:, 3], [0, 0, 0, 1], atol=1e-6):
        raise ValueError("Homogeneous extrinsic matrices must end with [0, 0, 0, 1]")
    rotations = matrices[:, :3, :3]
    rotation_atol = determinant_atol = 1e-3
    if bf16_input:
        # BF16 rounding perturbs a unit rotation row by at most u=eps/2 in L2.
        # Bound the resulting Gram/determinant errors without projecting poses.
        roundoff = torch.finfo(torch.bfloat16).eps / 2
        rotation_atol = 2 * roundoff + roundoff ** 2
        determinant_atol = (1 + roundoff) ** 3 - 1
    if (not np.allclose(rotations @ rotations.transpose(0, 2, 1), np.eye(3), atol=rotation_atol)
            or not np.allclose(np.linalg.det(rotations), 1, atol=determinant_atol)):
        raise ValueError("Each extrinsic rotation must be a valid orthonormal rotation with determinant +1")
    return matrices[:, :3].copy()


def prepare_inputs(
    images: ImageInput,
    depths: Any = None,
    intrinsics: Any = None,
    extrinsics: Any = None,
    target_size: int = 518,
    preprocess_mode: str = "crop",
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Prepare one sequence without changing any caller-owned images or geometry.

    Geometry is specified at the original RGB resolution. Depth is camera-axis
    Z in meters (zero/negative/nonfinite values are missing). Intrinsics are
    pinhole matrices. Extrinsics map world coordinates into camera coordinates.
    Every provided depth frame is an array; zeros represent absent observations.
    The returned batch has a leading batch size of one; metadata records each
    frame's affine pixel transform and original/processed image sizes.
    """
    if isinstance(target_size, bool) or not isinstance(target_size, int) or target_size < 14 or target_size % 14:
        raise ValueError("target_size must be a positive integer multiple of 14")
    if preprocess_mode not in ("crop", "pad"):
        raise ValueError("preprocess_mode must be 'crop' or 'pad'")
    values, names, paths = _select_images(images)
    loaded = [_rgb_image(value) for value in values]
    transforms = [_transform_spec(*image.size, target_size, preprocess_mode) for image in loaded]
    max_height = max(spec["processed_size"][0] for spec in transforms)
    max_width = max(spec["processed_size"][1] for spec in transforms)
    for spec in transforms:
        height, width = spec["processed_size"]
        extra_height, extra_width = max_height - height, max_width - width
        left, top, right, bottom = spec["padding"]
        spec["padding"] = [left + extra_width // 2, top + extra_height // 2,
                           right + extra_width - extra_width // 2, bottom + extra_height - extra_height // 2]
        spec["processed_size"] = [max_height, max_width]
        original_height, original_width = spec["original_size"]
        resized_height, resized_width = spec["resized_size"]
        sx, sy = resized_width / original_width, resized_height / original_height
        spec["pixel_transform"] = [[sx, 0.0, float(spec["padding"][0])],
                                   [0.0, sy, float(spec["padding"][1] - spec["crop_top"])],
                                   [0.0, 0.0, 1.0]]
    tensors = []
    for image, spec in zip(loaded, transforms):
        height, width = spec["resized_size"]
        resized = image.resize((width, height), Image.Resampling.BICUBIC)
        tensor = torch.from_numpy(np.array(resized, copy=True)).permute(2, 0, 1).float().div(255)
        top = spec["crop_top"]
        tensor = tensor[:, top:top + spec["crop_height"]]
        left, top_pad, right, bottom = spec["padding"]
        tensor = F.pad(tensor, (left, right, top_pad, bottom), value=1.0)
        tensors.append(tensor)
    batch = {"images": torch.stack(tensors).unsqueeze(0)}
    frame_count = len(loaded)
    # RGB copies are no longer needed while the geometric arrays are prepared.
    del loaded, tensors, resized, image, tensor
    if depths is not None:
        frames = _prepare_depths(depths, [spec["original_size"] for spec in transforms])
        depth_array = np.stack([transform_mask(frame, spec) for frame, spec in zip(frames, transforms)])
        batch["depth"] = torch.from_numpy(depth_array).unsqueeze(0)
    if intrinsics is not None:
        matrices = _prepare_intrinsics(intrinsics, frame_count)
        affine = np.asarray([spec["pixel_transform"] for spec in transforms], dtype=np.float32)
        batch["intrinsic"] = torch.from_numpy(affine @ matrices).unsqueeze(0)
    if extrinsics is not None:
        batch["extrinsic"] = torch.from_numpy(_prepare_extrinsics(extrinsics, frame_count)).unsqueeze(0)
    metadata = {"frame_names": names, "image_paths": paths, "transforms": transforms,
                "processed_size": [max_height, max_width], "preprocess_mode": preprocess_mode,
                "target_size": target_size}
    return batch, metadata
