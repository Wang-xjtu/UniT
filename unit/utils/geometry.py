# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE.txt file in the root directory of this source tree.
# Modified for UniT.


"""Coordinate transforms shared by models, inference, and external adapters."""

from collections.abc import Mapping

import torch
import numpy as np


def to_hmat(extri: torch.Tensor) -> torch.Tensor:
    """Extend [B, S, 3, 4] world-to-camera transforms to homogeneous matrices."""
    B, S, _, _ = extri.shape
    R = extri[..., :3, :3]
    t = extri[..., :3, 3]
    I = (
        torch.eye(4, device=extri.device, dtype=extri.dtype)
        .view(1, 1, 4, 4)
        .repeat(B, S, 1, 1)
    )
    I[..., :3, :3] = R
    I[..., :3, 3] = t
    return I



def camera_point_extrinsic_to_world_point(
    camera_points,
    extrinsics,
):
    """Transform camera-local XYZ [B, S, H, W, 3] into common coordinates.

    Extrinsics [B, S, 3, 4] map common coordinates to camera coordinates.
    This is a rigid transform; camera intrinsics are not involved.
    """
    B, S, _, _ = extrinsics.shape

    extrinsics_homog = torch.cat(
        [
            extrinsics,
            torch.zeros((B, S, 1, 4), device=extrinsics.device),
        ],
        dim=-2,
    )

    extrinsics_homog[:, :, -1, -1] = 1.0

    # extrinsic_inv, the inverse of extrinsic matrix of each frame
    # which can be also viewed as the cam_to_world extrinsic matrix
    extrinsics_homog = extrinsics_homog.view(B * S, 4, 4)
    extrinsic_inv = closed_form_inverse_se3(extrinsics_homog)
    extrinsic_inv = extrinsic_inv.view(B, S, 4, 4)
    R_inv = extrinsic_inv[:, :, :3, :3]  # shape: (B, S, 3, 3)
    t_inv = extrinsic_inv[:, :, :3, 3]  # shape: (B, S, 3)

    # Apply the rotation and translation to the camera coordinates
    world_points = (
        camera_points @ R_inv.transpose(-1, -2).unsqueeze(2)
    ) + t_inv.unsqueeze(2).unsqueeze(3)

    return world_points



def express_predictions_in_reference_frame(
    predictions: Mapping[str, torch.Tensor],
    reference_camera_to_world: torch.Tensor | None = None,
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    """Express predictions in the first camera's frame, or a supplied reference.

    ``predictions`` contains camera-local ``cam_points`` [B, S, H, W, 3] and
    world-to-camera ``extrinsics`` [B, S, 3, 4]. If no reference is supplied,
    the first predicted camera defines it. Reuse the returned [B, 4, 4]
    camera-to-world reference to normalize later groups in the same frame.

    The returned dictionary contains normalized ``extrinsics`` and
    ``world_points``; other values are shared with the input. No input tensors
    are modified or detached. The caller controls precision and gradient mode.
    This coordinate convention does not require retaining the first frame in
    the model's attention cache.
    """
    extrinsics = to_hmat(predictions["extrinsics"])
    if reference_camera_to_world is None:
        reference_camera_to_world = closed_form_inverse_se3(extrinsics[:, 0])
    relative_extrinsics = torch.matmul(
        extrinsics, reference_camera_to_world.unsqueeze(1)
    )
    normalized = dict(predictions)
    normalized["extrinsics"] = relative_extrinsics[:, :, :3]
    normalized["world_points"] = camera_point_extrinsic_to_world_point(
        normalized["cam_points"], normalized["extrinsics"]
    )
    return normalized, reference_camera_to_world


def intrinsics_to_camera_rays(intrinsic: torch.Tensor, data_size: tuple[int, int]) -> torch.Tensor:
    """Return pinhole ray coordinates (x/z, y/z, 1) in [N, H, W, 3]."""
    if intrinsic.ndim == 2:
        intrinsic = intrinsic.unsqueeze(0)  # (1, 3, 3)

    N = intrinsic.shape[0]
    H, W = data_size
    device, dtype = intrinsic.device, intrinsic.dtype

    fu = intrinsic[:, 0, 0].view(N, 1, 1)
    fv = intrinsic[:, 1, 1].view(N, 1, 1)
    cu = intrinsic[:, 0, 2].view(N, 1, 1)
    cv = intrinsic[:, 1, 2].view(N, 1, 1)

    u = torch.arange(W, device=device, dtype=dtype).view(1, 1, W).expand(N, H, W)
    v = torch.arange(H, device=device, dtype=dtype).view(1, H, 1).expand(N, H, W)

    x = (u - cu) / fu  # (N, H, W)
    y = (v - cv) / fv  # (N, H, W)
    z = torch.ones((N, H, W), device=device, dtype=dtype)

    rays_cam = torch.stack((x, y, z), dim=-1)  # (N, H, W, 3)
    return rays_cam



def closed_form_inverse_se3(se3, R=None, T=None):
    """
    Compute the inverse of each 4x4 (or 3x4) SE3 matrix in a batch.

    If `R` and `T` are provided, they must correspond to the rotation and translation
    components of `se3`. Otherwise, they will be extracted from `se3`.

    Args:
        se3: Nx4x4 or Nx3x4 array or tensor of SE3 matrices.
        R (optional): Nx3x3 array or tensor of rotation matrices.
        T (optional): Nx3x1 array or tensor of translation vectors.

    Returns:
        Inverted SE3 matrices with the same type and device as `se3`.

    Shapes:
        se3: (N, 4, 4)
        R: (N, 3, 3)
        T: (N, 3, 1)
    """
    # Check if se3 is a numpy array or a torch tensor
    is_numpy = isinstance(se3, np.ndarray)

    # Validate shapes
    if se3.shape[-2:] != (4, 4) and se3.shape[-2:] != (3, 4):
        raise ValueError(f"se3 must be of shape (N,4,4), got {se3.shape}.")

    # Extract R and T if not provided
    if R is None:
        R = se3[:, :3, :3]  # (N,3,3)
    if T is None:
        T = se3[:, :3, 3:]  # (N,3,1)

    # Transpose R
    if is_numpy:
        # Compute the transpose of the rotation for NumPy
        R_transposed = np.transpose(R, (0, 2, 1))
        # -R^T t for NumPy
        top_right = -np.matmul(R_transposed, T)
        inverted_matrix = np.tile(np.eye(4), (len(R), 1, 1))
    else:
        R_transposed = R.transpose(1, 2)  # (N,3,3)
        top_right = -torch.bmm(R_transposed, T)  # (N,3,1)
        inverted_matrix = torch.eye(4, 4)[None].repeat(len(R), 1, 1)
        inverted_matrix = inverted_matrix.to(R.dtype).to(R.device)

    inverted_matrix[:, :3, :3] = R_transposed
    inverted_matrix[:, :3, 3:] = top_right

    return inverted_matrix

