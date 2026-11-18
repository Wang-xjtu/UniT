# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE.txt file in the root directory of this source tree.
# Modified for UniT.

"""The model's 12D world-to-camera pose representation."""

import torch


def encode_extrinsics(extrinsics: torch.Tensor) -> torch.Tensor:
    """Encode [B, S, 3, 4] poses as translation(3) then rotation(9), in FP32."""
    rotation = extrinsics[:, :, :3, :3]
    translation = extrinsics[:, :, :3, 3]
    batch_size, frames, _, _ = rotation.shape
    flattened_rotation = rotation.reshape(batch_size, frames, -1)
    return torch.cat([translation, flattened_rotation], dim=-1).float()


def decode_extrinsics(pose_encoding: torch.Tensor) -> torch.Tensor:
    """Decode [B, S, 12] into [B, S, 3, 4] world-to-camera transforms.

    CameraHead already projects its rotation prediction to SO(3); decoding
    only restores the matrix layout. Axes are x right, y down, and z forward.
    """
    translation = pose_encoding[..., :3]
    flattened_rotation = pose_encoding[..., 3:]
    batch_size, frames, _ = flattened_rotation.shape
    rotation = flattened_rotation.reshape(batch_size, frames, 3, 3)
    return torch.cat([rotation, translation[..., None]], dim=-1)
