# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE.txt file in the root directory of this source tree.
# Modified for UniT.


import torch


def activate_translation(encoded: torch.Tensor) -> torch.Tensor:
    """Decode four raw values as a vector times a positive scale factor.

    The first three values are not unit-normalized, so the exponential factor
    is not itself the translation magnitude.
    """
    translation_vector = encoded[..., :3]
    positive_scale = torch.exp(encoded[..., 3:4])
    return translation_vector * positive_scale


def activate_point_map(out: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Decode [B,4,H,W] into local XYZ [B,H,W,3] and confidence [B,H,W].

    The first two channels parameterize x/z and y/z, the third parameterizes
    log depth, and the fourth gives confidence as 1 + exp(raw_confidence).
    """
    feature_map = out.permute(0, 2, 3, 1)
    xyz = feature_map[:, :, :, :-1]
    confidence = feature_map[:, :, :, -1]
    xy, depth = xyz.split([2, 1], dim=-1)
    depth = torch.exp(depth)
    points = torch.cat([xy * depth, depth], dim=-1)
    return points, 1 + confidence.exp()
