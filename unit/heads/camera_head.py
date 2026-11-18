# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE.txt file in the root directory of this source tree.
# Modified for UniT.

import torch
import torch.nn as nn

from unit.layers.block import Block
from unit.heads.head_act import activate_translation
from unit.layers.kvcache import get_group_causal_mask


class CameraHead(nn.Module):
    """Predict camera extrinsics in one pass over shared camera-token features.

    The released head returns translation (3) and a flattened rotation matrix
    (9). Rotation is projected to SO(3) with SVD. Unlike the original VGGT head,
    this head does not iteratively refine a pose estimate. Relative-coordinate
    normalization is performed by the inference layer after prediction.
    """

    def __init__(
        self,
        dim_in: int = 2048,
        trunk_depth: int = 4,
        num_heads: int = 16,
        mlp_ratio: int = 4,
        init_values: float = 0.01,
    ):
        super().__init__()

        # Build the trunk using a sequence of transformer blocks.
        self.trunk = nn.Sequential(
            *[
                Block(
                    dim=dim_in,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    init_values=init_values,
                )
                for _ in range(trunk_depth)
            ]
        )

        self.token_norm = nn.LayerNorm(dim_in)

        self.trans_branch = nn.Linear(dim_in, 4)  # Raw translation vector and log scale.
        self.rot_branch = nn.Linear(dim_in, 9)

    def setup_caches(
        self,
        batch_size: int,
        tokens_per_frame: int,
        dtype: torch.dtype,
        device: torch.device,
        queue_capacity: int,
    ) -> None:
        """Allocate the camera trunk's per-frame KV queues.

        Args:
            tokens_per_frame (int): One camera token per frame in UniT.
            dtype (torch.dtype): dtype for the caches.
            device (torch.device): device for the caches.
            queue_capacity (int): Maximum retained frames.
        """
        for i in range(len(self.trunk)):
            self.trunk[i].setup_caches(
                batch_size, tokens_per_frame, dtype, device, queue_capacity
            )

    def reset_caches(self):
        """Reset the key value caches."""
        for i in range(len(self.trunk)):
            self.trunk[i].reset_caches()

    def forward(
        self,
        aggregated_tokens_list: list[torch.Tensor | None],
        group_size: int | torch.Tensor,
    ) -> torch.Tensor:
        """Decode one pose encoding per frame from the final backbone features.

        Args:
            aggregated_tokens_list (list): List of token tensors from the network;
                the last tensor is used for prediction.
            group_size: Shared integer or integer Tensor [B] during training. The
                inference scheduler supplies evaluation groups and cached context.

        Returns:
            Tensor [B, S, 12] for the released head: translation followed by the
            flattened, orthogonalized rotation matrix.
        """
        # Use tokens from the last block for camera prediction.
        tokens = aggregated_tokens_list[-1]

        # Extract the camera tokens
        pose_tokens = self.token_norm(tokens[:, :, 0])
        B, S, C = pose_tokens.shape

        if self.training:
            attn_mask = get_group_causal_mask(
                1, S, group_size, pose_tokens.device, batch_size=B,
            )
            for idx in range(len(self.trunk)):
                pose_tokens = self.trunk[idx](pose_tokens, None, attn_mask)
        else:
            pose_tokens = self.trunk(pose_tokens)

        translation = activate_translation(self.trans_branch(pose_tokens))
        rotation = self.rot_branch(pose_tokens)
        rotation = self.svd_orthogonalize(rotation.view(B * S, -1)).view(B, S, -1)

        return torch.cat([translation, rotation], dim=-1)

    def svd_orthogonalize(self, m):
        """Convert 9D representation to SO(3) using SVD orthogonalization.

        Args:
          m: [BATCH, 9] or [BATCH, 3, 3] unconstrained rotation parameters.

        Returns:
          [BATCH, 3, 3] SO(3) rotation matrices.
        """
        if m.dim() < 3:
            m = m.reshape((-1, 3, 3))
        m_transpose = torch.transpose(
            torch.nn.functional.normalize(m, p=2, dim=-1), dim0=-1, dim1=-2
        )
        u, _, v = torch.svd(m_transpose)
        det = torch.det(torch.matmul(v, u.transpose(-2, -1)))
        # Check orientation reflection.
        r = torch.matmul(
            torch.cat([v[:, :, :-1], v[:, :, -1:] * det.view(-1, 1, 1)], dim=2),
            u.transpose(-2, -1),
        )
        return r
