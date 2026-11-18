# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE.txt file in the root directory of this source tree.
# Modified for UniT.

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from unit.layers import PatchEmbed, Mlp, SPNorm
from unit.layers.block import Block, ModalAttentionBlock
from unit.layers.rope import RotaryPositionEmbedding2D, PositionGetter
from unit.layers.vision_transformer import vit_large
from unit.layers.kvcache import get_group_causal_mask

_RESNET_MEAN = [0.485, 0.456, 0.406]
_RESNET_STD = [0.229, 0.224, 0.225]


class Aggregator(nn.Module):
    """Alternate per-frame and group-causal attention with four modal stages.

    Point/pose tokenizers run once, then stages 0, 5, 12, and 18 reuse the same
    modal tokens. Camera/register tokens are shared by all frames. Training uses
    a group-causal mask; evaluation processes scheduled groups with KV history.
    """

    def __init__(
        self,
        img_size=518,
        patch_size=14,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4.0,
        num_register_tokens=4,
        block_fn=Block,
        qkv_bias=True,
        proj_bias=True,
        ffn_bias=True,
        patch_embed="dinov2_vitl14_reg",
        qk_norm=True,
        rope_freq=100,
        init_values=0.01,
    ):
        super().__init__()
        if rope_freq <= 0:
            raise ValueError("rope_freq must be positive for the UniT backbone")
        self.patch_embed = self._build_patch_embed(
            patch_embed,
            img_size,
            patch_size,
            num_register_tokens,
            embed_dim=embed_dim,
            in_chans=3,
        )

        self.point_embed = nn.Sequential(
            self._build_patch_embed(
                "conv",
                img_size,
                patch_size,
                num_register_tokens,
                embed_dim=embed_dim,
                in_chans=5,
            ),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
            SPNorm(embed_dim),
        )
        self.pose_embed = nn.Sequential(
            Mlp(13, embed_dim, embed_dim),  # translation(3), rotation(9), validity(1)
            SPNorm(embed_dim),
        )

        self.rope = RotaryPositionEmbedding2D(frequency=rope_freq)
        self.position_getter = PositionGetter()
        self.modal_stage_indices = (0, 5, 12, 18)

        # Keep this registered name to preserve existing checkpoint keys.
        self.anchor_blocks = nn.ModuleList(
            [
                ModalAttentionBlock(
                    dim=embed_dim,
                    num_heads=num_heads,
                    qkv_bias=qkv_bias,
                    proj_bias=proj_bias,
                    qk_norm=qk_norm,
                    rope=self.rope,
                )
                for _ in range(len(self.modal_stage_indices))
            ]
        )

        self.frame_blocks = nn.ModuleList(
            [
                block_fn(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    proj_bias=proj_bias,
                    ffn_bias=ffn_bias,
                    init_values=init_values,
                    qk_norm=qk_norm,
                    rope=self.rope,
                )
                for _ in range(depth)
            ]
        )

        self.global_blocks = nn.ModuleList(
            [
                block_fn(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    proj_bias=proj_bias,
                    ffn_bias=ffn_bias,
                    init_values=init_values,
                    qk_norm=qk_norm,
                    rope=self.rope,
                )
                for _ in range(depth)
            ]
        )

        self.depth = depth
        self.patch_size = patch_size

        # All frames share the same camera token and register tokens. There is
        # no separate token parameter for the first frame.
        self.camera_token = nn.Parameter(torch.randn(1, 1, 1, embed_dim))
        self.register_token = nn.Parameter(
            torch.randn(1, 1, num_register_tokens, embed_dim)
        )

        # The patch tokens start after the camera and register tokens
        self.patch_start_idx = 1 + num_register_tokens

        # Initialize parameters with small values
        nn.init.normal_(self.camera_token, std=1e-6)
        nn.init.normal_(self.register_token, std=1e-6)

        # Register normalization constants as buffers
        for name, value in (
            ("_resnet_mean", _RESNET_MEAN),
            ("_resnet_std", _RESNET_STD),
        ):
            self.register_buffer(
                name, torch.FloatTensor(value).view(1, 1, 3, 1, 1), persistent=False
            )

        self.use_reentrant = False

    def _build_patch_embed(
        self,
        patch_embed,
        img_size,
        patch_size,
        num_register_tokens,
        interpolate_antialias=True,
        interpolate_offset=0.0,
        init_values=1.0,
        embed_dim=1024,
        in_chans=3,
    ):
        """
        Build the patch embed layer. If 'conv', we use a
        simple PatchEmbed conv layer. Otherwise, we use a vision transformer.
        """

        if patch_embed == "conv":
            embed_layer = PatchEmbed(
                img_size=img_size,
                patch_size=patch_size,
                in_chans=in_chans,
                embed_dim=embed_dim,
            )
        elif patch_embed == "dinov2_vitl14_reg":
            embed_layer = vit_large(
                in_chans=in_chans,
                img_size=img_size,
                patch_size=patch_size,
                num_register_tokens=num_register_tokens,
                interpolate_antialias=interpolate_antialias,
                interpolate_offset=interpolate_offset,
                init_values=init_values,
            )

            # Disable gradient updates for mask token
            if hasattr(embed_layer, "mask_token"):
                embed_layer.mask_token.requires_grad_(False)
        else:
            raise ValueError("patch_embed must be 'conv' or 'dinov2_vitl14_reg'")

        return embed_layer

    def setup_caches(
        self,
        batch_size: int,
        tokens_per_frame: int,
        dtype: torch.dtype,
        device: torch.device,
        queue_capacity: int,
    ) -> None:
        """Allocate per-frame KV queues for global attention blocks.

        Args:
            tokens_per_frame (int): Camera, register, and image tokens per frame.
            dtype (torch.dtype): dtype for the caches.
            device (torch.device): device for the caches.
            queue_capacity (int): Maximum retained frames.
        """
        for i in range(len(self.global_blocks)):
            self.global_blocks[i].setup_caches(
                batch_size, tokens_per_frame, dtype, device, queue_capacity
            )

    def reset_caches(self):
        """Reset the key value caches."""
        for i in range(len(self.global_blocks)):
            self.global_blocks[i].reset_caches()

    def forward(
        self,
        images: torch.Tensor,
        ray_depth_features: torch.Tensor,
        pose_features: torch.Tensor,
        group_size: int | torch.Tensor,
        keep_layer_idx=None,
    ) -> tuple[list[torch.Tensor | None], int]:
        """Return concatenated frame/global features at the requested layers.

        Inputs are RGB [B,S,3,H,W], ray-depth with two validity channels
        [B,S,5,H,W], and translation/rotation with validity [B,S,13]. Output
        features are [B,S,P,2C]; unrequested layers contain None. P includes the
        camera token, register tokens, and image patches. Training accepts a shared
        integer group size or a separate integer size for each batch item.
        """
        B, S, channels, H, W = images.shape
        point_channels = ray_depth_features.shape[2]
        if channels != 3:
            raise ValueError(f"Expected 3 RGB channels, got {channels}")
        if images.shape[-2:] != ray_depth_features.shape[-2:]:
            raise ValueError("RGB and ray-depth features must have matching spatial sizes")

        # Tokenize once; every modal stage reuses the point and pose tokens.
        images = (images - self._resnet_mean) / self._resnet_std
        images = images.view(B * S, channels, H, W)
        patch_tokens = self.patch_embed(images)
        if isinstance(patch_tokens, dict):
            patch_tokens = patch_tokens["x_norm_patchtokens"]
        point_tokens = self.point_embed(ray_depth_features.view(B * S, point_channels, H, W))
        pose_tokens = self.pose_embed(pose_features.view(B * S, 1, -1))
        camera_tokens = expand_shared_tokens(self.camera_token, B, S)
        register_tokens = expand_shared_tokens(self.register_token, B, S)
        tokens = torch.cat([camera_tokens, register_tokens, patch_tokens], dim=1)
        modal_tokens = torch.cat([pose_tokens, point_tokens], dim=1)
        _, P, C = tokens.shape

        frame_pos = self.position_getter(
            B * S, H // self.patch_size, W // self.patch_size, device=images.device
        )
        # Zero positions leave camera/register tokens unrotated by spatial RoPE.
        frame_pos = frame_pos + 1
        special_pos = torch.zeros(B * S, self.patch_start_idx, 2).to(images.device).to(frame_pos.dtype)
        frame_pos = torch.cat([special_pos, frame_pos], dim=1)
        global_pos = frame_pos.view(B, S, P, 2).view(B, S * P, 2)
        attn_mask = (
            get_group_causal_mask(P, S, group_size, tokens.device, batch_size=B)
            if self.training else None
        )

        keep = None if keep_layer_idx is None else set(keep_layer_idx)
        outputs = [None] * self.depth
        modal_index = 0
        for layer_index in range(self.depth):
            tokens = tokens.view(B, S, P, C).view(B * S, P, C)
            if layer_index in self.modal_stage_indices:
                tokens = self._apply_modal_attention(
                    tokens, modal_tokens, frame_pos, modal_index,
                )
                modal_index += 1

            frame_tokens = self._run_attention_block(
                self.frame_blocks[layer_index], tokens, frame_pos,
            )
            global_tokens = self._run_attention_block(
                self.global_blocks[layer_index],
                frame_tokens.view(B, S, P, C).view(B, S * P, C),
                global_pos, attn_mask,
            )
            if keep is None or layer_index in keep:
                outputs[layer_index] = torch.cat(
                    [frame_tokens.view(B, S, P, C), global_tokens.view(B, S, P, C)], dim=-1,
                )
            tokens = global_tokens

        return outputs, self.patch_start_idx

    def _run_attention_block(self, block, tokens, pos, attn_mask=None):
        """Use the same block computation, checkpointing only during training."""
        if self.training:
            return checkpoint(block, tokens, pos, attn_mask, use_reentrant=self.use_reentrant)
        return block(tokens, pos=pos, attn_mask=attn_mask)

    def _apply_modal_attention(self, tokens, modal_tokens, pos, modal_index):
        """Fuse camera/patch tokens while leaving register tokens untouched."""
        content_tokens = torch.cat([tokens[:, :1], tokens[:, self.patch_start_idx:]], dim=1)
        content_pos = torch.cat([pos[:, :1], pos[:, self.patch_start_idx:]], dim=1)
        block = self.anchor_blocks[modal_index]
        if self.training:
            content_tokens = checkpoint(
                block, content_tokens, modal_tokens, content_pos,
                use_reentrant=self.use_reentrant,
            )
        else:
            content_tokens = block(content_tokens, modal_tokens, content_pos)
        return torch.cat(
            [content_tokens[:, :1], tokens[:, 1:self.patch_start_idx], content_tokens[:, 1:]],
            dim=1,
        ).contiguous()


def expand_shared_tokens(token_tensor: torch.Tensor, batch_size: int, frames: int) -> torch.Tensor:
    """Broadcast one shared token set to [batch_size * frames, token_count, dim]."""
    expanded = token_tensor.expand(batch_size, frames, *token_tensor.shape[2:])
    return expanded.view(batch_size * frames, *expanded.shape[2:])
