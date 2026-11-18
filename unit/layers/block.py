# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in licenses/Apache-2.0.txt.
# Modified for UniT.

# References:
#   https://github.com/facebookresearch/dino/blob/master/vision_transformer.py
#   https://github.com/rwightman/pytorch-image-models/tree/master/timm/layers/patch_embed.py
from typing import Callable

import torch
from torch import nn, Tensor

from .attention import Attention, ModalAttention
from .drop_path import DropPath
from .layer_scale import LayerScale
from .mlp import Mlp


class Block(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        proj_bias: bool = True,
        ffn_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        init_values=None,
        drop_path: float = 0.0,
        act_layer: Callable[..., nn.Module] = nn.GELU,
        norm_layer: Callable[..., nn.Module] = nn.LayerNorm,
        attn_class: Callable[..., nn.Module] = Attention,
        qk_norm: bool = False,
        rope=None,
    ) -> None:
        super().__init__()

        self.norm1 = norm_layer(dim)

        self.attn = attn_class(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            proj_bias=proj_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
            qk_norm=qk_norm,
            rope=rope,
        )

        self.ls1 = (
            LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        )
        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
            bias=ffn_bias,
        )
        self.ls2 = (
            LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        )
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.sample_drop_ratio = drop_path

    def setup_caches(
        self,
        batch_size: int,
        tokens_per_frame: int,
        dtype: torch.dtype,
        device: torch.device,
        queue_capacity: int,
    ) -> None:
        """Setup key value caches for attention calculation. If called
        after kv_cache is already setup, this will be skipped.

        Args:
            tokens_per_frame (int): Number of tokens stored per frame.
            dtype (torch.dtype): dtype for the caches.
            device (torch.device): device for the caches.
            queue_capacity (int): Maximum number of retained frames.
        """
        self.attn.setup_caches(
            batch_size, tokens_per_frame, dtype, device, queue_capacity
        )

    def reset_caches(self):
        """Reset the key value caches."""
        self.attn.reset_caches()

    def forward(self, x: Tensor, pos=None, attn_mask=None) -> Tensor:
        def attn_residual_func(x: Tensor, pos=None, attn_mask=None) -> Tensor:
            return self.ls1(self.attn(self.norm1(x), pos=pos, attn_mask=attn_mask))

        def ffn_residual_func(x: Tensor, attn_mask=None) -> Tensor:
            return self.ls2(self.mlp(self.norm2(x)))

        if self.training and self.sample_drop_ratio > 0.1:
            # the overhead is compensated only for a drop path rate larger than 0.1
            x = drop_add_residual_stochastic_depth(
                x,
                pos=pos,
                attn_mask=attn_mask,
                residual_func=attn_residual_func,
                sample_drop_ratio=self.sample_drop_ratio,
            )
            x = drop_add_residual_stochastic_depth(
                x,
                residual_func=ffn_residual_func,
                sample_drop_ratio=self.sample_drop_ratio,
            )
        elif self.training and self.sample_drop_ratio > 0.0:
            x = x + self.drop_path1(attn_residual_func(x, pos=pos, attn_mask=attn_mask))
            x = x + self.drop_path1(ffn_residual_func(x))  # Preserve the trained residual-drop behavior.
        else:
            x = x + attn_residual_func(x, pos=pos, attn_mask=attn_mask)
            x = x + ffn_residual_func(x)
        return x


def drop_add_residual_stochastic_depth(
    x: Tensor,
    residual_func: Callable[[Tensor], Tensor],
    sample_drop_ratio: float = 0.0,
    pos=None,
    attn_mask=None,
) -> Tensor:
    # 1) extract subset using permutation
    b, n, d = x.shape
    sample_subset_size = max(int(b * (1 - sample_drop_ratio)), 1)
    brange = (torch.randperm(b, device=x.device))[:sample_subset_size]
    x_subset = x[brange]

    # 2) apply residual_func to get residual
    if pos is not None:
        # if necessary, apply rope to the subset
        pos = pos[brange]
        residual = residual_func(x_subset, pos=pos, attn_mask=attn_mask)
    else:
        residual = residual_func(x_subset, attn_mask=attn_mask)

    x_flat = x.flatten(1)
    residual = residual.flatten(1)

    residual_scale_factor = b / sample_subset_size

    # 3) add the residual
    x_plus_residual = torch.index_add(
        x_flat, 0, brange, residual.to(dtype=x.dtype), alpha=residual_scale_factor
    )
    return x_plus_residual.view_as(x)


class ModalAttentionBlock(nn.Module):
    """Inject modal tokens through attention and a residual projection.

    ``anchor_attn`` and ``anchor_proj`` are historical registered names retained
    for checkpoint compatibility. They implement modal fusion, independently of
    the anchor-free camera representation discussed in the paper.
    """

    def __init__(
        self,
        dim=1024,
        num_heads=16,
        qkv_bias=True,
        proj_bias=True,
        qk_norm=True,
        rope=None,
    ):
        super().__init__()
        self.anchor_attn = ModalAttention(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            proj_bias=proj_bias,
            qk_norm=qk_norm,
            rope=rope,
        )
        self.anchor_proj = nn.Linear(dim, dim)

        # Only projection weights are zero-initialized; the bias keeps its default.
        nn.init.zeros_(self.anchor_proj.weight)

    def forward(self, x, modal_tokens, pos):
        """Fuse image content with corresponding point/pose tokens."""
        modal_update = self.anchor_attn(x, modal_tokens, pos)
        modal_update = self.anchor_proj(modal_update)
        x = x + modal_update
        return x
