# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in licenses/Apache-2.0.txt.
# Modified for UniT.

# References:
#   https://github.com/facebookresearch/dino/blob/master/vision_transformer.py
#   https://github.com/rwightman/pytorch-image-models/tree/master/timm/models/vision_transformer.py

import logging

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from unit.layers.kvcache import QueueKVCache

logger = logging.getLogger(__name__)


class Attention(nn.Module):
    """Self-attention with optional spatial RoPE and a queue of frame keys/values."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        norm_layer: type[nn.Module] = nn.LayerNorm,
        qk_norm: bool = False,
        rope=None,
    ) -> None:
        super().__init__()
        if dim % num_heads:
            raise ValueError("dim must be divisible by num_heads")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)
        self.rope = rope
        self.kv_cache = None

    def setup_caches(
        self,
        batch_size: int,
        tokens_per_frame: int,
        dtype: torch.dtype,
        device: torch.device,
        queue_capacity: int,
    ) -> None:
        """Allocate a fixed-capacity frame queue, preserving an existing cache."""
        if self.kv_cache is not None:
            logger.warning("KV cache already exists; skipping repeated setup.")
            return
        self.kv_cache = QueueKVCache(
            batch_size=batch_size,
            tokens_per_frame=tokens_per_frame,
            queue_capacity=queue_capacity,
            num_kv_heads=self.num_heads,
            head_dim=self.head_dim,
            dtype=dtype,
            device=device,
        )

    def reset_caches(self) -> None:
        """Release an initialized frame queue."""
        if self.kv_cache is None:
            raise RuntimeError("KV cache is not initialized; call setup_caches() first")
        self.kv_cache = None

    def _attend(
        self, q: Tensor, k: Tensor, v: Tensor, pos: Tensor | None,
        attn_mask: Tensor | None,
    ) -> Tensor:
        """Apply the shared cache, normalization, RoPE, SDPA, and output steps.

        Projections have shape [B, heads, tokens, head_dim]. Cached keys retain
        their repeated per-frame spatial positions; queries cover this group.
        """
        B, _, N, _ = q.shape
        pos_k = pos
        if self.kv_cache is not None:
            k, v = self.kv_cache.update(k, v)
            if pos_k is not None:
                L = k.shape[2]
                repeats = (L + N - 1) // N
                pos_k = pos.repeat(1, repeats, 1)[:, :L]

        q, k = self.q_norm(q), self.k_norm(k)
        if self.rope is not None:
            q = self.rope(q, pos)
            k = self.rope(k, pos_k)

        x = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask,
            dropout_p=self.attn_drop.p if self.training else 0.0,
        )
        x = x.transpose(1, 2).reshape(B, N, self.num_heads * self.head_dim)
        x = self.proj(x)
        return self.proj_drop(x)

    def forward(self, x: Tensor, pos=None, attn_mask=None) -> Tensor:
        B, N, _ = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv.unbind(0)
        return self._attend(q, k, v, pos, attn_mask)


class ModalAttention(Attention):
    """Attend from image tokens to spatially aligned image/modal token pairs.

    Image content produces the queries. Channel-wise concatenation with the
    corresponding point/pose tokens produces keys and values.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        norm_layer: type[nn.Module] = nn.LayerNorm,
        qk_norm: bool = False,
        rope=None,
    ) -> None:
        super().__init__(
            dim=dim, num_heads=num_heads, qkv_bias=qkv_bias,
            proj_bias=proj_bias, attn_drop=attn_drop, proj_drop=proj_drop,
            norm_layer=norm_layer, qk_norm=qk_norm, rope=rope,
        )
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(2 * dim, 2 * dim, bias=qkv_bias)
        # Keep construction order identical to the trained model's initialization.
        self.qkv = None

    def forward(
        self, x: Tensor, modal_tokens: Tensor, pos=None, attn_mask=None,
    ) -> Tensor:
        B, N, _ = x.shape
        q = self.q(x)
        q = q.reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        kv = (
            self.kv(torch.cat([x, modal_tokens], dim=-1))
            .reshape(B, N, 2, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        k, v = kv.unbind(0)
        return self._attend(q, k, v, pos, attn_mask)
