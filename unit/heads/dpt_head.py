# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE.txt file in the root directory of this source tree.
# Modified for UniT.


# Inspired by https://github.com/DepthAnything/Depth-Anything-V2


import torch
import torch.nn as nn
from .head_act import activate_point_map
from .utils import create_uv_grid, position_grid_to_embed


class DPTHead(nn.Module):
    """Decode local point maps and confidence from four backbone feature maps.

    Features at layers [4, 11, 17, 23] are the ends of the four stages whose
    Modal Attention starts at [0, 5, 12, 18]. Multi-scale DPT fusion preserves
    the trained module hierarchy and outputs XYZ in each camera's coordinates.
    """

    def __init__(
        self,
        dim_in: int,
        patch_size: int = 14,
        features: int = 256,
        out_channels: tuple[int, ...] = (256, 512, 1024, 1024),
        intermediate_layer_idx: tuple[int, ...] = (4, 11, 17, 23),
        pos_embed: bool = True,
        down_ratio: int = 1,
    ) -> None:
        super(DPTHead, self).__init__()
        self.patch_size = patch_size
        self.pos_embed = pos_embed
        self.down_ratio = down_ratio
        self.intermediate_layer_idx = intermediate_layer_idx

        self.norm = nn.LayerNorm(dim_in)

        # Projection layers for each output channel from tokens.
        self.projects = nn.ModuleList(
            [
                nn.Conv2d(
                    in_channels=dim_in,
                    out_channels=oc,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )
                for oc in out_channels
            ]
        )

        # Resize layers for upsampling feature maps.
        self.resize_layers = nn.ModuleList(
            [
                nn.ConvTranspose2d(
                    in_channels=out_channels[0],
                    out_channels=out_channels[0],
                    kernel_size=4,
                    stride=4,
                    padding=0,
                ),
                nn.ConvTranspose2d(
                    in_channels=out_channels[1],
                    out_channels=out_channels[1],
                    kernel_size=2,
                    stride=2,
                    padding=0,
                ),
                nn.Identity(),
                nn.Conv2d(
                    in_channels=out_channels[3],
                    out_channels=out_channels[3],
                    kernel_size=3,
                    stride=2,
                    padding=1,
                ),
            ]
        )

        self.scratch = _make_scratch(out_channels, features)

        # Attach additional modules to scratch.
        self.scratch.stem_transpose = None
        self.scratch.refinenet1 = _make_fusion_block(features)
        self.scratch.refinenet2 = _make_fusion_block(features)
        self.scratch.refinenet3 = _make_fusion_block(features)
        self.scratch.refinenet4 = _make_fusion_block(features, has_residual=False)

        head_features_1 = features
        head_features_2 = 32

        self.scratch.output_conv1 = nn.Conv2d(
            head_features_1,
            head_features_1 // 2,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        conv2_in_channels = head_features_1 // 2

        self.scratch.output_conv2 = nn.Sequential(
            nn.Conv2d(
                conv2_in_channels,
                head_features_2,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                head_features_2, 4, kernel_size=1, stride=1, padding=0
            ),
        )

    def forward(
        self,
        aggregated_tokens_list: list[torch.Tensor | None],
        images: torch.Tensor,
        patch_start_idx: int,
        frames_chunk_size: int | None = 8,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return local XYZ [B,S,H,W,3] and confidence [B,S,H,W].

        Inputs are selected backbone features [B,S,P,2C] and RGB [B,S,3,H,W].
        ``patch_start_idx`` skips camera/register tokens. Only the four DPT
        feature entries are read; other entries may be None. Frames are chunked
        to limit temporary memory, or processed together when the limit is None.
        """
        sequence_length = images.shape[1]
        if frames_chunk_size is None or frames_chunk_size >= sequence_length:
            return self._forward_impl(aggregated_tokens_list, images, patch_start_idx)
        if frames_chunk_size <= 0:
            raise ValueError("frames_chunk_size must be positive or None")

        point_chunks, confidence_chunks = [], []
        for start in range(0, sequence_length, frames_chunk_size):
            points, confidence = self._forward_impl(
                aggregated_tokens_list, images, patch_start_idx,
                start, min(start + frames_chunk_size, sequence_length),
            )
            point_chunks.append(points)
            confidence_chunks.append(confidence)
        return torch.cat(point_chunks, dim=1), torch.cat(confidence_chunks, dim=1)

    def _forward_impl(
        self,
        aggregated_tokens_list: list[torch.Tensor | None],
        images: torch.Tensor,
        patch_start_idx: int,
        frames_start_idx: int | None = None,
        frames_end_idx: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Decode all frames or a contiguous slice without changing feature order."""
        if frames_start_idx is not None and frames_end_idx is not None:
            images = images[:, frames_start_idx:frames_end_idx].contiguous()

        B, S, _, H, W = images.shape

        patch_h, patch_w = H // self.patch_size, W // self.patch_size

        out = []
        for dpt_idx, layer_idx in enumerate(self.intermediate_layer_idx):
            features = aggregated_tokens_list[layer_idx]
            if features is None:
                raise ValueError(f"DPT feature layer {layer_idx} is missing")
            x = features[:, :, patch_start_idx:]

            # Select frames if processing a chunk
            if frames_start_idx is not None and frames_end_idx is not None:
                x = x[:, frames_start_idx:frames_end_idx]

            x = x.reshape(B * S, -1, x.shape[-1])

            x = self.norm(x)

            x = x.permute(0, 2, 1).reshape((x.shape[0], x.shape[-1], patch_h, patch_w))

            x = self.projects[dpt_idx](x)
            if self.pos_embed:
                x = self._apply_pos_embed(x, W, H)
            x = self.resize_layers[dpt_idx](x)

            out.append(x)

        # Fuse features from multiple layers.
        out = self.scratch_forward(out)
        # Interpolate fused output to match target image resolution.
        out = custom_interpolate(
            out,
            (
                int(patch_h * self.patch_size / self.down_ratio),
                int(patch_w * self.patch_size / self.down_ratio),
            ),
            mode="bilinear",
            align_corners=True,
        )

        if self.pos_embed:
            out = self._apply_pos_embed(out, W, H)

        out = self.scratch.output_conv2(out)
        preds, conf = activate_point_map(out)

        preds = preds.view(B, S, *preds.shape[1:])
        conf = conf.view(B, S, *conf.shape[1:])
        return preds, conf

    def _apply_pos_embed(
        self, x: torch.Tensor, W: int, H: int, ratio: float = 0.1
    ) -> torch.Tensor:
        """
        Apply positional embedding to tensor x.
        """
        patch_w = x.shape[-1]
        patch_h = x.shape[-2]
        pos_embed = create_uv_grid(
            patch_w, patch_h, aspect_ratio=W / H, dtype=x.dtype, device=x.device
        )
        pos_embed = position_grid_to_embed(pos_embed, x.shape[1])
        pos_embed = pos_embed * ratio
        pos_embed = pos_embed.permute(2, 0, 1)[None].expand(x.shape[0], -1, -1, -1)
        return x + pos_embed

    def scratch_forward(self, features: list[torch.Tensor]) -> torch.Tensor:
        """
        Forward pass through the fusion blocks.

        Args:
            features (List[Tensor]): List of feature maps from different layers.

        Returns:
            Tensor: Fused feature map.
        """
        layer_1, layer_2, layer_3, layer_4 = features

        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)

        out = self.scratch.refinenet4(layer_4_rn, size=layer_3_rn.shape[2:])
        del layer_4_rn, layer_4

        out = self.scratch.refinenet3(out, layer_3_rn, size=layer_2_rn.shape[2:])
        del layer_3_rn, layer_3

        out = self.scratch.refinenet2(out, layer_2_rn, size=layer_1_rn.shape[2:])
        del layer_2_rn, layer_2

        out = self.scratch.refinenet1(out, layer_1_rn)
        del layer_1_rn, layer_1

        out = self.scratch.output_conv1(out)
        return out


def _make_fusion_block(features: int, has_residual: bool = True) -> nn.Module:
    return FeatureFusionBlock(features, nn.ReLU(inplace=True), has_residual)


def _make_scratch(in_channels: tuple[int, ...], features: int) -> nn.Module:
    """Project each of the four feature scales into a common channel width."""
    scratch = nn.Module()
    for index, channels in enumerate(in_channels, start=1):
        setattr(scratch, f"layer{index}_rn", nn.Conv2d(
            channels, features, kernel_size=3, stride=1, padding=1, bias=False,
        ))
    return scratch


class ResidualConvUnit(nn.Module):
    """Two activated 3x3 convolutions and an identity residual connection."""

    def __init__(self, features: int, activation: nn.Module) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(features, features, 3, stride=1, padding=1, bias=True)
        self.conv2 = nn.Conv2d(features, features, 3, stride=1, padding=1, bias=True)
        self.activation = activation
        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(self.activation(x))
        out = self.conv2(self.activation(out))
        return self.skip_add.add(out, x)


class FeatureFusionBlock(nn.Module):
    """Fuse an optional lateral feature, upsample, and project channels."""

    def __init__(
        self, features: int, activation: nn.Module, has_residual: bool = True,
    ) -> None:
        super().__init__()
        self.out_conv = nn.Conv2d(features, features, 1, stride=1, padding=0, bias=True)
        if has_residual:
            self.resConfUnit1 = ResidualConvUnit(features, activation)
        self.has_residual = has_residual
        self.resConfUnit2 = ResidualConvUnit(features, activation)
        self.skip_add = nn.quantized.FloatFunctional()

    def forward(
        self, x: torch.Tensor, lateral: torch.Tensor | None = None,
        size: tuple[int, int] | None = None,
    ) -> torch.Tensor:
        if self.has_residual:
            if lateral is None:
                raise ValueError("This DPT fusion stage requires a lateral feature")
            x = self.skip_add.add(x, self.resConfUnit1(lateral))
        x = self.resConfUnit2(x)
        x = custom_interpolate(
            x, size=size, scale_factor=2 if size is None else None,
            mode="bilinear", align_corners=True,
        )
        return self.out_conv(x)


def custom_interpolate(
    x: torch.Tensor,
    size: tuple[int, int] | None = None,
    scale_factor: float | None = None,
    mode: str = "bilinear",
    align_corners: bool = True,
) -> torch.Tensor:
    """
    Custom interpolate to avoid INT_MAX issues in nn.functional.interpolate.
    """
    if size is None:
        size = (int(x.shape[-2] * scale_factor), int(x.shape[-1] * scale_factor))

    INT_MAX = 1610612736

    input_elements = size[0] * size[1] * x.shape[0] * x.shape[1]

    if input_elements > INT_MAX:
        chunks = torch.chunk(x, chunks=(input_elements // INT_MAX) + 1, dim=0)
        interpolated_chunks = [
            nn.functional.interpolate(
                chunk, size=size, mode=mode, align_corners=align_corners
            )
            for chunk in chunks
        ]
        x = torch.cat(interpolated_chunks, dim=0)
        return x.contiguous()
    else:
        return nn.functional.interpolate(
            x, size=size, mode=mode, align_corners=align_corners
        )
