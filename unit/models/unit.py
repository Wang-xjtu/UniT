# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE.txt file in the root directory of this source tree.
# Modified for UniT.
from __future__ import annotations

from pathlib import Path
from typing import Any, Self

import torch
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin

from unit.heads.camera_head import CameraHead
from unit.heads.dpt_head import DPTHead
from unit.models.aggregator import Aggregator
from unit.utils.pose_enc import encode_extrinsics, decode_extrinsics


class UniT(nn.Module, PyTorchModelHubMixin):
    """Group Causal Transformer for RGB and optional geometric observations.

    Modal tokenizers and four modal attention stages combine image content with
    ray-depth and pose features. The heads predict local point maps, confidence,
    and camera extrinsics. Public inference additionally expresses all predicted
    poses and world points in the first predicted camera's coordinate system.
    """

    @classmethod
    def from_checkpoint(
        cls, checkpoint: str | Path, *, device: str | torch.device = "auto"
    ) -> Self:
        """Strictly load a local PyTorch or Safetensors checkpoint."""
        from unit.inference.api import from_checkpoint

        return from_checkpoint(cls, checkpoint, device=device)

    @classmethod
    def from_pretrained(
        cls,
        repo_id: str,
        *,
        filename: str | None = None,
        revision: str | None = None,
        cache_dir: str | Path | None = None,
        local_files_only: bool = False,
        token: str | bool | None = None,
        device: str | torch.device = "auto",
        force_download: bool = False,
    ) -> Self:
        """Load UniT from a Hugging Face Hub repository.

        By default, prefer ``model.safetensors`` and fall back to ``checkpoint.pt``.
        Use ``from_checkpoint`` for an already downloaded file.
        """
        from huggingface_hub.constants import HF_HUB_OFFLINE

        from unit.inference.api import resolve_device

        if (
            not isinstance(repo_id, str)
            or repo_id.startswith(("./", "../", "/", "~"))
            or Path(repo_id).exists()
        ):
            raise ValueError(
                "from_pretrained expects a Hugging Face repository ID. "
                "Use from_checkpoint for local files."
            )

        options = dict(
            filename=filename, revision=revision,
            cache_dir=cache_dir, local_files_only=local_files_only,
            token=token, device=resolve_device(device), force_download=force_download,
        )
        # Older cached snapshots can contain weights without a config.json.
        if local_files_only or HF_HUB_OFFLINE:
            return cls._from_pretrained(model_id=repo_id, **options)
        return super().from_pretrained(repo_id, **options)

    @classmethod
    def _from_pretrained(cls, *, model_id: str, **kwargs: Any) -> Self:
        from unit.inference.api import _load_hub_model

        # Hub 0.x forwards these options even when unset.
        kwargs.pop("proxies", None)
        kwargs.pop("resume_download", None)
        return _load_hub_model(cls, model_id, **kwargs)

    def predict(
        self, inputs: Any, *, group_size: int | None = None,
        queue_capacity: int | None = None, hybrid: bool = False, target_size: int = 518,
        preprocess_mode: str = "crop", postprocess: bool = False,
    ) -> dict[str, Any]:
        """Infer from an input dictionary containing one original-resolution sequence.

        Use ``images`` with optional ``depth``, ``intrinsic``, and ``extrinsic``
        dictionary fields, without a batch axis. Images may be a directory,
        ordered paths, or RGB NumPy/PIL images. Direct image input is accepted
        as an RGB shorthand.
        Returns CPU tensors without a batch axis. ``group_size=None`` is offline;
        ``1`` is online. ``queue_capacity=None`` uses the sequence's frame count.
        ``hybrid=True`` uses hybrid offline-online inference: process the first
        group jointly, then continue one frame at a time.
        See the README for input/output shapes and coordinates.
        """
        from unit.inference.api import predict

        return predict(
            self, inputs, group_size=group_size,
            queue_capacity=queue_capacity, hybrid=hybrid, target_size=target_size,
            preprocess_mode=preprocess_mode, postprocess=postprocess,
        )

    def __init__(self):
        """Construct the released architecture; weights are loaded separately."""
        super().__init__()
        embed_dim = 1024
        self.aggregator = Aggregator()
        self.camera_head = CameraHead(dim_in=2 * embed_dim)
        self.point_head = DPTHead(dim_in=2 * embed_dim)
        self.keep_layer_idx = set(self.point_head.intermediate_layer_idx)
        self.keep_layer_idx.add(self.aggregator.depth - 1)

    def setup_caches(
        self,
        batch_size: int,
        height: int,
        width: int,
        patch_size: int,
        dtype: torch.dtype,
        device: torch.device,
        queue_capacity: int,
    ) -> None:
        self.aggregator.setup_caches(
            batch_size,
            (height // patch_size) * (width // patch_size) + self.aggregator.patch_start_idx,
            dtype,
            device,
            queue_capacity,
        )
        self.camera_head.setup_caches(
            batch_size, 1, torch.float32, device, queue_capacity
        )

    def reset_caches(self) -> None:
        self.aggregator.reset_caches()
        self.camera_head.reset_caches()

    def _encode_modal_inputs(
        self, ray_depth: torch.Tensor, extrinsics: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Append validity channels to ray-depth and 12D extrinsic features.

        ``ray_depth`` contains ray_x, ray_y, and camera-z depth,
        rather than camera-space XYZ. Missing modalities are encoded as zeros.
        """
        dtype = ray_depth.dtype
        eps = 1e-6

        pose_encoding = encode_extrinsics(extrinsics)

        B, S, H, W, _ = ray_depth.shape

        depth_valid_mask = (ray_depth[..., 2] > 0).to(dtype)
        ray_frame_valid = ray_depth[..., :2].abs().sum(dim=(-1, -2, -3)) > eps
        ray_valid_mask = (
            ray_frame_valid.to(dtype).view(B, S, 1, 1).expand(B, S, H, W)
        )
        pose_frame_valid = pose_encoding.abs().sum(dim=-1) > eps
        pose_valid_mask = pose_frame_valid.to(pose_encoding.dtype).unsqueeze(-1)

        ray_depth_features = torch.cat(
            [ray_depth, ray_valid_mask.unsqueeze(-1), depth_valid_mask.unsqueeze(-1)],
            dim=-1,
        )
        pose_features = torch.cat([pose_encoding, pose_valid_mask], dim=-1)

        ray_depth_features = ray_depth_features.permute(0, 1, 4, 2, 3).contiguous()
        pose_features = pose_features.contiguous()

        return ray_depth_features, pose_features

    def forward(
        self,
        images: torch.Tensor,
        ray_depth: torch.Tensor,
        extrinsics: torch.Tensor,
        group_size: int | torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Predict geometry from an already prepared Tensor batch.

        Args:
            images: RGB values in [0, 1], shaped [B, S, 3, H, W] or [S, 3, H, W].
            ray_depth: Ray-depth maps [B, S, H, W, 3], containing
                normalized ray_x/ray_y and camera-z depth in meters, not XYZ.
            extrinsics: World-to-camera observations [B, S, 3, 4]; zero matrices
                represent missing pose observations.
            group_size: Frames per causal group. Training accepts one size per
                batch item; evaluation is grouped by the inference scheduler.

        Returns:
            Camera-space ``cam_points`` [B, S, H, W, 3], ``cam_points_conf``
            [B, S, H, W], and world-to-camera ``extrinsics`` [B, S, 3, 4].
            First-camera coordinate normalization is performed by the inference
            layer, not this forward method.
        """
        if len(images.shape) == 4:
            images = images.unsqueeze(0)

        ray_depth_features, pose_features = self._encode_modal_inputs(ray_depth, extrinsics)

        aggregated_tokens_list, patch_start_idx = self.aggregator(
            images, ray_depth_features, pose_features, group_size,
            keep_layer_idx=self.keep_layer_idx,
        )

        predictions = {}

        with torch.autocast(device_type=images.device.type, enabled=False):
            pose_encoding = self.camera_head(aggregated_tokens_list, group_size)
            local_point_maps, point_confidence = self.point_head(
                aggregated_tokens_list,
                images=images,
                patch_start_idx=patch_start_idx,
            )
            camera_extrinsics = decode_extrinsics(pose_encoding)

        predictions["extrinsics"] = camera_extrinsics
        predictions["cam_points"] = local_point_maps
        predictions["cam_points_conf"] = point_confidence

        return predictions
