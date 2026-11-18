"""Optional geometric corrections applied after network prediction."""

import numpy as np
import torch

from unit.utils.geometry import intrinsics_to_camera_rays


def align_predictions_to_observations(
    local_point_maps: torch.Tensor,
    camera_extrinsics: torch.Tensor,
    observed_depths: torch.Tensor | None = None,
    observed_intrinsics: torch.Tensor | None = None,
    observed_extrinsics: torch.Tensor | None = None,
    confidence: torch.Tensor | None = None,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Optionally constrain rays, scale, and poses with supplied observations.

    Scale uses the mean of valid per-frame median depth ratios. Missing pixels
    and frames do not contribute. Pose alignment is loaded only when needed.
    This postprocessing is separate from the network's Modal Attention layers.
    """
    B, S, H, W, _ = local_point_maps.shape
    if B != 1:
        raise ValueError(f"Geometric postprocessing expects one sequence, got {B}")
    has_observed_intrinsics = observed_intrinsics is not None
    has_depth_observation = observed_depths is not None
    has_valid_depth_scale = False
    has_pose = observed_extrinsics is not None

    # Separate predicted ray directions from camera-axis depth.
    pred_depth = local_point_maps[..., -1:]
    pred_camera_rays = local_point_maps / torch.where(
        pred_depth != 0, pred_depth, torch.ones_like(pred_depth)
    )

    # Optionally replace predicted rays with rays from the observed intrinsics.
    if not has_observed_intrinsics:
        final_camera_rays = pred_camera_rays
    else:
        final_camera_rays = (
            intrinsics_to_camera_rays(observed_intrinsics.view(B * S, 3, 3), (H, W))
            .view(B, S, H, W, 3)
            .to(pred_camera_rays.dtype)
        )

    # Estimate scale only from valid observed depths.
    if not has_depth_observation:
        final_depths = pred_depth
        depth_scale = 1.0
    else:
        # observed_depths:  (B,S,H,W)
        # pred_depth: (B,S,H,W,1)
        # confidence:       (B,S,H,W)
        observed_depths = observed_depths.unsqueeze(-1)  # (B,S,H,W,1)
        observed_flat = observed_depths.reshape(B, S, -1)  # (B,S,N)
        pred_flat = pred_depth.reshape(B, S, -1)  # (B,S,N)
        conf_flat = (confidence.reshape(B, S, -1) if confidence is not None
                     else torch.ones_like(pred_flat))
        # filter valid points
        finite_conf = torch.where(torch.isfinite(conf_flat), conf_flat, torch.nan)
        thres = torch.nanquantile(finite_conf, 0.1, dim=-1, keepdim=True)
        valid_mask = (
            (conf_flat >= thres) & torch.isfinite(conf_flat)
            & torch.isfinite(observed_flat) & (observed_flat > 0)
            & torch.isfinite(pred_flat) & (pred_flat > eps)
        )
        ratio_flat = observed_flat / (pred_flat + eps)  # (B,S,N)
        ratio_flat = torch.where(
            valid_mask, ratio_flat, torch.full_like(ratio_flat, float("nan"))
        )
        median_ratio = torch.nanmedian(ratio_flat, dim=-1).values  # (B,S)
        valid_scales = median_ratio[torch.isfinite(median_ratio) & (median_ratio > 0)]
        has_valid_depth_scale = valid_scales.numel() > 0
        depth_scale = valid_scales.mean() if has_valid_depth_scale else 1.0

        final_depths = pred_depth * depth_scale

    # Optionally constrain camera extrinsics using observed poses.
    if not has_pose:
        final_extrinsics = camera_extrinsics
        extr_scale = 1.0
    else:
        umeyama_scale = _pose_scale(camera_extrinsics, observed_extrinsics, eps)
        extr_scale = torch.tensor(
            1.0 / float(umeyama_scale),
            dtype=final_depths.dtype,
            device=final_depths.device,
        )
        final_extrinsics = observed_extrinsics.to(camera_extrinsics.dtype)

    # Keep local points and camera translation on the same scale.
    if has_valid_depth_scale and (not has_pose):
        # Observed depth constrains the scale of predicted translation.
        final_extrinsics = final_extrinsics.clone()
        final_extrinsics[:, :, :3, 3] *= depth_scale
    elif has_pose and (not has_valid_depth_scale):
        # Observed translation constrains the scale of predicted depth.
        final_depths = final_depths * extr_scale

    # Return corrected local point maps and world-to-camera extrinsics.
    final_camera_points = final_camera_rays * final_depths

    return final_camera_points, final_extrinsics


def _pose_scale(predicted: torch.Tensor, observed: torch.Tensor, eps: float) -> float:
    """Estimate the existing pose scale, with a short/degenerate-path fallback."""
    pred = predicted[0].detach().float().cpu().numpy()
    obs = observed[0].detach().float().cpu().numpy()
    frames = len(pred)
    if frames >= 3:
        try:
            from evo.core.geometry import GeometryException
            from unit.utils.pose_align import align_poses_umeyama
        except ImportError as exc:
            raise ImportError(
                "Pose postprocessing requires: pip install -e '.[postprocess]'"
            ) from exc
        try:
            _, _, scale, _ = align_poses_umeyama(
                pred, obs, ransac=frames >= 10,
                return_aligned=True, random_state=42,
            )
            if np.isfinite(scale) and scale > eps:
                return float(scale)
        except (GeometryException, np.linalg.LinAlgError, ValueError):
            pass  # Short or degenerate trajectories cannot determine Sim(3).
    predicted_steps = np.linalg.norm(np.diff(pred[:, :3, 3:4], axis=0), axis=1)
    observed_steps = np.linalg.norm(np.diff(obs[:, :3, 3:4], axis=0), axis=1)
    valid = np.isfinite(predicted_steps) & np.isfinite(observed_steps) & (predicted_steps > eps) & (observed_steps > eps)
    return float(np.mean(predicted_steps[valid] / observed_steps[valid])) if valid.any() else 1.0
