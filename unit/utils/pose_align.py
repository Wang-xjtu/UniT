# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
# Modified for UniT. See licenses/Apache-2.0.txt for the license text.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np


def transpose_last_two_axes(arr):
    """
    for np < 2
    """
    if arr.ndim < 2:
        return arr
    axes = list(range(arr.ndim))
    # swap the last two
    axes[-2], axes[-1] = axes[-1], axes[-2]
    return arr.transpose(axes)


def affine_inverse_np(A: np.ndarray):
    R = A[..., :3, :3]
    T = A[..., :3, 3:]
    P = A[..., 3:, :]
    return np.concatenate(
        [
            np.concatenate(
                [transpose_last_two_axes(R), -transpose_last_two_axes(R) @ T], axis=-1
            ),
            P,
        ],
        axis=-2,
    )


def _to44(ext):
    if ext.shape[1] == 3:
        out = np.eye(4)[None].repeat(len(ext), 0)
        out[:, :3, :4] = ext
        return out
    return ext


def _poses_from_ext(ext_ref, ext_est):
    ext_ref = _to44(ext_ref)
    ext_est = _to44(ext_est)
    pose_ref = affine_inverse_np(ext_ref)
    pose_est = affine_inverse_np(ext_est)
    return pose_ref, pose_est


def _umeyama_sim3_from_paths(pose_ref, pose_est):
    from evo.core.trajectory import PosePath3D

    path_ref = PosePath3D(poses_se3=pose_ref.copy())
    path_est = PosePath3D(poses_se3=pose_est.copy())
    r, t, s = path_est.align(path_ref, correct_scale=True)
    pose_est_aligned = np.stack(path_est.poses_se3)
    return r, t, s, pose_est_aligned


def _apply_sim3_to_poses(poses, r, t, s):
    out = poses.copy()
    Ri = poses[:, :3, :3]
    ti = poses[:, :3, 3]
    out[:, :3, :3] = r @ Ri
    out[:, :3, 3] = (r @ (s * ti.T)).T + t
    return out


def _median_nn_thresh(pose_ref, pose_est_aligned):
    P_ref = pose_ref[:, :3, 3]
    P_est = pose_est_aligned[:, :3, 3]
    dists = []
    for p in P_est:
        dd = np.linalg.norm(P_ref - p[None, :], axis=1)
        dists.append(dd.min())
    return float(np.median(dists)) if dists else 0.0


def _ransac_align_sim3(
    pose_ref, pose_est, sub_n=None, inlier_thresh=None, max_iters=10, random_state=None
):
    from evo.core.geometry import GeometryException

    rng = np.random.default_rng(random_state)
    N = pose_ref.shape[0]
    idx_all = np.arange(N)
    if sub_n is None:
        sub_n = max(3, (N + 1) // 2)
    else:
        sub_n = max(3, min(sub_n, N))

    # Pre-alignment + default threshold
    r0, t0, s0, pose_est0 = _umeyama_sim3_from_paths(pose_ref, pose_est)
    if inlier_thresh is None:
        inlier_thresh = _median_nn_thresh(pose_ref, pose_est0)

    P_ref_all = pose_ref[:, :3, 3]

    best_model = (r0, t0, s0)
    best_inliers = None
    best_score = (-1, np.inf)  # (num_inliers, mean_err)

    for _ in range(max_iters):
        sample = rng.choice(idx_all, size=sub_n, replace=False)
        try:
            r, t, s, _ = _umeyama_sim3_from_paths(pose_ref[sample], pose_est[sample])
        except (GeometryException, ValueError, np.linalg.LinAlgError):
            continue
        pose_h = _apply_sim3_to_poses(pose_est, r, t, s)
        P_h = pose_h[:, :3, 3]
        errs = np.linalg.norm(P_h - P_ref_all, axis=1)  # Match by same index
        inliers = errs <= inlier_thresh
        k = int(inliers.sum())
        mean_err = float(errs[inliers].mean()) if k > 0 else np.inf
        if (k > best_score[0]) or (k == best_score[0] and mean_err < best_score[1]):
            best_score = (k, mean_err)
            best_model = (r, t, s)
            best_inliers = inliers

    # Fit again with best inliers
    if best_inliers is not None and best_inliers.sum() >= 3:
        r, t, s, _ = _umeyama_sim3_from_paths(
            pose_ref[best_inliers], pose_est[best_inliers]
        )
    else:
        r, t, s = best_model
    return r, t, s


def align_poses_umeyama(
    ext_ref: np.ndarray,
    ext_est: np.ndarray,
    return_aligned=False,
    ransac=False,
    sub_n=None,
    inlier_thresh=None,
    ransac_max_iters=10,
    random_state=None,
):
    """
    Align estimated trajectory to reference using Umeyama Sim(3).
    Default no RANSAC; if ransac=True, use RANSAC (max iterations default 10).
    - sub_n defaults to half the number of frames (rounded up, at least 3)
    - inlier_thresh defaults to median of "distance from each estimated pose to
      nearest reference pose after pre-alignment"
    Returns rotation (3x3), translation (3,), scale; optionally returns aligned extrinsics (4x4).
    """
    pose_ref, pose_est = _poses_from_ext(ext_ref, ext_est)

    if not ransac:
        r, t, s, pose_est_aligned = _umeyama_sim3_from_paths(pose_ref, pose_est)
    else:
        r, t, s = _ransac_align_sim3(
            pose_ref,
            pose_est,
            sub_n=sub_n,
            inlier_thresh=inlier_thresh,
            max_iters=ransac_max_iters,
            random_state=random_state,
        )
        pose_est_aligned = _apply_sim3_to_poses(pose_est, r, t, s)

    if return_aligned:
        ext_est_aligned = affine_inverse_np(pose_est_aligned)
        return r, t, s, ext_est_aligned
    return r, t, s
