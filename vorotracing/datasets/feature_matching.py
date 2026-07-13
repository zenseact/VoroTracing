import torch
from torch import Tensor
import numpy as np
from scipy.cluster.vq import kmeans, vq
from scipy.spatial.distance import cdist
from rich.progress import track


@torch.no_grad()
def get_pointcloud_from_images(
    images: Tensor,
    poses: Tensor,
    K: Tensor,
    num_nn_per_ref: int = 3,
    max_num_ref_views: int = 100,
    num_samples_per_pair: int = 15000,
    reproj_error_threshold: float = 2.0,
    setting: str = "fast",
) -> tuple[Tensor, Tensor]:
    """Get a dense pointcloud from images and poses using RoMaV2 feature matching
    and two-view triangulation.

    For each selected reference view, matches are computed against its nearest
    neighbors. Correspondences are triangulated via DLT and filtered by
    reprojection error and positive depth.

    Args:
        images: [N, H, W, 3] float32 in [0, 1]
        poses: [N, 3, 4] camera-to-world matrices
        K: [3, 3] camera intrinsic matrix (maps camera coords to pixel coords)
        num_nn_per_ref: nearest neighbor views per reference
        max_num_ref_views: maximum reference views to select via k-means
        num_samples_per_pair: correspondences to sample per image pair
        reproj_error_threshold: max reprojection error (pixels) for filtering
        setting: RoMaV2 quality setting ("turbo", "fast", "base", "precise")

    Returns:
        pointcloud: [M, 3] world coordinates
        colors: [M, 3] RGB in [0, 1]
    """
    from romav2 import RoMaV2

    N, H, W, _ = images.shape
    assert N >= 2, "Need at least 2 images for feature matching"

    num_ref_views = min(max_num_ref_views, N)
    num_nn = min(num_nn_per_ref, N - 1)

    romav2 = RoMaV2(RoMaV2.Cfg(setting=setting))

    selected_views, closest_neighbors = _find_overlapping_views(
        poses, num_ref_views, num_nn
    )
    proj_matrices = _build_projection_matrices(poses, K)

    all_points: list[Tensor] = []
    all_colors: list[Tensor] = []

    for i in track(
        range(len(selected_views)),
        description="Generating dense pointcloud with RoMaV2...",
    ):
        view_idx = selected_views[i]
        img1_np = images[view_idx].cpu().numpy()  # [H, W, 3] float32
        proj1 = proj_matrices[view_idx]

        for nn_idx in closest_neighbors[view_idx]:
            nn_idx = nn_idx.item()
            img2_np = images[nn_idx].cpu().numpy()
            proj2 = proj_matrices[nn_idx]

            preds = romav2.match(img1_np, img2_np)
            try:
                matches, confidence, _, _ = romav2.sample(preds, num_samples_per_pair)
            except RuntimeError:
                continue

            # matches: [M, 4] normalized (xA, yA, xB, yB) in [-1, 1]
            matches = matches.cpu()
            kpts1, kpts2 = RoMaV2.to_pixel_coordinates(matches, H, W, H, W)

            points_3d, mask = _triangulate_and_filter(
                kpts1, kpts2, proj1, proj2, reproj_error_threshold
            )

            x_idx = kpts1[:, 0].long().clamp(0, W - 1)
            y_idx = kpts1[:, 1].long().clamp(0, H - 1)
            colors = images[view_idx, y_idx, x_idx]

            all_points.append(points_3d[mask])
            all_colors.append(colors[mask])

    if not all_points:
        return torch.zeros(0, 3), torch.zeros(0, 3)

    return torch.cat(all_points), torch.cat(all_colors)


def _build_projection_matrices(poses: Tensor, K: Tensor) -> Tensor:
    """Build [N, 3, 4] world-to-pixel projection matrices from c2w poses and intrinsics.

    proj = K @ w2c[:3, :] where w2c = inv(c2w).
    """
    N = poses.shape[0]
    c2w = (
        torch.eye(4, device=poses.device, dtype=poses.dtype)
        .unsqueeze(0)
        .expand(N, -1, -1)
        .clone()
    )
    c2w[:, :3, :] = poses
    w2c = torch.linalg.inv(c2w)
    return K.unsqueeze(0) @ w2c[:, :3, :]


def _triangulate_and_filter(
    kpts1: Tensor,
    kpts2: Tensor,
    proj1: Tensor,
    proj2: Tensor,
    reproj_threshold: float,
) -> tuple[Tensor, Tensor]:
    """Triangulate via DLT and return a boolean mask for valid points.

    Points are rejected if they have negative depth in either view,
    high reprojection error, or non-finite coordinates.
    """
    points_3d = _triangulate_dlt(kpts1, kpts2, proj1, proj2)

    pts_h = torch.cat(
        [points_3d, torch.ones(points_3d.shape[0], 1, device=points_3d.device)],
        dim=1,
    )

    depth1 = (proj1[2:3] @ pts_h.T).squeeze(0)
    depth2 = (proj2[2:3] @ pts_h.T).squeeze(0)

    err1 = _reprojection_error(pts_h, kpts1, proj1)
    err2 = _reprojection_error(pts_h, kpts2, proj2)

    mask = (
        (depth1 > 0)
        & (depth2 > 0)
        & (torch.max(err1, err2) < reproj_threshold)
        & points_3d.isfinite().all(dim=1)
    )
    return points_3d, mask


def _triangulate_dlt(
    kpts1: Tensor, kpts2: Tensor, proj1: Tensor, proj2: Tensor
) -> Tensor:
    """DLT triangulation from two-view pixel correspondences.

    Solves the 4x4 homogeneous system per correspondence via SVD.

    Args:
        kpts1, kpts2: [M, 2] pixel coordinates (x, y)
        proj1, proj2: [3, 4] projection matrices (world to pixel)
    Returns:
        [M, 3] world coordinates
    """
    M = kpts1.shape[0]
    x1, y1 = kpts1[:, 0], kpts1[:, 1]
    x2, y2 = kpts2[:, 0], kpts2[:, 1]

    A = torch.zeros(M, 4, 4, device=kpts1.device, dtype=kpts1.dtype)
    A[:, 0] = x1[:, None] * proj1[2:3] - proj1[0:1]
    A[:, 1] = y1[:, None] * proj1[2:3] - proj1[1:2]
    A[:, 2] = x2[:, None] * proj2[2:3] - proj2[0:1]
    A[:, 3] = y2[:, None] * proj2[2:3] - proj2[1:2]

    _, _, Vh = torch.linalg.svd(A)
    X_h = Vh[:, -1]
    return X_h[:, :3] / X_h[:, 3:4]


def _reprojection_error(pts_h: Tensor, kpts: Tensor, proj: Tensor) -> Tensor:
    """Reprojection error (pixels) for homogeneous 3D points projected into a view.

    Args:
        pts_h: [M, 4] homogeneous world coordinates
        kpts: [M, 2] observed pixel coordinates (x, y)
        proj: [3, 4] projection matrix
    """
    projected = (proj @ pts_h.T).T  # [M, 3]
    projected_2d = projected[:, :2] / (projected[:, 2:3] + 1e-8)
    return (projected_2d - kpts).norm(dim=1)


def _find_overlapping_views(
    poses: Tensor, num_ref_views: int, num_nn_per_ref: int
) -> tuple[list[int], Tensor]:
    """Select reference views via k-means and find nearest neighbors for all views.

    Returns:
        selected_indices: sorted list of reference view indices
        closest_neighbors: [N, num_nn_per_ref] tensor of neighbor indices for every view
    """
    poses_flat = poses.reshape(-1, 12).detach().cpu()
    selected_indices = _select_reference_views(poses_flat, num_ref_views)

    dist = torch.cdist(poses_flat, poses_flat, p=2)
    dist.fill_diagonal_(float("inf"))
    _, closest_neighbors = torch.topk(dist, num_nn_per_ref, largest=False, dim=1)

    return selected_indices, closest_neighbors


def _select_reference_views(poses: Tensor, num_ref_views: int) -> list[int]:
    """Select K representative cameras via k-means clustering of flattened poses."""
    poses_np = poses.numpy()
    cluster_centers, _ = kmeans(poses_np, num_ref_views)
    cluster_assignments, _ = vq(poses_np, cluster_centers)

    selected_indices = []
    for k in range(num_ref_views):
        cluster_mask = cluster_assignments == k
        if not cluster_mask.any():
            continue
        cluster_members = poses_np[cluster_mask]
        distances = cdist(cluster_centers[k : k + 1], cluster_members)[0]
        nearest_idx = np.where(cluster_mask)[0][np.argmin(distances)]
        selected_indices.append(int(nearest_idx))

    return sorted(selected_indices)
