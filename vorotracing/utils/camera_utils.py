import numpy as np

def unpad_poses(p):
    """Remove the homogeneous bottom row from [..., 4, 4] pose matrices."""
    return p[..., :3, :4]

def pad_poses(p):
    """Pad [..., 3, 4] pose matrices with a homogeneous bottom row [0,0,0,1]."""
    bottom = np.broadcast_to([0, 0, 0, 1.], p[..., :1, :4].shape)
    return np.concatenate([p[..., :3, :4], bottom], axis=-2)

def transform_poses_pca(poses: np.ndarray, normalize_scale: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Transforms poses so principal components lie on XYZ axes.

    Args:
        poses: a (N, 3, 4) array containing the cameras' camera to world transforms.

    Returns:
        A tuple (poses, transform), with the transformed poses and the applied
        camera_to_world transforms.
    """
    assert poses.shape[0] > 1, "At least 2 poses are required to compute PCA"
    assert poses.shape[1] == 3 and poses.shape[2] == 4, "Poses must be (N, 3, 4) arrays"
    t = poses[:, :3, 3]
    t_mean = t.mean(axis=0)
    t = t - t_mean

    eigval, eigvec = np.linalg.eig(t.T @ t)
    # Sort eigenvectors in order of largest to smallest eigenvalue.
    inds = np.argsort(eigval)[::-1]
    eigvec = eigvec[:, inds]
    rot = eigvec.T
    if np.linalg.det(rot) < 0:
        rot = np.diag(np.array([1, 1, -1])) @ rot

    transform = np.concatenate([rot, rot @ -t_mean[:, None]], -1)
    poses_recentered = unpad_poses(transform @ pad_poses(poses))
    transform = np.concatenate([transform, np.eye(4)[3:]], axis=0)

    # Flip coordinate system if z component of y-axis is negative
    if poses_recentered.mean(axis=0)[2, 1] < 0:
        poses_recentered = np.diag(np.array([1, -1, -1])) @ poses_recentered
        transform = np.diag(np.array([1, -1, -1, 1])) @ transform

    # Just make sure it's it in the [-1, 1]^3 cube
    if normalize_scale:
        scale_factor = 1. / np.max(np.abs(poses_recentered[:, :3, 3]))
        poses_recentered[:, :3, 3] *= scale_factor
        transform = np.diag(np.array([scale_factor] * 3 + [1])) @ transform

    return poses_recentered, transform