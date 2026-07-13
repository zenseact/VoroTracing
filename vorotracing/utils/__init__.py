import numpy as np
import torch


def inverse_softplus(x, beta, scale=1):
    # log(exp(scale*x)-1)/scale
    out = x / scale
    mask = x * beta < 20 * scale
    out[mask] = torch.log(torch.exp(beta * out[mask]) - 1 + 1e-10) / beta
    return out


def psnr(img1, img2):
    mse = ((img1 - img2) ** 2).view(-1, img1.shape[-1]).mean(0, keepdim=True)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))


def get_expon_lr_func(
    lr_init,
    lr_final,
    warmup_steps=0,
    max_steps=1_000,
):
    """
    Copied from Plenoxels

    Continuous learning rate decay function. Adapted from JaxNeRF
    The returned rate is lr_init when step=0 and lr_final when step=max_steps, and
    is log-linearly interpolated elsewhere (equivalent to exponential decay).
    If lr_delay_steps>0 then the learning rate will be scaled by some smooth
    function of lr_delay_mult, such that the initial learning rate is
    lr_init*lr_delay_mult at the beginning of optimization but will be eased back
    to the normal learning rate when steps>lr_delay_steps.
    :param conf: config subtree 'lr' or similar
    :param max_steps: int, the number of steps during optimization.
    :return HoF which takes step as input
    """

    def helper(step):
        if warmup_steps and step < warmup_steps:
            return lr_init * step / warmup_steps
        elif step > max_steps:
            return 0
        t = np.clip((step - warmup_steps) / (max_steps - warmup_steps), 0, 1)
        log_lerp = np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)
        return log_lerp

    return helper


def get_cosine_lr_func(
    lr_init,
    lr_final,
    warmup_steps=0,
    max_steps=10_000,
):
    """
    Copied from Plenoxels

    Continuous learning rate decay function. Adapted from JaxNeRF
    The returned rate is lr_init when step=0 and lr_final when step=max_steps, and
    is log-linearly interpolated elsewhere (equivalent to exponential decay).
    If lr_delay_steps>0 then the learning rate will be scaled by some smooth
    function of lr_delay_mult, such that the initial learning rate is
    lr_init*lr_delay_mult at the beginning of optimization but will be eased back
    to the normal learning rate when steps>lr_delay_steps.
    :param conf: config subtree 'lr' or similar
    :param max_steps: int, the number of steps during optimization.
    :return HoF which takes step as input
    """

    def helper(step):
        if warmup_steps and step < warmup_steps:
            return lr_init * step / warmup_steps
        elif step > max_steps:
            return 0.0
        lr_cos = lr_final + 0.5 * (lr_init - lr_final) * (
            1 + np.cos(np.pi * (step - warmup_steps) / (max_steps - warmup_steps))
        )
        return lr_cos

    return helper


def generate_camera_rays(K, T_c2w, width, height, device="cpu"):
    """
    Generate camera rays in world space.

    Args:
        K: (3, 3) camera intrinsics tensor
        T_c2w: (4, 4) camera-to-world transform tensor
        width: image width
        height: image height
        device: torch device string ('cpu' or 'cuda')

    Returns:
        rays: (num_rays, 6) tensor, [x, y, z, dx, dy, dz]
    """
    K = K.to(device)
    T_c2w = T_c2w.to(device)

    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    # Pixel grid
    u, v = torch.meshgrid(
        torch.arange(width, device=device),
        torch.arange(height, device=device),
        indexing="xy",
    )

    # Normalized camera coordinates
    x = (u - cx) / fx
    y = (v - cy) / fy
    dirs_cam = torch.stack([x, y, torch.ones_like(x)], dim=-1)
    dirs_cam = dirs_cam / torch.norm(dirs_cam, dim=-1, keepdim=True)

    # Transform directions to world space
    R = T_c2w[:3, :3]
    t = T_c2w[:3, 3]

    dirs_world = torch.matmul(dirs_cam.view(-1, 3), R.T)
    dirs_world = dirs_world / torch.norm(dirs_world, dim=-1, keepdim=True)

    origins_world = t.expand_as(dirs_world)

    # Concatenate [origin, direction]
    rays = torch.cat([origins_world, dirs_world], dim=-1)

    return rays  # shape: [num_rays, 6]
