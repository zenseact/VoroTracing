# This file is heavily inspired by https://github.com/SuLvXiangXin/zipnerf-pytorch/blob/main/internal/datasets.py
import numpy as np
import os
from typing import Literal, Type
from dataclasses import dataclass, field
import pycolmap
from pycolmap import Camera
from PIL import Image
import torch
import abc
from rich.progress import track
from rich.console import Console

from vorotracing.utils.camera_utils import transform_poses_pca

CONSOLE = Console(width=120)


@dataclass
class DatasetConfig:
    dataset_type: Type = None
    data_path: str = "data/mipnerf360"
    scene: str = "garden"
    antialiasing: Literal["none", "jitter", "full"] = "none"
    """For antialiasing we randomize the ray within the pixel cone when applying jitter is true.
    'full' generates 5 rays per pixel (center + 4 at 0.25 offsets) to be averaged after rendering."""
    init_points: Literal["original", "romav2"] = "romav2"
    """Source for initial 3D points. 'original' uses the original 3D points from the dataset.
    'romav2' generates a dense pointcloud via RoMaV2 feature matching and triangulation
    (requires _load_data to set self.K with the 3x3 intrinsic matrix)."""
    pca_transform: bool = True
    """Apply the VoroTracing PCA recentering/rotation to COLMAP poses and points.
    Checkpoints trained in raw COLMAP coordinates must be rendered with this disabled.
    """

    # Offsets for "full" antialiasing: center + 4 corners at 0.25 from center
    FULL_AA_OFFSETS: tuple = (
        (0.5, 0.5),
        (0.25, 0.25),
        (0.75, 0.25),
        (0.25, 0.75),
        (0.75, 0.75),
    )

    def load(
        self,
        split: Literal["train", "test"],
        downsample_factor: int = 1,
        return_alphas: bool = True,
    ) -> "Dataset":
        # Update config with runtime parameters
        return self.dataset_type(
            self,
            split,
            downsample_factor=downsample_factor,
            return_alphas=return_alphas,
        )


class Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        config: DatasetConfig,
        split: Literal["train", "test"],
        downsample_factor: int = 1,
        return_alphas: bool = False,
    ):
        self.config = config
        self.split = split
        self.return_alphas = return_alphas

        CONSOLE.log(
            f"Loading images from {self.config.scene} in {self.config.data_path}, downsample factor: {downsample_factor}, split: {self.split}"
        )

        self._init_dataset(downsample_factor=downsample_factor)

    def _init_dataset(self, downsample_factor: int = 1, **kwargs):
        """Initialize the dataset.
        This can be called at any time to reinitialize the dataset. For exmaple if one wants
        to change a config (e.g. the downsample factor).

        Args:
            **kwargs: changes to the config
        """
        for k, v in kwargs.items():
            setattr(self.config, k, v)

        self.downsample_factor = downsample_factor

        self._load_data()

        if self.split == "train" and self.config.init_points == "romav2":
            from vorotracing.datasets.feature_matching import get_pointcloud_from_images

            assert hasattr(self, "K"), (
                "RoMaV2 init requires self.K (3x3 intrinsic matrix) to be set by _load_data()"
            )
            CONSOLE.log(
                "Replacing SfM points with RoMaV2 feature-matched pointcloud..."
            )
            self.points3D, self.points3D_colors = get_pointcloud_from_images(
                images=self.rgbs,
                poses=self.poses,
                K=self.K,
            )

        assert (
            hasattr(self, "poses")
            and hasattr(self, "rgbs")
            and hasattr(self, "alphas")
            and hasattr(self, "points3D")
            and hasattr(self, "points3D_colors")
        ), "Missing required attributes for dataset"

        assert self.poses.ndim == 3 and self.poses.shape[-2:] == (3, 4), (
            "Poses must be 3D (N, 3, 4)"
        )
        assert self.rgbs.ndim == 4 and self.rgbs.shape[-1] == 3, (
            "Rgbs must be 4D (N, H, W, 3)"
        )
        assert self.alphas.ndim == 4, "Alphas must be 4D (N, H, W, 1)"
        assert self.points3D.ndim == 2, "Points3D must be 2D (M, 3)"
        assert self.points3D_colors.ndim == 2, "Points3D colors must be 2D (M, 3)"

        # Create default ray directions for regular camera
        H, W = self.rgbs.shape[1:3]
        x = np.arange(W, dtype=np.float32) + 0.5
        y = np.arange(H, dtype=np.float32) + 0.5
        x, y = np.meshgrid(x, y)
        pix_coords = np.stack([x, y], axis=-1).reshape(-1, 2)
        ip_coords = self._img_coords_to_camera_coords(pix_coords).numpy()
        ip_coords = np.concatenate([ip_coords, np.ones_like(ip_coords[:, :1])], axis=-1)
        ray_dirs = ip_coords / np.linalg.norm(ip_coords, axis=-1, keepdims=True)

        self.default_ray_dirs = torch.from_numpy(ray_dirs.reshape(H, W, 3))

    def __len__(self):
        return self.rgbs.shape[0]

    @abc.abstractmethod
    def _load_data(self) -> None:
        """Load data from disk.
        This should create 5 attributes:
            - self.poses: (N, 3, 4) torch.tensor
            - self.rgbs: (N, H, W, 3) torch.tensor
            - self.alphas: (N, H, W, 1) torch.tensor
            - self.points3D: (M, 3) torch.tensor
            - self.points3D_colors: (M, 3) torch.tensor
        """
        pass

    @abc.abstractmethod
    def _img_coords_to_camera_coords(self, img_coords: torch.Tensor) -> torch.Tensor:
        """Convert image coordinates to camera coordinates.
        Different datasets may have different camera types or methods to convert image coordinates to camera coordinates."""
        pass

    def __getitem__(self, item):
        """Get a single Image."""
        out = {}
        out["rgbs"] = self.rgbs[item].cuda()
        if self.return_alphas:
            out["alphas"] = self.alphas[item].cuda()

        H, W = self.default_ray_dirs.shape[:2]

        if self.config.antialiasing == "full":
            # Build a full pixel grid for this image and generate 5 AA rays per pixel
            y = torch.arange(H, dtype=torch.long)
            x = torch.arange(W, dtype=torch.long)
            yy, xx = torch.meshgrid(y, x, indexing="ij")
            img_ids = torch.full((H * W,), item, dtype=torch.long)
            rays = self._generate_full_aa_rays(img_ids, xx.reshape(-1), yy.reshape(-1))
            out["rays"] = rays.reshape(H, W, 5, -1).cuda()
            return out
        else:
            out["ray_dirs"] = self.default_ray_dirs.reshape(H * W, -1)
            out["poses"] = self.poses[item].unsqueeze(0).expand(H * W, -1, -1)

            out = self._preprocess_batch(out)
            out["rays"] = out["rays"].reshape(H, W, -1)
            return out

    def sample_batch(self, rays_per_batch: int, device: torch.device = "cuda"):
        assert self.split == "train", "next_batch() is only for training"
        N, H, W, _ = self.rgbs.shape
        out = {}

        img_ids = torch.randint(0, N, (rays_per_batch,))
        xs = torch.randint(0, W, (rays_per_batch,))
        ys = torch.randint(0, H, (rays_per_batch,))

        if self.config.antialiasing == "full":
            # Full antialiasing: generate 5 rays per pixel
            rays = self._generate_full_aa_rays(img_ids, xs, ys, device)
            out = {
                "rays": rays,
                "rgbs": self.rgbs[img_ids, ys, xs].to(device, non_blocking=True),
            }
            if self.return_alphas:
                out["alphas"] = self.alphas[img_ids, ys, xs].to(
                    device, non_blocking=True
                )
            return out

        if self.split != "train" or self.config.antialiasing == "none":
            # No jitter: sample at pixel center
            out["rgbs"] = self.rgbs[img_ids, ys, xs]
            if self.return_alphas:
                out["alphas"] = self.alphas[img_ids, ys, xs]
            xs = xs.float() + 0.5
            ys = ys.float() + 0.5
        elif self.config.antialiasing == "jitter":
            # Jitter ray but use discrete pixel color
            out["rgbs"] = self.rgbs[img_ids, ys, xs]
            if self.return_alphas:
                out["alphas"] = self.alphas[img_ids, ys, xs]
            xs = xs.float() + torch.rand(rays_per_batch)
            ys = ys.float() + torch.rand(rays_per_batch)
        else:
            raise ValueError(f"Invalid antialiasing method: {self.config.antialiasing}")

        ip_coords = self._img_coords_to_camera_coords(torch.stack([xs, ys], dim=-1))
        ip_coords = torch.cat([ip_coords, torch.ones_like(ip_coords[:, :1])], dim=-1)
        ray_dirs = ip_coords / torch.linalg.norm(ip_coords, dim=-1, keepdim=True)

        out["poses"] = self.poses[img_ids]
        out["ray_dirs"] = ray_dirs

        return self._preprocess_batch(out, device=device)

    def sample_patch_batch(
        self, rays_per_batch: int, patch_size: int = 16, device: torch.device = "cuda"
    ):
        """Sample contiguous PxP pixel patches instead of random rays.

        Rays are laid out patch-contiguous (patch-major, row-major within patch) so a
        32-thread warp covers a spatially-local sub-block of a patch -> warp-coherent
        memory access in the trace kernels (measured ~1.5x faster fwd+bwd). Also keeps
        spatial structure for a future SSIM loss. Returns the same keys as
        sample_batch plus 'patch_size' and 'n_patches'.
        """
        assert self.split == "train", "sample_patch_batch is only for training"
        assert self.config.antialiasing != "full", (
            "patch batching does not support full antialiasing yet"
        )
        N, H, W, _ = self.rgbs.shape
        P = int(patch_size)
        assert W >= P and H >= P, f"patch_size {P} larger than image {H}x{W}"
        n_patches = max(1, rays_per_batch // (P * P))

        img_ids_p = torch.randint(0, N, (n_patches,))
        x0 = torch.randint(0, W - P + 1, (n_patches,))
        y0 = torch.randint(0, H - P + 1, (n_patches,))
        dyy, dxx = torch.meshgrid(
            torch.arange(P), torch.arange(P), indexing="ij"
        )
        dxx = dxx.reshape(-1)  # [P*P] row-major within patch
        dyy = dyy.reshape(-1)
        xs = (x0[:, None] + dxx[None, :]).reshape(-1)  # [n_patches*P*P], patch-contiguous
        ys = (y0[:, None] + dyy[None, :]).reshape(-1)
        img_ids = img_ids_p[:, None].expand(n_patches, P * P).reshape(-1)

        out = {"rgbs": self.rgbs[img_ids, ys, xs]}
        if self.return_alphas:
            out["alphas"] = self.alphas[img_ids, ys, xs]
        if self.config.antialiasing == "jitter":
            xsf = xs.float() + torch.rand(xs.shape[0])
            ysf = ys.float() + torch.rand(ys.shape[0])
        else:  # "none": pixel center
            xsf = xs.float() + 0.5
            ysf = ys.float() + 0.5

        ip_coords = self._img_coords_to_camera_coords(torch.stack([xsf, ysf], dim=-1))
        ip_coords = torch.cat([ip_coords, torch.ones_like(ip_coords[:, :1])], dim=-1)
        ray_dirs = ip_coords / torch.linalg.norm(ip_coords, dim=-1, keepdim=True)
        out["poses"] = self.poses[img_ids]
        out["ray_dirs"] = ray_dirs

        res = self._preprocess_batch(out, device=device)
        res["patch_size"] = P
        res["n_patches"] = n_patches
        return res

    def _preprocess_batch(self, batch: dict, device: torch.device = "cuda"):
        """
        Move to GPU and preprocess batch.
        """
        poses = batch["poses"].to(device, non_blocking=True)
        ray_dirs = batch["ray_dirs"].to(device, non_blocking=True)
        rgbs = batch["rgbs"].to(device, non_blocking=True)

        R = poses[:, :3, :3]
        t = poses[:, :3, 3]

        dirs_world = torch.bmm(R, ray_dirs.unsqueeze(-1)).squeeze(-1)
        rays = torch.cat([t, dirs_world], dim=-1)

        if self.return_alphas:
            alphas = batch["alphas"].to(device, non_blocking=True)
            return {"rays": rays, "rgbs": rgbs, "alphas": alphas}
        else:
            return {"rays": rays, "rgbs": rgbs}

    def _generate_full_aa_rays(
        self,
        img_ids: torch.Tensor,
        xs: torch.Tensor,
        ys: torch.Tensor,
        device: torch.device = None,
    ) -> torch.Tensor:
        """Generate full-antialiasing rays for an arbitrary set of pixels."""
        all_rays = []
        for ox, oy in self.config.FULL_AA_OFFSETS:
            xs_offset = xs.float() + ox
            ys_offset = ys.float() + oy

            ip_coords = self._img_coords_to_camera_coords(
                torch.stack([xs_offset, ys_offset], dim=-1)
            )
            ip_coords = torch.cat(
                [ip_coords, torch.ones_like(ip_coords[:, :1])], dim=-1
            )
            ray_dirs = ip_coords / torch.linalg.norm(ip_coords, dim=-1, keepdim=True)

            poses = self.poses[img_ids]
            if device is not None:
                poses = poses.to(device, non_blocking=True)
                ray_dirs = ray_dirs.to(device, non_blocking=True)

            R = poses[:, :3, :3]
            t = poses[:, :3, 3]

            dirs_world = torch.bmm(R, ray_dirs.unsqueeze(-1)).squeeze(-1)
            all_rays.append(torch.cat([t, dirs_world], dim=-1))

        # Stack to (B, 5, 6)
        return torch.stack(all_rays, dim=1)

    def pin_memory(self):
        self.poses = self.poses.pin_memory()
        self.rgbs = self.rgbs.pin_memory()
        self.alphas = self.alphas.pin_memory()

    def get_initial_points(self):
        return self.points3D, self.points3D_colors


def get_cam_ray_dirs(camera: Camera):
    x = np.arange(camera.width, dtype=np.float32) + 0.5
    y = np.arange(camera.height, dtype=np.float32) + 0.5
    x, y = np.meshgrid(x, y)
    pix_coords = np.stack([x, y], axis=-1).reshape(-1, 2)
    ip_coords = camera.cam_from_img(pix_coords)
    ip_coords = np.concatenate([ip_coords, np.ones_like(ip_coords[:, :1])], axis=-1)
    ray_dirs = ip_coords / np.linalg.norm(ip_coords, axis=-1, keepdims=True)
    return torch.tensor(ray_dirs, dtype=torch.float32)


@dataclass
class ColmapConfig(DatasetConfig):
    dataset_type: Type = field(default_factory=lambda: ColmapDataset)
    split_stride: int = 8


class ColmapDataset(Dataset):
    """COLMAP-based dataset (Mip-NeRF360, Deep Blending, Tanks & Temples, etc.)."""

    def _img_coords_to_camera_coords(self, img_coords: torch.Tensor) -> torch.Tensor:
        """Convert image coordinates to camera coordinates."""
        coords = self.camera.cam_from_img(img_coords)
        return torch.tensor(coords, dtype=torch.float32)

    def _load_data(self) -> None:
        """Load images from disk."""
        assert self.downsample_factor in [1, 2, 4, 8]

        if self.downsample_factor > 1:
            image_dir_sufix = f"_{self.downsample_factor}"
        else:
            image_dir_sufix = ""

        images_dir = os.path.join(
            self.config.data_path, self.config.scene, "images" + image_dir_sufix
        )
        colmap_dir = os.path.join(self.config.data_path, self.config.scene, "sparse/0/")
        if not os.path.exists(colmap_dir):
            raise ValueError(f"COLMAP directory {colmap_dir} not found")
        if not os.path.exists(images_dir):
            raise ValueError(f"Images directory {images_dir} not found")

        reconstruction = pycolmap.Reconstruction()
        reconstruction.read(colmap_dir)

        # Get all images and poses first to ensure consistent coordinate system
        all_images = []
        all_image_poses = []
        all_names = sorted(im.name for im in reconstruction.images.values())

        for name in all_names:
            image = None
            for image_id in reconstruction.images:
                if reconstruction.images[image_id].name == name:
                    image = reconstruction.images[image_id]
                    break
            if image is None:
                continue

            all_images.append(image)
            c2w = torch.tensor(
                image.cam_from_world().inverse().matrix(), dtype=torch.float32
            )
            all_image_poses.append(c2w)

        all_image_poses = torch.stack(all_image_poses).numpy()

        # Get image width and height (downscaled versions do not match colmap camera)
        im = Image.open(os.path.join(images_dir, all_images[0].name))
        self.img_wh = im.size
        im.close()

        # Get camera and rescale it to the image width and height
        if len(reconstruction.cameras) > 1:
            raise ValueError("Multiple cameras are not supported")
        self.camera = list(reconstruction.cameras.values())[0]
        self.camera.rescale(self.img_wh[0], self.img_wh[1])
        self.K = torch.tensor(self.camera.calibration_matrix(), dtype=torch.float32)

        # Compute transform on ALL poses, unless a checkpoint was trained in raw
        # COLMAP coordinates.
        if self.config.pca_transform:
            all_image_poses, origin_transform = transform_poses_pca(all_image_poses)
        else:
            origin_transform = np.eye(4, dtype=np.float32)

        # Now filter for the specific split
        images = []
        image_poses = []

        indices = np.arange(len(all_names))
        if self.split == "train":
            split_indices = indices % self.config.split_stride != 0
        elif self.split == "test":
            split_indices = indices % self.config.split_stride == 0
        else:
            raise ValueError(f"Invalid split: {self.split}")

        for i, name in enumerate(all_names):
            if split_indices[i]:
                images.append(all_images[i])
                image_poses.append(all_image_poses[i])

        image_poses = np.array(image_poses)

        poses = []
        rgbs = []
        alphas = []
        for i in track(range(len(images)), description="Loading images..."):
            img_pose = torch.tensor(image_poses[i], dtype=torch.float32)

            img = Image.open(os.path.join(images_dir, images[i].name))
            if np.array(img).shape[-1] == 4:
                img = img.convert("RGBA")
                rgbas = torch.tensor(np.array(img), dtype=torch.float32) / 255.0
                img_alpha = rgbas[..., 3:4]
                img_rgb = rgbas[..., :3] * img_alpha + (1 - img_alpha)
            else:
                img = img.convert("RGB")
                img_rgb = torch.tensor(np.array(img), dtype=torch.float32) / 255.0
                img_alpha = torch.ones_like(img_rgb[..., :1])
            img.close()

            poses.append(img_pose)
            rgbs.append(img_rgb)
            alphas.append(img_alpha)

        self.poses = torch.stack(poses)
        self.rgbs = torch.stack(rgbs)
        self.alphas = torch.stack(alphas)
        self.image_names = [images[i].name for i in range(len(images))]

        points3D = []
        points3D_colors = []
        for point in reconstruction.points3D.values():
            points3D.append(point.xyz)
            points3D_colors.append(point.color)

        points3D = np.array(points3D, dtype=np.float32)
        # Apply the same coordinate transform to points as to camera poses.
        points3D = np.concatenate([points3D, np.ones_like(points3D[:, :1])], axis=-1)
        points3D = (origin_transform @ points3D.T).T[:, :3]

        self.points3D = torch.tensor(points3D, dtype=torch.float32)
        self.points3D_colors = (
            torch.tensor(points3D_colors, dtype=torch.float32) / 255.0
        )

        num_rays = self.rgbs.shape[0] * self.rgbs.shape[1] * self.rgbs.shape[2]
        CONSOLE.log(f"Loaded {num_rays} rays, {self.points3D.shape[0]} initial points")
