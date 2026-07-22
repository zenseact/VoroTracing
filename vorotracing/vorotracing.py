import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Optional, List, Literal
from pathlib import Path
from typing import Union

import vorotracing.cuda as ops


@dataclass
class VoroTracingConfig:
    oct_map_res: int = 8
    """Resolution of diffuse octahedral texture map (per side). Total texels = res^2.
    Must match the compile-time `OCTMAP_RES` constant in tracing_octmap.cu."""
    spec_oct_map_res: int = 8
    """Resolution of specular (view-dependent) octahedral texture map (per side).
    Must match the compile-time `OCTMAP_RES_SPEC` constant in tracing_octmap.cu.
    Lower values reduce view-dependent capacity / overfitting."""
    activation_scale: float = 1.0
    """Scale factor for the activation function of density."""
    up_direction: List[float] = field(default_factory=lambda: [0, 0, 0])
    """Up direction for the scene. If [0, 0, 0], use the global up direction. This is unused in the model but required if one wants to display correctly in the viewer after training."""


class VoroTracing(torch.nn.Module):
    def __init__(
        self,
        config: VoroTracingConfig,
        points: torch.Tensor,
        points_diffuse: Optional[torch.Tensor] = None,
        points_specular: Optional[torch.Tensor] = None,
        points_density: Optional[torch.Tensor] = None,
        points_adjacency: Optional[torch.Tensor] = None,
        points_adjacency_offsets: Optional[torch.Tensor] = None,
        device: torch.device = torch.device("cuda"),
    ) -> None:
        super().__init__()
        self.config = config
        self.device = device
        self.optimizer = None

        assert points is not None, "points must be provided"
        assert points.shape[0] > 0, "points must have at least 1 point"
        assert points.shape[1] == 3, "points must have 3 dimensions"
        assert points.dtype == torch.float32, "points must be a float32 tensor"

        diff_map_size = 3 * config.oct_map_res * config.oct_map_res
        spec_map_size = 3 * config.spec_oct_map_res * config.spec_oct_map_res
        n_pts = points.shape[0]

        if points_diffuse is not None:
            assert points_diffuse.shape == (n_pts, diff_map_size)
            self.att_diffuse = nn.Parameter(points_diffuse.to(self.device))
        else:
            # logit(0.5) = 0.0, so sigmoid(0) = 0.5 (neutral gray)
            self.att_diffuse = nn.Parameter(
                torch.zeros(
                    n_pts, diff_map_size, device=self.device, dtype=torch.float32
                )
            )

        if points_specular is not None:
            assert points_specular.shape == (n_pts, spec_map_size)
            self.att_specular = nn.Parameter(points_specular.to(self.device))
        else:
            # Specular is an unbounded signed residual added on top of the
            # sigmoid-activated diffuse, so zero init = "no specular contribution".
            self.att_specular = nn.Parameter(
                torch.zeros(
                    n_pts, spec_map_size, device=self.device, dtype=torch.float32
                )
            )

        # Point density
        if points_density is not None:
            assert points_density.shape[0] == points.shape[0], (
                "points and points_density must have the same number of points"
            )
            assert points_density.shape[1] == 1, "points_density must have 1 channel"
            assert points_density.dtype == torch.float32, (
                "points_density must be a float32 tensor"
            )

            self.density = nn.Parameter(points_density.to(self.device))
        else:
            self.density = nn.Parameter(
                torch.zeros(points.shape[0], 1, device=self.device, dtype=torch.float32)
            )

        self.triangulation = None
        self.bvh = ops.NNBVH()

        # Point adjacency and offsets
        if points_adjacency is not None and points_adjacency_offsets is not None:
            assert points_adjacency_offsets.shape[0] == points.shape[0] + 1, (
                "points_adjacency_offsets must have the same number of points + 1"
            )

            self.primal_points = nn.Parameter(points.to(self.device))

            self.point_adjacency = points_adjacency.to(self.device, dtype=torch.uint32)
            self.point_adjacency_offsets = points_adjacency_offsets.to(
                self.device, dtype=torch.uint32
            )

            self.bvh.build(self.primal_points)
        else:
            points = points.to(self.device)
            self.primal_points = nn.Parameter(points)
            self.update_triangulation()

    @classmethod
    def from_random(
        cls,
        config: VoroTracingConfig,
        num_points: int,
        scale: float = 25,
        device: torch.device = torch.device("cuda"),
    ) -> "VoroTracing":
        primal_points = torch.randn(num_points, 3, device=device) * scale
        primal_points = primal_points.to(device)

        model = cls(config=config, points=primal_points, device=device)
        return model

    @classmethod
    def from_pointcloud(
        cls,
        config: VoroTracingConfig,
        points: torch.Tensor,
        points_colors: torch.Tensor,
        num_random: int = 5_000,
        max_points: Optional[int] = None,
        subsample_density_alpha: float = 1.0,
        device: torch.device = torch.device("cuda"),
    ) -> "VoroTracing":
        assert points is not None, "points must be provided"
        attr_dtype = torch.float32
        points = points.to(device)
        points_colors = points_colors.to(device)

        points_mean = points.mean(dim=0, keepdim=True).to(device)
        points_std = points.std(dim=0, keepdim=True).to(device)

        random = (
            torch.randn([num_random, 3], device=device) * points_std * 3 + points_mean
        ).to(device)

        if max_points is None:
            num_samples = int(0.9 * points.shape[0])
        else:
            num_samples = max_points - num_random
            if num_samples <= 0:
                raise ValueError(
                    "max_points must be larger than num_random when initializing "
                    f"from a point cloud, got max_points={max_points}, "
                    f"num_random={num_random}"
                )
            num_samples = min(num_samples, points.shape[0])

        print(
            f"Starting with {num_samples} points from {points.shape[0]} point cloud points"
        )
        if num_samples < points.shape[0]:
            if subsample_density_alpha > 0:
                grid_res = 128
                bbox_min = points.min(dim=0).values
                bbox_max = points.max(dim=0).values
                bbox_size = (bbox_max - bbox_min).clamp(min=1e-6)
                voxel_size = bbox_size / grid_res
                grid_coords = (
                    ((points - bbox_min) / voxel_size).long().clamp(0, grid_res - 1)
                )
                linear_idx = (
                    grid_coords[:, 0] * grid_res * grid_res
                    + grid_coords[:, 1] * grid_res
                    + grid_coords[:, 2]
                )
                counts = torch.bincount(linear_idx, minlength=grid_res**3)
                weights = (1.0 / counts[linear_idx].float()) ** subsample_density_alpha
                points_idx = torch.multinomial(weights, num_samples, replacement=False)
                print(
                    f"Density-aware subsampling (alpha={subsample_density_alpha}): "
                    f"{grid_res}^3 voxel grid, "
                    f"{counts.nonzero().shape[0]} occupied voxels"
                )
            else:
                points_idx = torch.randperm(points.shape[0], device=device)[
                    :num_samples
                ]
        else:
            points_idx = torch.arange(points.shape[0], device=device)
        samp_points = points[points_idx]
        samp_points += torch.randn_like(samp_points) * 1e-2

        primal_points = torch.cat([samp_points, random], dim=0)
        primal_density = torch.cat(
            [
                torch.log(
                    F.softplus(
                        torch.rand(samp_points.shape[0], 1, dtype=attr_dtype), beta=10
                    )
                ),
                # clamp_min keeps it finite preventing crashes
                torch.log(
                    F.softplus(
                        -25 * torch.ones(num_random, 1, dtype=attr_dtype), beta=10
                    ).clamp_min(1e-30)
                ),
            ],
            dim=0,
        ).to(device)

        # Initialize the diffuse octmap from the per-point colors: store
        # logit(color) so that sigmoid in the CUDA shader recovers `color`.
        # Random-extension points stay at logit(0.5) = 0 (neutral gray).
        R = config.oct_map_res
        eps = 1e-3
        samp_colors = points_colors[points_idx].clamp(eps, 1.0 - eps)
        logit_colors = torch.log(samp_colors / (1.0 - samp_colors)).to(attr_dtype)
        samp_diffuse = (
            logit_colors[:, None, :]
            .expand(-1, R * R, -1)
            .reshape(samp_colors.shape[0], 3 * R * R)
        )
        random_diffuse = torch.zeros(
            num_random, 3 * R * R, dtype=attr_dtype, device=device
        )
        primal_diffuse = torch.cat([samp_diffuse, random_diffuse], dim=0)

        model = cls(
            config=config,
            points=primal_points,
            points_diffuse=primal_diffuse,
            points_density=primal_density,
            device=device,
        )
        return model

    @classmethod
    def from_pretrained(
        cls,
        pt_path: Union[str, Path],
        config: Union[VoroTracingConfig, str],
        device: torch.device = torch.device("cuda"),
    ) -> "VoroTracing":
        if isinstance(config, str):
            raise NotImplementedError("Loading config from string is not implemented")
            config = VoroTracingConfig.from_yaml(config)  # TODO: Implement this

        scene_data = torch.load(pt_path)

        points = scene_data["xyz"]
        points_density = scene_data["density"]
        points_adjacency = scene_data["adjacency"]
        points_adjacency_offsets = scene_data["adjacency_offsets"]

        model = cls(
            config=config,
            points=points,
            points_diffuse=scene_data["color_diffuse"],
            points_specular=scene_data["color_specular"],
            points_density=points_density,
            points_adjacency=points_adjacency,
            points_adjacency_offsets=points_adjacency_offsets,
            device=device,
        )

        return model

    def set_optimizer(self, optimizer: torch.optim.Optimizer):
        self.optimizer = optimizer

    def _permute_points(self, permutation):
        """Permute the points and the associated attributes in the optimizer.
        This is needed to keep the optimizer in a valid state (e.g. momentum of each parameter) after a permutation.
        """
        assert self.optimizer is not None, (
            "Optimizer must be assigned with set_optimizer() first"
        )

        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if "env" not in group["name"]:
                stored_state = self.optimizer.state.get(group["params"][0], None)
                if stored_state is not None:
                    stored_state["exp_avg"] = stored_state["exp_avg"][permutation]
                    stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][permutation]

                    del self.optimizer.state[group["params"][0]]
                    group["params"][0] = nn.Parameter(
                        (group["params"][0][permutation].requires_grad_(True))
                    )
                    self.optimizer.state[group["params"][0]] = stored_state

                    optimizable_tensors[group["name"]] = group["params"][0]
                else:
                    group["params"][0] = nn.Parameter(
                        group["params"][0][permutation].requires_grad_(True)
                    )
                    optimizable_tensors[group["name"]] = group["params"][0]

        self.primal_points = optimizable_tensors["primal_points"]
        self.density = optimizable_tensors["density"]
        self.att_diffuse = optimizable_tensors["att_diffuse"]
        self.att_specular = optimizable_tensors["att_specular"]

    def update_triangulation(self):
        """Update the triangulation with new point positions."""
        if not self.primal_points.isfinite().all():
            raise RuntimeError("NaN in points")

        adjacency, offsets, _ = ops.build_voronoi_adjacency(
            self.primal_points.detach(), debug=False
        )
        self.point_adjacency = adjacency
        self.point_adjacency_offsets = offsets

        self.bvh.build(self.primal_points)

    def get_starting_point(self, rays):
        """Given rays shaped [..., 6],
        return nearest-neighbour start indices with the same leading shape.
        """
        with torch.no_grad():
            if rays.ndim == 1:
                rays = rays[None, :]
            camera_origins = rays[..., :3]

            # Flatten all leading dimensions
            leading_shape = camera_origins.shape[:-1]
            flat_origins = camera_origins.reshape(-1, 3)

            unique_cameras, inverse_indices = torch.unique(
                flat_origins, dim=0, return_inverse=True
            )

            nn_inds = self.bvh.query(self.primal_points, unique_cameras).long()

            start_point_flat = nn_inds[inverse_indices]
            start_point = start_point_flat.view(*leading_shape)
            return start_point.type(torch.uint32)

    def _get_primal_density(self):
        return self.config.activation_scale * torch.exp(self.density)

    def _get_trace_data(self):
        """Get the trace data for rendering.

        Returns the per-cell attributes as three separate tensors so the kernel
        can read each without paying for a Python-side `torch.cat`
        materialization (~7 ms on a 2M-point scene with 1M rays).
        """
        return (
            self.primal_points,
            self.att_diffuse,
            self.att_specular,
            self._get_primal_density(),
            self.point_adjacency,
            self.point_adjacency_offsets,
        )

    def forward(
        self,
        rays: torch.Tensor,
        start_point: Optional[torch.Tensor] = None,
        depth_quantiles: Optional[torch.Tensor] = None,
        weight_threshold: float = 0.001,
        max_intersections: int = 1024,
        return_contribution: bool = False,
    ):
        """Render a batch of rays.
        The rays shape will define the output shape. The output shape is the same as the rays shape, except for the last dimension.

        Args:
            rays: [..., 6] (3 origins + 3 directions)
            start_point: [...,] optional
            depth_quantiles: [..., K] optional transmittance thresholds for depth output
            weight_threshold: float optional
            max_intersections: int optional
            return_contribution: bool optional
        Returns:
            rgba: [..., 4]
            depth: [..., K] if depth_quantiles was provided, otherwise an empty tensor
            contribution: [..., 1]
            num_intersections: [..., 1]
            errbox: ErrorBox
            distortion: [..., 1] mip-NeRF 360 per-ray distortion loss
        """
        (
            points,
            diffuse,
            specular,
            density,
            point_adjacency,
            point_adjacency_offsets,
        ) = self._get_trace_data()

        if start_point is None:
            start_point = self.get_starting_point(rays)
        else:
            start_point = torch.broadcast_to(start_point, rays.shape[:-1])

        return ops.TraceVoroTracing.apply(
            points,
            diffuse,
            specular,
            density,
            point_adjacency,
            point_adjacency_offsets,
            rays,
            start_point,
            depth_quantiles,
            weight_threshold,
            max_intersections,
            return_contribution,
        )

    def save_checkpoint(self, out_path: str):
        points = self.primal_points.detach().float().cpu()
        density = self.density.detach().float().cpu()
        adjacency = self.point_adjacency.cpu()
        adjacency_offsets = self.point_adjacency_offsets.cpu()

        checkpoint = {
            "xyz": points,
            "density": density,
            "adjacency": adjacency.long(),
            "adjacency_offsets": adjacency_offsets.long(),
            "color_diffuse": self.att_diffuse.detach().float().cpu(),
            "color_specular": self.att_specular.detach().float().cpu(),
        }

        torch.save(checkpoint, out_path)


class VoroTracingInfer:
    """Lightweight inference wrapper. Stores fp16 attributes for fast rendering."""

    def __init__(
        self,
        model: VoroTracing,
        sort_morton: bool = True,
        quantize: Literal["fp16", "q8"] = "fp16",
    ):
        """
        Args:
            sort_morton: bool, whether to sort points by Morton code for better cache locality during traversal.
            quantize: "fp16" for half-precision, "q8" for int8 quantized logits.
        """
        self.primal_points = model.primal_points.detach()
        self.density = (
            model.config.activation_scale * torch.exp(model.density)
        ).detach()
        self.point_adjacency = model.point_adjacency.clone()
        self.point_adjacency_offsets = model.point_adjacency_offsets
        self.bvh = model.bvh
        self.device = model.device
        self.quantize = quantize

        if quantize == "q8":
            diffuse_fp32 = model.att_diffuse.detach()
            specular_fp32 = model.att_specular.detach()
            self.diffuse_q8, self.diff_scale, self.diff_offset = (
                self._quantize_to_uint8(diffuse_fp32)
            )
            self.specular_q8, self.spec_scale, self.spec_offset = (
                self._quantize_to_uint8(specular_fp32)
            )
            self.diffuse_fp16 = None
            self.specular_fp16 = None
        elif quantize == "fp16":
            R_diff = model.config.oct_map_res
            R_spec = model.config.spec_oct_map_res
            self.diffuse_fp16 = self._pad_3to4(
                model.att_diffuse.detach().half(), R_diff
            )
            self.specular_fp16 = self._pad_3to4(
                model.att_specular.detach().half(), R_spec
            )
        else:
            raise ValueError(f"Invalid quantization method: {quantize}")

        if sort_morton:
            self._reorder_by_morton()

        self.adjacent_diff = ops.prefetch_octmap_adj(
            self.primal_points,
            self.point_adjacency,
            self.point_adjacency_offsets,
        )

    @staticmethod
    def _quantize_to_uint8(tensor):
        """Uniform quantization to uint8. Returns (quantized, scale, offset)."""
        vmin = tensor.min().item()
        vmax = tensor.max().item()
        scale = (vmax - vmin) / 255.0
        offset = vmin
        quantized = ((tensor - offset) / scale).round().clamp(0, 255).to(torch.uint8)
        return quantized, scale, offset

    @staticmethod
    def _pad_3to4(tensor, R):
        # (N, R*R*3) -> (N, R*R*4) so the inference kernel can issue a single
        # 8-byte load per texel (LDG.E.64) instead of 3 small ones.
        n = tensor.shape[0]
        t = tensor.view(n, R * R, 3)
        pad = torch.zeros(n, R * R, 1, dtype=t.dtype, device=t.device)
        return torch.cat([t, pad], dim=-1).contiguous().view(n, R * R * 4)

    def _reorder_by_morton(self):
        """Reorder points by Morton code for better cache locality during traversal."""
        pts = self.primal_points
        n = pts.shape[0]

        bbox_min = pts.min(dim=0).values
        bbox_max = pts.max(dim=0).values
        bbox_size = (bbox_max - bbox_min).clamp(min=1e-6)

        norm = ((pts - bbox_min) / bbox_size * 1023.0).clamp(0, 1023).long()
        x, y, z = norm[:, 0], norm[:, 1], norm[:, 2]

        morton = torch.zeros(n, dtype=torch.long, device=pts.device)
        for i in range(10):
            morton |= ((x >> i) & 1) << (3 * i)
            morton |= ((y >> i) & 1) << (3 * i + 1)
            morton |= ((z >> i) & 1) << (3 * i + 2)

        perm = morton.argsort()
        inv_perm = torch.empty_like(perm)
        inv_perm[perm] = torch.arange(n, device=pts.device)

        self.primal_points = self.primal_points[perm].contiguous()
        self.density = self.density[perm].contiguous()
        if self.quantize == "q8":
            self.diffuse_q8 = self.diffuse_q8[perm].contiguous()
            self.specular_q8 = self.specular_q8[perm].contiguous()
        else:
            self.diffuse_fp16 = self.diffuse_fp16[perm].contiguous()
            self.specular_fp16 = self.specular_fp16[perm].contiguous()

        offsets = self.point_adjacency_offsets.long()
        adj = self.point_adjacency.long()
        adj_data_len = offsets[-1].item()
        adj[:adj_data_len] = inv_perm[adj[:adj_data_len]]

        new_offsets = torch.zeros_like(offsets)
        counts = offsets[1:] - offsets[:-1]
        new_counts = counts[perm]
        new_offsets[1:] = new_counts.cumsum(0)

        new_adj = torch.empty_like(adj)
        old_starts = offsets[:-1]
        new_starts = new_offsets[:-1]
        src_starts = old_starts[perm]
        dst_starts = new_starts

        # Batch copy adjacency lists
        max_deg = counts.max().item()
        for d in range(max_deg):
            mask = d < new_counts
            src_idx = src_starts[mask] + d
            dst_idx = dst_starts[mask] + d
            new_adj[dst_idx] = adj[src_idx]

        self.point_adjacency = new_adj.to(torch.uint32)
        self.point_adjacency_offsets = new_offsets.to(torch.uint32)
        self.bvh.build(self.primal_points)

    @classmethod
    def from_pretrained(
        cls,
        pt_path,
        config,
        device=torch.device("cuda"),
        quantize="fp16",
        sort_morton: bool = True,
    ):
        model = VoroTracing.from_pretrained(pt_path, config, device=device)
        return cls(model, quantize=quantize, sort_morton=sort_morton)

    def get_starting_point(self, rays):
        """Find the BVH cell index used as the per-ray traversal starting point.

        WARNING: Assumes all rays in the batch share the same camera origin (the typical
        pinhole / single-view case): runs a single BVH query and broadcasts
        the result. Skipping the torch.unique-based dedup that the multi-origin
        path used saves ~0.7 ms per frame at 1.6 M rays.
        """
        with torch.no_grad():
            if rays.ndim == 1:
                rays = rays[None, :]
            leading_shape = rays.shape[:-1]
            origin = rays.reshape(-1, rays.shape[-1])[:1, :3]  # (1, 3)
            nn_idx = self.bvh.query(self.primal_points, origin)  # (1,) uint32
            n = 1
            for s in leading_shape:
                n *= s
            return (
                nn_idx.expand(n).reshape(leading_shape).contiguous().type(torch.uint32)
            )

    @torch.no_grad()
    def _tile_perm(self, H, W, TH=4, TW=8):
        """uint32 perm (cached per resolution) mapping warp/thread order -> pixel
        index, so a 32-thread warp covers a compact THxTW block (warp coherence).
        Consumed kernel-side as ray_id = perm[thread_idx] (thread t handles pixel
        perm[t]); output written to perm[t] stays image-ordered."""
        cache = getattr(self, "_tile_perm_cache", None)
        if cache is None:
            cache = self._tile_perm_cache = {}
        key = (int(H), int(W))
        if key not in cache:
            dev = self.primal_points.device
            y = torch.arange(H, device=dev).view(H, 1).expand(H, W)
            x = torch.arange(W, device=dev).view(1, W).expand(H, W)
            tpr = (W + TW - 1) // TW
            within = (y % TH) * TW + (x % TW)
            k = ((y // TH) * tpr + (x // TW)).long() * (TH * TW) + within.long()
            perm = torch.argsort(k.reshape(-1)).to(torch.int32).contiguous()
            cache[key] = perm
        return cache[key]

    def render(
        self,
        rays,
        start_point=None,
        weight_threshold=0.001,
        max_intersections=1024,
        cell_skip_threshold=0.0,
        use_warp_perm: bool = True,
    ):
        # Warp-coherence tiling: for a full (H,W,6) image render, pass a tile-order
        # permutation so a 32-thread warp covers a compact 8x4 block. Applied
        # kernel-side (thread t handles ray perm[t]); rays/output stay in image
        # order, so there is NO gather/scatter. Other ray shapes -> identity.
        ray_perm = torch.empty(0, dtype=torch.int32, device=self.primal_points.device)
        if use_warp_perm and rays.ndim == 3 and rays.shape[-1] == 6:
            ray_perm = self._tile_perm(int(rays.shape[0]), int(rays.shape[1]))

        if start_point is None:
            start_point = self.get_starting_point(rays)
        else:
            start_point = torch.broadcast_to(start_point, rays.shape[:-1])
        if self.quantize == "q8":
            rgba_fp16 = ops.trace_octmap_infer_q8(
                self.primal_points,
                self.diffuse_q8,
                self.specular_q8,
                self.density,
                self.point_adjacency,
                self.point_adjacency_offsets,
                rays,
                start_point,
                self.adjacent_diff,
                self.diff_scale,
                self.diff_offset,
                self.spec_scale,
                self.spec_offset,
                weight_threshold,
                max_intersections,
                cell_skip_threshold,
            )
        else:
            rgba_fp16 = ops.trace_octmap_infer(
                self.primal_points,
                self.diffuse_fp16,
                self.specular_fp16,
                self.density,
                self.point_adjacency,
                self.point_adjacency_offsets,
                rays,
                start_point,
                self.adjacent_diff,
                ray_perm,
                weight_threshold,
                max_intersections,
                cell_skip_threshold,
            )
        return rgba_fp16
