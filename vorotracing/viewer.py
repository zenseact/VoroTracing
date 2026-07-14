from typing import Literal, List, Optional, Tuple
import matplotlib.pyplot as plt
import nerfview
import torch
import viser
import numpy as np
from scipy.spatial.transform import Rotation

from vorotracing.utils import generate_camera_rays
from PIL import Image
import datetime
import os


class VoroTracingViewer(nerfview.Viewer):
    def __init__(
        self,
        model: torch.nn.Module,
        port: int = 7007,
        verbose: bool = False,
        mode: Literal["rendering", "training"] = "rendering",
        up_direction: Optional[List[float]] = None,
    ):
        server = viser.ViserServer(port=port, verbose=verbose)
        self.model = model

        super().__init__(server=server, render_fn=self._viewer_render_fn, mode=mode)

        self._setup_custom_panels()

        if up_direction is not None:
            server.scene.set_up_direction(up_direction)

    def _setup_custom_panels(self):
        # RENDER MODE
        with self.server.gui.add_folder("Render Output"):
            self._render_mode_dropdown = self.server.gui.add_dropdown(
                "Mode",
                options=["RGB", "Mean Depth", "Depth at Threshold"],
                initial_value="RGB",
            )
            self._mean_depth_samples = self.server.gui.add_slider(
                "Mean depth samples (N)", min=4, max=128, step=4, initial_value=32
            )
            self._depth_threshold = self.server.gui.add_slider(
                "Transmittance threshold (T)",
                min=0.01,
                max=0.99,
                step=0.01,
                initial_value=0.5,
            )
            self._depth_colormap_dropdown = self.server.gui.add_dropdown(
                "Depth colormap",
                options=["viridis", "inferno", "plasma", "magma", "turbo"],
                initial_value="turbo",
            )

        # PRIMAL POINTS POINTCLOUD
        with self.server.gui.add_folder("Primal Points"):
            self._show_primal_points_checkbox = self.server.gui.add_checkbox(
                "Show primal points", initial_value=False
            )
            self._show_primal_points_checkbox.on_update(
                self._show_primal_points_callback
            )
            self._primal_points_handle = None
            self._color_primal_points_dropdown = self.server.gui.add_dropdown(
                "Color primal points by",
                options=["height", "density", "weight"],
                initial_value="density",
            )
            self._primal_points_clip_range_checkbox = self.server.gui.add_checkbox(
                "Clip primal points color range", initial_value=False
            )
            self._primal_points_clip_range_min = self.server.gui.add_number(
                "Primal points color range min", initial_value=0.0, step=0.001
            )
            self._primal_points_clip_range_max = self.server.gui.add_number(
                "Primal points color range max", initial_value=1.0, step=0.001
            )

        # VORONOI INSPECTION
        with self.server.gui.add_folder("VoroTracing Inspection"):
            self._vorotracing_inspection_checkbox = self.server.gui.add_checkbox(
                "VoroTracing inspection", initial_value=False
            )
            self._vorotracing_inspection_checkbox.on_update(
                self._show_vorotracing_inspection_callback
            )
            self._vorotracing_image_handle = None
            self._vorotracing_gizmo_handle = None
            self._vorotracing_image_size = (8.0, 4.5)  # Default 16:9
            self._vorotracing_image_rendering = False
            self._vorotracing_image_scale_slider = self.server.gui.add_slider(
                "VoroTracing image scale", min=0.1, max=10.0, step=0.1, initial_value=1.0
            )

            @self._vorotracing_image_scale_slider.on_update
            def _(_) -> None:
                if not self._vorotracing_image_rendering:
                    self._vorotracing_image_rendering = True
                    self._update_vorotracing_image()
                    self._vorotracing_image_rendering = False

            self._vorotracing_save_image_button = self.server.gui.add_button(
                "Save high-res image"
            )
            self._vorotracing_save_image_button.on_click(self._save_vorotracing_image_callback)
            self._vorotracing_color_metric_dropdown = self.server.gui.add_dropdown(
                "VoroTracing color metric",
                options=["density", "avg_neighbor_dist", "density_dist_mix"],
                initial_value="density",
            )
            self._vorotracing_colormap_dropdown = self.server.gui.add_dropdown(
                "Colormap",
                options=["viridis", "inferno", "plasma", "magma", "cividis"],
                initial_value="inferno",
            )
            self._vorotracing_invert_colormap_checkbox = self.server.gui.add_checkbox(
                "Invert colormap", initial_value=False
            )
            self._vorotracing_color_gamma = self.server.gui.add_number(
                "Color gamma", initial_value=1.0, step=0.1
            )

    def _show_vorotracing_inspection_callback(self, event):
        if event.target.value:  # Checkbox is checked
            if self._vorotracing_gizmo_handle is None:
                self._vorotracing_gizmo_handle = self.server.scene.add_transform_controls(
                    name="vorotracing_image_gizmo",
                    disable_sliders=True,
                    position=np.array([0, 0, 0]),
                    wxyz=np.array([1, 0, 0, 0]),
                )

                @self._vorotracing_gizmo_handle.on_update
                def _(_) -> None:
                    if not self._vorotracing_image_rendering:
                        self._vorotracing_image_rendering = True
                        self._update_vorotracing_image()
                        self._vorotracing_image_rendering = False
                    else:
                        return

                self._update_vorotracing_image()
            else:
                if self._vorotracing_image_handle is not None:
                    self._vorotracing_image_handle.remove()
                    self._vorotracing_image_handle = None
                self._vorotracing_gizmo_handle.remove()
                self._vorotracing_gizmo_handle = None

    @nerfview.with_viewer_lock
    def _update_vorotracing_image(self, resolution: Optional[Tuple[int, int]] = None):
        if resolution is None:
            width, height = 1000, 1000
        else:
            width, height = resolution

        position = self._vorotracing_gizmo_handle.position
        orientation = self._vorotracing_gizmo_handle.wxyz  # wxyz quaternion
        scale = self._vorotracing_image_scale_slider.value
        size = (self._vorotracing_image_size[0] * scale, self._vorotracing_image_size[1] * scale)

        # Compute the 3D position of each pixel in the image
        u, v = np.meshgrid(
            np.linspace(-0.5, 0.5, width), np.linspace(-0.5, 0.5, height)
        )
        u = u.flatten()
        v = v.flatten()
        pixels = np.stack([u * size[0], v * size[1], np.zeros_like(u)], axis=-1)

        rot = Rotation.from_quat(
            [orientation[1], orientation[2], orientation[3], orientation[0]]
        )  # scipy uses xyzw
        pixels = rot.apply(pixels)
        pixels = pixels + position
        pixels = pixels.reshape(height, width, 3)

        # probe aabb tree for each position voronoi index
        pixels_flat = pixels.reshape(-1, 3)
        pixels_tensor = torch.tensor(
            pixels_flat, dtype=torch.float32, device=self.model.device
        )
        voronoi_indices = self.model.get_starting_point(pixels_tensor)

        # Generate colors based on density + random component
        indices = voronoi_indices.to(torch.int64)

        with torch.no_grad():
            if (
                self._vorotracing_color_metric_dropdown.value == "density"
                or self._vorotracing_color_metric_dropdown.value == "density_dist_mix"
            ):
                densities = self.model._get_primal_density().squeeze(-1)

            if (
                self._vorotracing_color_metric_dropdown.value == "avg_neighbor_dist"
                or self._vorotracing_color_metric_dropdown.value == "density_dist_mix"
            ):
                points = self.model.primal_points
                # _get_trace_data returns points, attributes, point_adjacency, point_adjacency_offsets
                _, _, adj, offsets = self.model._get_trace_data()
                adj = adj.to(torch.int64).squeeze()
                offsets = offsets.to(torch.int64).squeeze()

                # adj can be padded, we must use offsets
                counts = (offsets[1:] - offsets[:-1]).clamp(min=1)
                src_indices = torch.repeat_interleave(
                    torch.arange(len(points), device=points.device), counts
                )

                # Filter out valid adjacency indices
                valid_adj = adj[: offsets[-1]]

                dist = torch.norm(points[src_indices] - points[valid_adj], dim=-1)

                # Average distance per source point
                avg_dist = torch.zeros(len(points), device=points.device)
                avg_dist.index_add_(0, src_indices, dist)
                avg_dist = avg_dist / counts
                neighbor_dist = avg_dist

            if self._vorotracing_color_metric_dropdown.value == "density":
                metric_values = densities
            elif self._vorotracing_color_metric_dropdown.value == "avg_neighbor_dist":
                metric_values = neighbor_dist
            elif self._vorotracing_color_metric_dropdown.value == "density_dist_mix":
                # Localize both before mixing to ensure they are in a similar range
                unique_indices = torch.unique(indices)
                v_dens = densities[unique_indices]
                v_dist = neighbor_dist[unique_indices]

                # Local normalization helper
                def loc_norm(v, ref):
                    rmin, rmax = ref.min(), ref.max()
                    return (v - rmin) / (rmax - rmin + 1e-8)

                metric_values = loc_norm(densities, v_dens) * loc_norm(
                    neighbor_dist, v_dist
                )
            else:
                raise ValueError(
                    f"Invalid metric: {self._vorotracing_color_metric_dropdown.value}"
                )

            # Normalize metric values based on visible cells
            if self._vorotracing_color_metric_dropdown.value != "density_dist_mix":
                unique_indices = torch.unique(indices)
                visible_metrics = metric_values[unique_indices]

                m_min = visible_metrics.min()
                m_max = visible_metrics.max()
                eps = 1e-8
                normalized_metric = (metric_values - m_min) / (m_max - m_min + eps)
            else:
                # Already mixed with local normalization
                normalized_metric = metric_values

            # Apply gamma correction
            gamma = self._vorotracing_color_gamma.value
            normalized_metric = normalized_metric.pow(gamma)

            normalized_metric = normalized_metric.clamp(0, 1)

            # Apply colormap (CPU)
            norm_metric_np = normalized_metric.cpu().numpy()
            if self._vorotracing_invert_colormap_checkbox.value:
                norm_metric_np = 1.0 - norm_metric_np

            # Use selected colormap
            cmap = getattr(plt.cm, self._vorotracing_colormap_dropdown.value)
            density_colors_np = cmap(norm_metric_np)[:, :3]
            density_colors = torch.from_numpy(density_colors_np).to(
                device=self.model.device, dtype=torch.float32
            )

            # Calculate random colors for all points
            point_indices = torch.arange(
                metric_values.shape[0], device=self.model.device, dtype=torch.int64
            )
            r = ((point_indices * 2654435761) % 256) / 255.0
            g = ((point_indices * 1597334677) % 256) / 255.0
            b = ((point_indices * 3812015801) % 256) / 255.0
            random_colors = torch.stack([r, g, b], dim=-1)

            # Mix colors: 90% metric, 10% random
            mixed_colors = 0.9 * density_colors + 0.1 * random_colors

            # Index into colors for each pixel
            colors = mixed_colors[indices]

        # Transfer to CPU only at the end
        colors_np = colors.cpu().numpy()
        rgb = colors_np.reshape(height, width, 3)
        if self._vorotracing_image_handle is None:
            self._vorotracing_image_handle = self.server.scene.add_image(
                name="vorotracing_image",
                image=rgb,
                render_width=size[0],
                render_height=size[1],
                position=position,
                wxyz=orientation,
            )
        else:
            self._vorotracing_image_handle.image = rgb
            self._vorotracing_image_handle.position = position
            self._vorotracing_image_handle.wxyz = orientation
            self._vorotracing_image_handle.render_width = size[0]
            self._vorotracing_image_handle.render_height = size[1]

        return rgb

    def _save_vorotracing_image_callback(self, _):
        print("Rendering high-res vorotracing image...")
        # Use a higher resolution for saving (16:9)
        hi_res = (2048, 1152)
        rgb = self._update_vorotracing_image(resolution=hi_res)

        # Create output directory
        out_dir = "vorotracing_images"
        os.makedirs(out_dir, exist_ok=True)

        # Save image
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"vorotracing_{timestamp}.png"
        filepath = os.path.join(out_dir, filename)

        img = Image.fromarray((rgb * 255).astype(np.uint8))
        img.save(filepath)
        print(f"High-res vorotracing image saved to {filepath}")

    def _show_primal_points_callback(self, event):
        if self._primal_points_handle is not None:
            self._primal_points_handle.remove()
            self._primal_points_handle = None

        if event.target.value:
            if self._color_primal_points_dropdown.value == "height":
                color_data = self.model.primal_points.detach().cpu().numpy()[:, 2]
            elif self._color_primal_points_dropdown.value == "density":
                color_data = (
                    self.model._get_primal_density().detach().cpu().squeeze(-1).numpy()
                )
            elif self._color_primal_points_dropdown.value == "weight":
                color_data = (
                    self.model._get_primal_weight().detach().cpu().squeeze(-1).numpy()
                )
            else:
                raise ValueError(
                    f"Invalid color option: {self._color_primal_points_dropdown.value}"
                )

            if self._primal_points_clip_range_checkbox.value:
                color_data = np.clip(
                    color_data,
                    self._primal_points_clip_range_min.value,
                    self._primal_points_clip_range_max.value,
                )

            color_data_norm = (color_data - color_data.min()) / (
                color_data.max() - color_data.min()
            )
            colors = plt.cm.inferno(color_data_norm)[:, :3]

            self._primal_points_handle = self.server.scene.add_point_cloud(
                name="primal_points",
                points=self.model.primal_points.detach().cpu().numpy(),
                colors=colors,
                point_size=0.01,
                point_shape="circle",
            )

    def _depth_to_rgb(self, depth_map: np.ndarray) -> np.ndarray:
        """Convert a raw depth map (t-values, -1 = invalid) to a colormapped RGB image."""
        cmap = getattr(plt.cm, self._depth_colormap_dropdown.value)
        valid = depth_map > 0
        if valid.any():
            valid_depths = depth_map[valid]
            lo = np.percentile(valid_depths, 2)
            hi = np.percentile(valid_depths, 98)
            normalized = np.zeros_like(depth_map)
            normalized[valid] = np.clip((valid_depths - lo) / (hi - lo + 1e-8), 0, 1)
        else:
            normalized = np.zeros_like(depth_map)
        rgb = cmap(normalized)[..., :3]
        rgb[~valid] = 0.0
        return rgb.astype(np.float32)

    @torch.no_grad()
    def _viewer_render_fn(
        self,
        camera_state: nerfview.CameraState,
        render_tab_state: nerfview.RenderTabState,
    ) -> Tuple[np.ndarray, np.ndarray]:
        device = self.model.device
        if render_tab_state.preview_render:
            width = render_tab_state.render_width
            height = render_tab_state.render_height
        else:
            width = render_tab_state.viewer_width
            height = render_tab_state.viewer_height
        K = camera_state.get_K((width, height))
        K = torch.tensor(K, device=device)
        T_c2w = torch.tensor(camera_state.c2w, device=device)
        rays = generate_camera_rays(K, T_c2w, width, height, device)
        rays = rays.to(dtype=torch.float32, device=device)

        mode = self._render_mode_dropdown.value

        if mode == "RGB":
            rgba_output, _, _, _, _, _ = self.model(rays)
            rgb = rgba_output[..., :3].cpu().detach().numpy().reshape(height, width, 3)
            return rgb

        if mode == "Mean Depth":
            n = int(self._mean_depth_samples.value)
            thresholds = torch.linspace(1 - 1 / (2 * n), 1 / (2 * n), n, device=device)
            thresholds = thresholds.expand(*rays.shape[:-1], n).contiguous()
            _, depth, _, _, _, _ = self.model(rays, depth_quantiles=thresholds)
            valid = depth > 0
            mean_depth = (depth * valid).sum(-1) / valid.sum(-1).clamp(min=1)
            mean_depth[valid.sum(-1) == 0] = -1.0
            depth_np = mean_depth.cpu().numpy().reshape(height, width)
            return self._depth_to_rgb(depth_np)

        threshold = float(self._depth_threshold.value)
        thresholds = torch.full(
            (*rays.shape[:-1], 1), threshold, device=device, dtype=torch.float32
        )
        _, depth, _, _, _, _ = self.model(rays, depth_quantiles=thresholds)
        depth_np = depth[..., 0].cpu().numpy().reshape(height, width)
        return self._depth_to_rgb(depth_np)
