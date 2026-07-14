import os
import time
from dataclasses import dataclass, field
from typing import Type

import numpy as np
import torch
import tqdm
import tyro
from PIL import Image
from pytorch_msssim import SSIM
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

import wandb
from vorotracing.config.base import InstantiateConfig
from vorotracing.datasets.datasets import DatasetConfig
from vorotracing.datasets.ray_batcher import RayBatcher
from vorotracing.vorotracing import VoroTracing, VoroTracingConfig
from vorotracing.utils import get_cosine_lr_func, psnr
from vorotracing.viewer import VoroTracingViewer


@dataclass
class VoroOptimizerConfig:
    points_lr_init: float = 2e-4
    points_lr_final: float = 5e-6
    density_lr_init: float = 1e-1
    density_lr_final: float = 1e-2
    attributes_lr_init: float = 5e-3
    attributes_lr_final: float = 5e-4
    freeze_points: int = 18_000
    specular_start: int = 4_000
    fused_adam: bool = True


@dataclass
class VoroTrainerConfig(InstantiateConfig):
    _target: Type = field(default_factory=lambda: VoroTracingTrainer)
    model_config: VoroTracingConfig = field(default_factory=VoroTracingConfig)
    dataset_config: DatasetConfig = field(default_factory=DatasetConfig)
    optimizer_config: VoroOptimizerConfig = field(default_factory=VoroOptimizerConfig)

    iterations: int = 20_000
    rays_per_batch: int = 1_000_000
    num_init_points: int = 131_072
    num_random_init_points: int = 5_000
    subsample_density_alpha: float = 1.0
    val_interval: int = 2500
    train_log_interval: int = 100

    # Patch-based training: sample warp-coherent PxP pixel patches instead of random
    patch_training: bool = False
    patch_size: int = 16
    # 3DGS-style SSIM term on patches: rgb_term = (1-w)*L1 + w*(1-SSIM). Requires patch_training
    ssim_weight: float = 0.0

    iter2downsample: dict[int, float] = field(default_factory=lambda: {0: 8, 5_000: 4})
    density_warmup_steps: int = 2_000
    white_background: bool = True
    quantile_weight: float = 1e-4
    contribution_weight: float = 0.0
    distortion_weight: float = 0.0

    # Per-cell adaptive distortion (Adaptive Shells analog). Modulates the global
    # distortion (binarization) pressure per cell by photometric confidence: confident
    # surface cells binarize fully while fuzzy/high-frequency cells (foliage, grass)
    # are spared and keep soft opacity for sub-cell detail.
    distortion_percell: bool = False
    distortion_percell_beta: float = 1.0  # larger -> gentler suppression
    distortion_percell_decay: float = 0.95  # EMA decay for per-cell residual
    specular_reg_weight: float = 0.0
    diffuse_tv_weight: float = 0.0
    diffuse_mean_pull_weight: float = 0.0

    out_dir: str = "output"
    wandb_project: str = "VoroTracing"
    experiment_name: str = "vorotracing-train"
    exit_on_completion: bool = False
    wandb: bool = False
    viewer: bool = True
    save_checkpoint: bool = True
    export_val: bool = False


class VoroTracingTrainer:
    config: VoroTrainerConfig

    def __init__(self, config: VoroTrainerConfig):
        self.config = config
        self.device = torch.device("cuda")

        self.out_dir = f"{self.config.out_dir}/{self.config.experiment_name}_{time.strftime('%Y%m%d_%H%M%S')}"
        if self.config.wandb:
            wandb.init(
                project=self.config.wandb_project,
                name=self.config.experiment_name,
                config=self.config,
            )

        # Setting up dataset
        self.train_dataset = config.dataset_config.load(
            split="train", downsample_factor=self.config.iter2downsample[0]
        )
        self.ray_batcher = RayBatcher(
            self.train_dataset,
            rays_per_batch=self.config.rays_per_batch,
            device=self.device,
            patch_size=self.config.patch_size if self.config.patch_training else 0,
        )

        self.test_dataset = config.dataset_config.load(
            split="test", downsample_factor=min(self.config.iter2downsample.values())
        )

        # Setting up pipeline
        self.rgb_loss = torch.nn.SmoothL1Loss(reduction="none")

        # Use fixed up direction since PCA already normalizes poses
        config.model_config.up_direction = [0.0, 0.0, -1.0]

        points, points_colors = self.train_dataset.get_initial_points()

        self.model = VoroTracing.from_pointcloud(
            config=config.model_config,
            points=points,
            points_colors=points_colors,
            num_random=self.config.num_random_init_points,
            max_points=self.config.num_init_points,
            subsample_density_alpha=self.config.subsample_density_alpha,
            device=self.device,
        )

        self._declare_optimizer(
            config=config.optimizer_config,
            warmup=config.density_warmup_steps,
            max_iterations=config.iterations,
        )

        self.psnr = psnr
        self.ssim = SSIM(data_range=1.0, size_average=True, channel=3)
        self.lpips = LearnedPerceptualImagePatchSimilarity(
            net_type="vgg", normalize=True
        ).to(self.device)
        self.lpips_3dgs = LearnedPerceptualImagePatchSimilarity(
            net_type="vgg", normalize=False
        ).to(self.device)

        # Setting up viewer
        if self.config.viewer:
            self.viewer = VoroTracingViewer(
                model=self.model,
                mode="training",
                up_direction=torch.tensor(self.config.model_config.up_direction),
            )

    def train(self):
        last_triangulation_update = 0
        triangulation_update_period = 1

        # Track training time
        training_start_time = time.time()

        progress = tqdm.trange(self.config.iterations)
        for step in progress:
            if self.config.viewer:
                while self.viewer.state == "paused":
                    time.sleep(0.01)
                self.viewer.lock.acquire()

            if step in self.config.iter2downsample and step:
                downsample = self.config.iter2downsample[step]
                self.train_dataset = self.config.dataset_config.load(
                    split="train", downsample_factor=downsample
                )
                self.ray_batcher = RayBatcher(
                    self.train_dataset,
                    rays_per_batch=self.config.rays_per_batch,
                    device=self.device,
                    patch_size=self.config.patch_size
                    if self.config.patch_training
                    else 0,
                )

            batch = self.ray_batcher.next_batch()
            ray_batch = batch["rays"]
            rgb_batch = batch["rgbs"]
            alpha_batch = batch.get("alphas", torch.ones_like(rgb_batch[..., :1]))

            if self.config.quantile_weight > 0:
                depth_quantiles = (
                    torch.rand(*ray_batch.shape[:-1], 2, device=self.device)
                    .sort(dim=-1, descending=True)
                    .values
                )
            else:
                depth_quantiles = None

            need_contribution = (
                self.config.contribution_weight > 0 or self.config.distortion_percell
            )
            rgba_output, depth, contribution, _, errbox, distortion = self.model(
                ray_batch,
                depth_quantiles=depth_quantiles,
                return_contribution=need_contribution,
            )

            if rgba_output.ndim == 3:  # multiple rays per pixel, average over them
                rgba_output = rgba_output.mean(dim=1)

            # White background
            opacity = rgba_output[..., -1:]
            if self.config.white_background:
                rgb_output = rgba_output[..., :3] + (1 - opacity)
            else:
                rgb_output = rgba_output[..., :3]

            color_loss = self.rgb_loss(rgb_batch, rgb_output)
            opacity_loss = ((alpha_batch - opacity) ** 2).mean()

            if depth_quantiles is not None:
                valid_depth_mask = (depth > 0).all(dim=-1)
                quant_loss = (depth[..., 0] - depth[..., 1]).abs()
                quant_loss = (quant_loss * valid_depth_mask).mean()
                w_depth = self.config.quantile_weight * min(
                    2 * step / self.config.iterations, 1
                )
            else:
                quant_loss = torch.zeros((), device=self.device)
                w_depth = 0.0

            if self.config.contribution_weight > 0:
                num_rays = ray_batch.reshape(-1, 6).shape[0]
                contribution_loss = contribution.sum() / num_rays
                w_contribution = self._contribution_weight(step)
            else:
                contribution_loss = torch.zeros((), device=self.device)
                w_contribution = 0.0

            gate_mean = 1.0  # per-cell distortion gate diagnostic (stays 1.0 when off)
            if self.config.distortion_weight > 0:
                distortion_loss = distortion.mean()
                # Half-time linear warmup: 0 at step 0, full weight
                # at iterations/2, constant thereafter. Lets geometry settle before
                # binarization pressure kicks in.

                # w_distortion = self.config.distortion_weight * min(
                #     2 * step / self.config.iterations, 1
                # )
                w_distortion = self.config.distortion_weight
            else:
                distortion_loss = torch.zeros((), device=self.device)
                w_distortion = 0.0

            # 3DGS-style photometric term: rgb_term = (1-w)*L1/Huber + w*(1-SSIM).
            # SSIM needs spatial patches, so it is only applied under patch_training.
            rgb_term = color_loss.mean()
            if self.config.ssim_weight > 0:
                ps, npat = batch.get("patch_size"), batch.get("n_patches")
                if ps is not None and npat is not None:
                    P = int(ps)
                    pred = (
                        rgb_output.reshape(npat, P, P, 3)
                        .permute(0, 3, 1, 2)
                        .clamp(0, 1)
                    )
                    gt = rgb_batch.reshape(npat, P, P, 3).permute(0, 3, 1, 2)
                    ssim_loss = 1.0 - self.ssim(pred, gt)
                    w = self.config.ssim_weight
                    rgb_term = (1.0 - w) * color_loss.mean() + w * ssim_loss
                elif not getattr(self, "_warned_ssim_nopatch", False):
                    print(
                        "[warn] ssim_weight>0 but batch has no patches "
                        "(enable --patch-training); ignoring SSIM term."
                    )
                    self._warned_ssim_nopatch = True

            loss = (
                rgb_term
                + opacity_loss
                + w_depth * quant_loss
                + w_contribution * contribution_loss
                + w_distortion * distortion_loss
            )

            # Per-cell adaptive distortion: attach the per-ray residual (for the
            # per-cell scatter) and the previous step's confidence gate to the errbox;
            # the gradient split + gating happens inside TraceVoroTracing.backward.
            percell_active = self.config.distortion_percell and w_distortion > 0
            if percell_active:
                ray_residual = color_loss.detach().mean(dim=-1).reshape(-1)  # [N]
                errbox.ray_error = ray_residual.to(
                    self.model.att_diffuse.dtype
                ).contiguous()
                errbox.cell_distortion_gate = getattr(self, "_cell_gate", None)
                if errbox.cell_distortion_gate is not None:
                    gate_mean = float(errbox.cell_distortion_gate.mean())

            self.optimizer.zero_grad(set_to_none=True)

            loss.backward()

            if percell_active:
                # Refresh the per-cell residual EMA -> next step's gate.
                self._update_cell_residual(
                    getattr(errbox, "point_error", None), contribution
                )

            # Manual regularization gradients (avoids large intermediate allocations)
            if self.config.specular_reg_weight > 0 and hasattr(
                self.model, "att_specular"
            ):
                spec = self.model.att_specular
                scale = 2.0 * self.config.specular_reg_weight / spec.numel()
                spec.grad.add_(spec.data, alpha=scale)

            if self.config.diffuse_tv_weight > 0 and hasattr(self.model, "att_diffuse"):
                R = self.model.config.oct_map_res
                diff = self.model.att_diffuse.data.reshape(-1, R, R, 3)
                grad = self.model.att_diffuse.grad.reshape(-1, R, R, 3)
                scale = 2.0 * self.config.diffuse_tv_weight / diff.numel()
                dh = diff[:, 1:, :, :] - diff[:, :-1, :, :]
                dw = diff[:, :, 1:, :] - diff[:, :, :-1, :]
                grad[:, 1:, :, :].add_(dh, alpha=scale)
                grad[:, :-1, :, :].add_(dh, alpha=-scale)
                grad[:, :, 1:, :].add_(dw, alpha=scale)
                grad[:, :, :-1, :].add_(dw, alpha=-scale)

            # Pull each diffuse texel toward its per-cell, per-channel mean.
            # Helps unseen-direction texels (which get no gradient from photometric
            # loss) by globally diffusing the per-cell signal across all texels.
            # Manual gradient: dL/d_att = 2*w/N_total * (att - mean).
            if self.config.diffuse_mean_pull_weight > 0 and hasattr(
                self.model, "att_diffuse"
            ):
                R = self.model.config.oct_map_res
                diff = self.model.att_diffuse.data.reshape(-1, R, R, 3)
                grad = self.model.att_diffuse.grad.reshape(-1, R, R, 3)
                mean = diff.mean(dim=(1, 2), keepdim=True)
                scale = 2.0 * self.config.diffuse_mean_pull_weight / diff.numel()
                grad.add_(diff, alpha=scale)
                grad.add_(mean, alpha=-scale)

            self.optimizer.step()
            self._update_learning_rate(step)

            # Cap log-density: distortion keeps pushing density up past opacity and
            # density=exp(self.density), so it runs away until exp()'s grad overflows
            # fp32 -> NaN. exp(30) is already fully opaque, so this clips nothing real.
            with torch.no_grad():
                self.model.density.clamp_(max=30.0)

            progress.set_postfix(color_loss=f"{color_loss.mean().item():.5f}")

            if (
                step % self.config.train_log_interval
                == self.config.train_log_interval - 1
            ):
                if self.config.wandb:
                    log_dict = {
                        "train/rgb_loss": color_loss.mean(),
                        "train/opacity_loss": opacity_loss.item(),
                        "train/quant_loss": quant_loss.item(),
                        "train/num_points": self.model.primal_points.shape[0],
                        "train/contribution_loss": contribution_loss.item(),
                        "train/contribution_weight": w_contribution,
                        "train/distortion_loss": distortion_loss.item(),
                        "train/distortion_weight": w_distortion,
                        "train/distortion_gate_mean": gate_mean,
                        "lr/points_lr": self.xyz_scheduler_args(step),
                        "lr/density_lr": self.den_scheduler_args(step),
                        "lr/attr_lr": self.attr_dc_scheduler_args(step),
                    }
                    if self.config.specular_reg_weight > 0 and hasattr(
                        self.model, "att_specular"
                    ):
                        log_dict["train/specular_reg"] = (
                            (self.model.att_specular.detach() ** 2).mean().item()
                        )
                    if self.config.diffuse_tv_weight > 0 and hasattr(
                        self.model, "att_diffuse"
                    ):
                        R = self.model.config.oct_map_res
                        d = self.model.att_diffuse.detach().reshape(-1, R, R, 3)
                        log_dict["train/diffuse_tv"] = (
                            (d[:, 1:] - d[:, :-1]).pow(2).mean()
                            + (d[:, :, 1:] - d[:, :, :-1]).pow(2).mean()
                        ).item()
                    if self.config.diffuse_mean_pull_weight > 0 and hasattr(
                        self.model, "att_diffuse"
                    ):
                        R = self.model.config.oct_map_res
                        d = self.model.att_diffuse.detach().reshape(-1, R, R, 3)
                        log_dict["train/diffuse_mean_pull"] = (
                            (d - d.mean(dim=(1, 2), keepdim=True)).pow(2).mean().item()
                        )
                    wandb.log(log_dict, step=step)

            if step % self.config.val_interval == self.config.val_interval - 1:
                val_metrics = self.validation()
                if self.config.wandb:
                    for k, v in val_metrics.items():
                        wandb.log({f"val/{k}": v}, step=step)

                    # Log a random validation image — both center-ray and antialiased
                    img_idx = torch.randint(0, len(self.test_dataset), (1,)).item()
                    out = self.test_dataset[img_idx]
                    rays, gt_rgb = out["rays"], out["rgbs"]
                    rays = rays.cuda()
                    with torch.no_grad():
                        output, _, _, _, _, _ = self.model(rays)

                    full_aa_mode = output.ndim == 4
                    gt_rgb_cpu = gt_rgb.cpu()

                    # Center ray (index 0 = 0.5, 0.5 offset)
                    output_center = output[..., 0, :] if full_aa_mode else output
                    rgb_center = self._output_to_rgb(output_center).reshape(
                        *gt_rgb.shape
                    )
                    combined_center = torch.cat([gt_rgb_cpu, rgb_center.cpu()], dim=1)
                    wandb.log(
                        {"images/rgb": wandb.Image(combined_center.numpy())}, step=step
                    )

                    # Antialiased (mean of all rays)
                    if full_aa_mode:
                        output_aa = output.mean(dim=2)
                        rgb_aa = self._output_to_rgb(output_aa).reshape(*gt_rgb.shape)
                        combined_aa = torch.cat(
                            [gt_rgb_cpu, rgb_center.cpu(), rgb_aa.cpu()],
                            dim=1,
                        )
                        wandb.log(
                            {"images/rgb_aa": wandb.Image(combined_aa.numpy())},
                            step=step,
                        )

            # Log training time at every 1000th step
            if step % 1000 == 0 and step > 0:
                elapsed_time = time.time() - training_start_time
                elapsed_hours = elapsed_time / 3600
                elapsed_minutes = (elapsed_time % 3600) / 60
                elapsed_seconds = elapsed_time % 60
                time_str = f"{int(elapsed_hours):02d}:{int(elapsed_minutes):02d}:{int(elapsed_seconds):02d}"
                print(
                    f"Step {step}: Elapsed training time: {time_str} ({elapsed_time:.2f}s)"
                )
                if self.config.wandb:
                    wandb.log(
                        {"train/elapsed_time_minutes": elapsed_time / 60}, step=step
                    )

            # Voronoi adjacency update
            if step - last_triangulation_update >= triangulation_update_period:
                self.model.update_triangulation(incremental=True)
                last_triangulation_update = step

                if triangulation_update_period < 100:
                    triangulation_update_period += 2

            # Release viewer lock
            if self.config.viewer:
                self.viewer.lock.release()

        # Log final training time
        total_training_time = time.time() - training_start_time
        total_hours = total_training_time / 3600
        total_minutes = (total_training_time % 3600) / 60
        total_seconds = total_training_time % 60
        time_str = (
            f"{int(total_hours):02d}:{int(total_minutes):02d}:{int(total_seconds):02d}"
        )
        print("\nTraining completed!")
        print(f"Total training time: {time_str} ({total_training_time:.2f}s)")
        if self.config.wandb:
            wandb.log(
                {"train/total_training_time_minutes": total_training_time / 60},
                step=self.config.iterations - 1,
            )

        # Save model and config
        if self.config.save_checkpoint:
            os.makedirs(self.out_dir, exist_ok=True)
            self.model.save_checkpoint(f"{self.out_dir}/model.pt")
            self._save_config(f"{self.out_dir}/config.yaml")

        if self.config.export_val:
            self.export_validation()

        if self.config.wandb:
            wandb.finish()

        if self.config.viewer and not self.config.exit_on_completion:
            print("Viewer running... Ctrl+C to exit.")
            time.sleep(1000000)

    def _save_config(self, path: str):
        import yaml
        from dataclasses import asdict

        def clean(obj):
            if isinstance(obj, dict):
                return {k: clean(v) for k, v in obj.items() if not isinstance(v, type)}
            if isinstance(obj, (list, tuple)):
                return type(obj)(clean(v) for v in obj)
            return obj

        config_dict = clean(asdict(self.config))
        config_dict["model"] = type(self.model).__name__
        with open(path, "w") as f:
            yaml.safe_dump(config_dict, f, sort_keys=False)

    def _contribution_weight(self, step: int) -> float:
        if self.config.contribution_weight <= 0:
            return 0.0
        t = min(max(step / max(self.config.iterations - 1, 1), 0.0), 1.0)
        return self.config.contribution_weight * (1e-3**t)

    def _update_cell_residual(self, point_error, contribution):
        """Refresh the per-cell photometric-residual EMA and rebuild the per-cell
        distortion confidence gate (Adaptive Shells analog).

        point_error[i]   = Σ_rays weight_i * ray_residual   (scattered in backward)
        contribution[i]  = Σ_rays weight_i                  (per-cell visible weight)
        -> per-cell mean residual cr_i = point_error_i / contribution_i.
        Gate g_i = exp(-ema_i / (beta * mean_ema)), normalized to mean 1 over seen
        cells so the total binarization budget is conserved (only redistributed:
        confident/low-residual cells get g>1, fuzzy/high-residual cells get g<1).
        """
        if point_error is None or contribution is None:
            return
        pe = point_error.detach().float().reshape(-1).clamp_min(0)
        ct = contribution.detach().float().reshape(-1)
        if pe.shape != ct.shape:
            return
        visible = ct > 1e-6
        cr = torch.zeros_like(pe)
        cr[visible] = pe[visible] / ct[visible]

        decay = self.config.distortion_percell_decay
        ema = getattr(self, "_cell_resid_ema", None)
        if ema is None or ema.shape != cr.shape:
            self._cell_resid_ema = cr.clone()
        else:
            ema[visible] = decay * ema[visible] + (1.0 - decay) * cr[visible]
        ema = self._cell_resid_ema

        seen = ema > 0
        if not bool(seen.any()):
            return
        m = ema[seen].mean().clamp_min(1e-8)
        beta = self.config.distortion_percell_beta
        g = torch.exp(-ema / (beta * m))
        g = g / g[seen].mean().clamp_min(1e-8)  # conserve total budget (mean 1)
        self._cell_gate = g.clamp(0.0, 3.0)

    def _compute_metrics(self, rgb_output, rgb_batch):
        """Compute PSNR, SSIM and LPIPS for a single image pair (H, W, 3)."""
        pred = rgb_output.permute(2, 0, 1)[None]
        gt = rgb_batch.permute(2, 0, 1)[None]
        return {
            "psnr": self.psnr(rgb_output, rgb_batch).mean(),
            "ssim": self.ssim(pred, gt),
            "lpips": self.lpips(pred, gt),
            "lpips_3dgs": self.lpips_3dgs(pred, gt),
        }

    def _output_to_rgb(self, output):
        """Apply white background composite and clip to [0, 1]."""
        opacity = output[..., -1:]
        return (output[..., :3] + (1 - opacity)).clip(0, 1)

    def validation(self) -> dict[str, float]:
        """Run validation and return two metric dicts: regular (center ray) and antialiased (mean of all rays)."""
        poses = self.test_dataset.poses[:, :, 3].reshape(-1, 3)
        start_points = self.model.get_starting_point(poses.cuda())

        metric_lists = {"psnr": [], "ssim": [], "lpips": [], "lpips_3dgs": []}
        metric_lists_aa = {"psnr": [], "ssim": [], "lpips": [], "lpips_3dgs": []}

        with torch.no_grad():
            for i in range(len(self.test_dataset)):
                batch = self.test_dataset[i]
                ray_batch = batch["rays"].cuda()
                rgb_batch = batch["rgbs"].cuda()

                output, _, _, _, _, _ = self.model(ray_batch, start_points[i])

                full_aa_mode = output.ndim == 4  # (H, W, num_aa, 4)

                # --- regular: center ray only (index 0 = offset 0.5, 0.5) ---
                output_center = output[..., 0, :] if full_aa_mode else output
                rgb_center = self._output_to_rgb(output_center).reshape(
                    *rgb_batch.shape
                )
                for k, v in self._compute_metrics(rgb_center, rgb_batch).items():
                    metric_lists[k].append(v)

                # --- antialiased: mean over all AA rays ---
                if full_aa_mode:
                    output_aa = output.mean(dim=2)
                    rgb_aa = self._output_to_rgb(output_aa).reshape(*rgb_batch.shape)
                    for k, v in self._compute_metrics(rgb_aa, rgb_batch).items():
                        metric_lists_aa[k].append(v)

                torch.cuda.synchronize()

        average_metrics = {k: torch.stack(metric_lists[k]).mean() for k in metric_lists}

        # Log antialiased metrics if they exist
        if len(metric_lists_aa["psnr"]) > 0:
            average_metrics_aa = {
                k: torch.stack(metric_lists_aa[k]).mean() for k in metric_lists_aa
            }
            for k, v in average_metrics_aa.items():
                average_metrics[f"{k}_aa"] = v

        return average_metrics

    def export_validation(self):
        renders_dir = f"{self.out_dir}/renders"
        gt_dir = f"{self.out_dir}/gt"
        os.makedirs(renders_dir, exist_ok=True)
        os.makedirs(gt_dir, exist_ok=True)

        poses = self.test_dataset.poses[:, :, 3].reshape(-1, 3)
        start_points = self.model.get_starting_point(poses.cuda())

        with torch.no_grad():
            for i in tqdm.trange(len(self.test_dataset), desc="Exporting val"):
                batch = self.test_dataset[i]
                ray_batch = batch["rays"].cuda()
                gt_rgb = batch["rgbs"]

                output, _, _, _, _, _ = self.model(ray_batch, start_points[i])
                if output.ndim == 4:
                    output = output.mean(dim=2)
                rgb = self._output_to_rgb(output).reshape(*gt_rgb.shape)

                stem = os.path.splitext(self.test_dataset.image_names[i])[0]
                render_img = Image.fromarray((rgb.cpu().numpy() * 255).astype(np.uint8))
                render_img.save(f"{renders_dir}/{stem}.png")

        print(f"Exported {len(self.test_dataset)} validation images to {self.out_dir}")

    def _declare_optimizer(self, config: VoroOptimizerConfig, warmup, max_iterations):
        params = [
            {
                "params": self.model.primal_points,
                "lr": config.points_lr_init,
                "name": "primal_points",
            },
            {
                "params": self.model.density,
                "lr": config.density_lr_init,
                "name": "density",
            },
            {
                "params": self.model.att_diffuse,
                "lr": config.attributes_lr_init,
                "name": "att_diffuse",
            },
            {
                "params": self.model.att_specular,
                "lr": config.attributes_lr_init,
                "name": "att_specular",
            },
        ]

        self.optimizer = torch.optim.Adam(params, eps=1e-15, fused=config.fused_adam)
        self.model.set_optimizer(self.optimizer)

        self.xyz_scheduler_args = get_cosine_lr_func(
            lr_init=config.points_lr_init,
            lr_final=config.points_lr_final,
            max_steps=config.freeze_points,
        )
        self.den_scheduler_args = get_cosine_lr_func(
            lr_init=config.density_lr_init,
            lr_final=config.density_lr_final,
            warmup_steps=warmup,
            max_steps=max_iterations,
        )
        self.attr_dc_scheduler_args = get_cosine_lr_func(
            lr_init=config.attributes_lr_init,
            lr_final=config.attributes_lr_final,
            max_steps=max_iterations,
        )

    def _update_learning_rate(self, iteration):
        """Learning rate scheduling per step"""
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "primal_points":
                lr = self.xyz_scheduler_args(iteration)
                param_group["lr"] = lr
            elif param_group["name"] == "density":
                lr = self.den_scheduler_args(iteration)
                param_group["lr"] = lr
            elif param_group["name"] == "att_diffuse":
                lr = self.attr_dc_scheduler_args(iteration)
                param_group["lr"] = lr
            elif param_group["name"] == "att_specular":
                spec_start = self.config.optimizer_config.specular_start
                lr = self.attr_dc_scheduler_args(iteration)
                if iteration < spec_start:
                    ramp = (iteration / spec_start) ** 2
                    lr *= ramp
                param_group["lr"] = lr


if __name__ == "__main__":
    config = tyro.cli(VoroTrainerConfig)
    trainer = config.setup()
    trainer.train()
