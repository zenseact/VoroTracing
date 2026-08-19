from pathlib import Path
from dataclasses import dataclass
from typing import Optional

import torch
import tyro
import yaml
from pytorch_msssim import SSIM
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from vorotracing.datasets.datasets import ColmapConfig
from vorotracing.vorotracing import VoroTracingConfig, VoroTracingInfer
from vorotracing.utils import psnr


@dataclass
class BenchmarkConfig:
    base_dir: Path = Path("output")
    """Base directory to scan for checkpoints (subdirs with model.pt + *.yaml)."""
    dataset_dir: Path = Path("data/mipnerf360")
    """Path to the dataset root."""
    scene: str = "garden"
    """Scene name within the dataset."""
    downsample: int = 4
    """Downsample factor for test images."""
    warmup: int = 10
    """Number of warmup iterations before timing."""
    timing_iters: int = 20
    """Number of timed iterations per view (takes median)."""
    csv: Optional[Path] = None
    """Optional path to save results as CSV."""
    white_background: bool = True
    """Whether to composite with white background."""
    quantize: str = "fp16"
    """Attribute quantization: 'fp16' or 'q8'."""
    sort_morton: bool = True
    """Reorder cells by Morton code before inference."""
    use_warp_perm: bool = True
    """Render full images in warp-coherent 4x8 screen-space tiles."""
    weight_threshold: float = 0.01
    """Transmittance threshold for early ray termination."""
    cell_skip_threshold: float = 1e-3
    """Per-cell contribution gate (fp16 only). e.g. 1e-4 skips texture loads on low-impact cells."""


def find_checkpoints(base_dir: Path) -> list[Path]:
    """Find all model.pt files in subdirectories of base_dir."""
    checkpoints = sorted(base_dir.glob("**/model.pt"))
    return checkpoints


def load_config(ckpt_path: Path) -> tuple[VoroTracingConfig, dict]:
    """Load model config and full training config from checkpoint YAML."""
    yaml_files = list(ckpt_path.parent.glob("*.yaml"))
    config_data = {}
    if yaml_files:
        with open(yaml_files[0], "r") as f:
            config_data = yaml.safe_load(f)
    model_fields = {
        k: v
        for k, v in config_data.get("model_config", config_data).items()
        if k in VoroTracingConfig.__dataclass_fields__
    }
    try:
        model_config = VoroTracingConfig(**model_fields)
    except Exception as e:
        print(f"  Warning: could not parse config ({e}), using defaults")
        model_config = VoroTracingConfig()
    return model_config, config_data


def get_eval_downsample(config_data: dict, fallback: int = 4) -> int:
    """Get the final (highest-res) downsample factor from training config."""
    iter2ds = config_data.get("iter2downsample", {})
    if iter2ds:
        return min(iter2ds.values())
    return fallback


def output_to_rgb(output, white_background=True):
    opacity = output[..., -1:]
    if white_background:
        return (output[..., :3] + (1 - opacity)).clip(0, 1)
    return output[..., :3].clip(0, 1)


def benchmark_checkpoint(
    ckpt_path: Path,
    dataset,
    config: BenchmarkConfig,
    ssim_metric,
    lpips_metric,
    lpips_3dgs_metric,
    device,
):
    print(f"\n{'=' * 60}")
    print(f"  {ckpt_path.parent.name}")
    print(f"{'=' * 60}")

    model_config, _ = load_config(ckpt_path)
    model = VoroTracingInfer.from_pretrained(
        ckpt_path,
        model_config,
        device=device,
        quantize=config.quantize,
        sort_morton=config.sort_morton,
    )
    num_points = model.primal_points.shape[0]
    print(f"  Points: {num_points:,}")

    metric_lists = {"psnr": [], "ssim": [], "lpips": [], "lpips_3dgs": []}
    timings_ms = []

    with torch.no_grad():
        for i in range(len(dataset)):
            batch = dataset[i]
            rays = batch["rays"].to(device)
            gt_rgb = batch["rgbs"].to(device)

            # Quality metrics (single pass)
            rgba = model.render(
                rays,
                weight_threshold=config.weight_threshold,
                cell_skip_threshold=config.cell_skip_threshold,
                use_warp_perm=config.use_warp_perm,
            ).float()
            rgb = output_to_rgb(rgba, config.white_background)
            rgb = rgb.reshape(*gt_rgb.shape)

            metric_lists["psnr"].append(psnr(rgb, gt_rgb).mean())
            pred = rgb.permute(2, 0, 1)[None]
            gt = gt_rgb.permute(2, 0, 1)[None]
            metric_lists["ssim"].append(ssim_metric(pred, gt))
            metric_lists["lpips"].append(lpips_metric(pred, gt))
            metric_lists["lpips_3dgs"].append(lpips_3dgs_metric(pred, gt))

            # Timing
            for _ in range(config.warmup):
                model.render(
                    rays,
                    weight_threshold=config.weight_threshold,
                    cell_skip_threshold=config.cell_skip_threshold,
                    use_warp_perm=config.use_warp_perm,
                )
            torch.cuda.synchronize()
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            for _ in range(config.timing_iters):
                model.render(
                    rays,
                    weight_threshold=config.weight_threshold,
                    cell_skip_threshold=config.cell_skip_threshold,
                    use_warp_perm=config.use_warp_perm,
                )
            end_event.record()
            torch.cuda.synchronize()
            timings_ms.append(start_event.elapsed_time(end_event) / config.timing_iters)

    avg_metrics = {k: torch.stack(v).mean().item() for k, v in metric_lists.items()}
    avg_ms = sum(timings_ms) / len(timings_ms)

    return {
        "name": ckpt_path.parent.name,
        "quantize": config.quantize,
        "sort_morton": config.sort_morton,
        "use_warp_perm": config.use_warp_perm,
        "weight_threshold": config.weight_threshold,
        "cell_skip_threshold": config.cell_skip_threshold,
        "points": num_points,
        "psnr": avg_metrics["psnr"],
        "ssim": avg_metrics["ssim"],
        "lpips": avg_metrics["lpips"],
        "lpips_3dgs": avg_metrics["lpips_3dgs"],
        "ms_per_frame": avg_ms,
        "fps": 1000.0 / avg_ms,
    }


def infer_dataset_from_checkpoint(
    ckpt_path: Path, config_data: dict, fallback_dir: str, fallback_scene: str
) -> tuple[str, str]:
    """Extract data_path and scene from the saved training config."""
    ds_config = config_data.get("dataset_config", {})
    data_path = ds_config.get("data_path", fallback_dir)
    scene = ds_config.get("scene", fallback_scene)
    # Remap server paths to local paths if the original doesn't exist
    if not Path(data_path).exists():
        name = Path(data_path).name
        local = Path("data") / name
        if local.exists():
            data_path = str(local)
    return data_path, scene


def run_benchmark(config: BenchmarkConfig):
    device = torch.device("cuda")

    checkpoints = find_checkpoints(config.base_dir)
    if not checkpoints:
        print(f"No checkpoints found in {config.base_dir}")
        return []

    print(f"Found {len(checkpoints)} checkpoint(s) in {config.base_dir}")

    ssim_metric = SSIM(data_range=1.0, size_average=True, channel=3)
    lpips_metric = LearnedPerceptualImagePatchSimilarity(
        net_type="vgg", normalize=True
    ).to(device)
    lpips_3dgs_metric = LearnedPerceptualImagePatchSimilarity(
        net_type="vgg", normalize=False
    ).to(device)

    dataset_cache = {}
    results = []
    for ckpt in checkpoints:
        try:
            model_config, train_config = load_config(ckpt)
            data_path, scene = infer_dataset_from_checkpoint(
                ckpt, train_config, str(config.dataset_dir), config.scene
            )
            ds_factor = get_eval_downsample(train_config, config.downsample)
            cache_key = (data_path, scene, ds_factor)
            if cache_key not in dataset_cache:
                ds_config = ColmapConfig(
                    data_path=data_path,
                    scene=scene,
                )
                dataset_cache[cache_key] = ds_config.load(
                    split="test", downsample_factor=ds_factor
                )
                print(
                    f"Loaded test set: {len(dataset_cache[cache_key])} images, "
                    f"scene={scene}, data_path={data_path}, downsample={ds_factor}"
                )
            r = benchmark_checkpoint(
                ckpt,
                dataset_cache[cache_key],
                config,
                ssim_metric,
                lpips_metric,
                lpips_3dgs_metric,
                device,
            )
            results.append(r)
        except Exception as e:
            print(f"  ERROR: {e}")
            continue

    return results


def main(config: BenchmarkConfig):
    results = run_benchmark(config)
    if not results:
        return

    # Print table
    print(f"\n{'=' * 90}")
    print(
        f"{'Checkpoint':<40} | {'Points':>8} | {'PSNR':>6} | {'SSIM':>6} | {'LPIPS':>6} | {'ms':>7} | {'FPS':>5}"
    )
    print(f"{'-' * 90}")
    for r in results:
        print(
            f"{r['name']:<40} | {r['points']:>8,} | {r['psnr']:>6.2f} | {r['ssim']:>6.4f} | "
            f"{r['lpips']:>6.4f} | {r['ms_per_frame']:>7.2f} | {r['fps']:>5.0f}"
        )

    if config.csv:
        import csv

        with open(config.csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        print(f"\nResults saved to {config.csv}")


if __name__ == "__main__":
    main(tyro.cli(BenchmarkConfig))
