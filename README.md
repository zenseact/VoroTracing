# VoroTracing

_Differentiable Voronoi Ray Tracing Beyond Rasterization Speeds_

[![Project Page](https://img.shields.io/badge/Project-Page-ffa)](https://research.zenseact.com/publications/vorotracing) [![arXiv Paper](https://img.shields.io/badge/arXiv-Paper-aff)](https://arxiv.org/abs/2608.17682) [![Web Viewer](https://img.shields.io/badge/Web-Viewer-afa)](https://research.zenseact.com/publications/vorotracing/viewer/)

VoroTracing is a differentiable Voronoi ray tracer for real-time novel view synthesis. Ray-based rendering expresses non-pinhole effects such as distortion, rolling shutter, and depth of field naturally, but is generally assumed too slow to compete with rasterization. We analyze the factors that govern throughput in differentiable Voronoi ray tracing (traversal length, per-cell work, and memory locality) and co-design the scene representation, the optimization, and the GPU execution around those costs.

Compact octahedral appearance textures cut memory traffic, surface-concentrated opacity promotes early ray termination, and the fixed-budget representation is optimized without pruning or densification. On Mip-NeRF 360, VoroTracing renders at 623 FPS on an RTX 5090, giving 3.2× the throughput of the fastest prior ray-based method and 2.8× that of 3D Gaussian Splatting at competitive reconstruction quality.

This repository contains the training, evaluation, and viewing code for the paper. Voronoi diagram construction is provided by [Paragram](https://github.com/zenseact/paragram).

## Install

The code requires an NVIDIA GPU with CUDA 12.x and NVCC. The CUDA extension is compiled lazily on first use, which takes a few minutes the first time.

Clone with submodules (`external/eigen` and `external/cubql` are required to build):

```bash
git clone --recurse-submodules https://github.com/zenseact/VoroTracing.git
```

If you already cloned without them:

```bash
git submodule update --init --recursive
```

Then install the environment with [uv](https://docs.astral.sh/uv/) from the repository root:

```bash
uv sync
```

### Compile Options

Useful build environment variables:

- `MAX_JOBS`: parallel compile jobs (defaults to 10).
- `VERBOSE=1`: print the full build log.
- `DEBUG_CUDA=1`: compile with debug CUDA flags.
- `PROFILE_CUDA=1`: compile with line info for profiling.
- `USE_PRECOMPILED_HEADERS=1`: enable precompiled headers.

## Data

Datasets are read in COLMAP format and are expected under `data/`:

```text
data/mipnerf360/garden/images
data/mipnerf360/garden/images_2
data/mipnerf360/garden/images_4
data/mipnerf360/garden/sparse/0
```

Downsampled image folders (`images_2`, `images_4`, `images_8`) are used directly: indoor scenes train at `images_2` and outdoor scenes at `images_4`. Every 8th image is held out as the test split, matching the standard Mip-NeRF 360 protocol.

## Train

Train a Mip-NeRF 360 outdoor scene:

```bash
uv run python train.py vorotracing-outdoor --dataset-config.data-path data/mipnerf360 --dataset-config.scene garden
```

Use `vorotracing-outdoor` for `garden`, `bicycle`, and `stump`; use `vorotracing-indoor` for `bonsai`, `counter`, `kitchen`, and `room`. The two configs differ in training resolution and appearance regularization.

The interactive viewer attaches to the running job by default and serves on `http://localhost:7007`, so you can watch the scene converge. Pass `--no-viewer` for headless runs, and `--wandb` to log metrics and validation renders to Weights & Biases.

Outputs are written to `output/<experiment-name>_<timestamp>/` and contain `model.pt` and the resolved `config.yaml`.

Run `uv run python train.py vorotracing-outdoor --help` to see every option.

## Evaluate

Evaluate all checkpoints found under an output directory:

```bash
uv run python benchmark_eval.py --base-dir output --dataset-dir data/mipnerf360 --csv results.csv
```

This walks every `model.pt` under `--base-dir` and reports PSNR, SSIM, LPIPS, LPIPS-3DGS, milliseconds per frame, and FPS for each. The scene name and evaluation resolution are read from each checkpoint's config, so a single run can cover all seven scenes at once; test sets are loaded once and shared across checkpoints.

The inference settings used in the paper are exposed as flags, with the paper defaults already in place:

- `--quantize {fp16,q8}`: attribute quantization (default `fp16`).
- `--no-sort-morton`: disable Morton-code reordering of cells before inference.
- `--no-use-warp-perm`: disable warp-coherent 4×8 screen-space tiling.
- `--weight-threshold`: transmittance threshold for early ray termination (default `0.01`).
- `--cell-skip-threshold`: per-cell contribution gate, fp16 only (default `1e-3`).

## Viewer

Two viewers are available.

**Web viewer.** Explore the pretrained scenes in the browser, no install required: [research.zenseact.com/publications/vorotracing/viewer](https://research.zenseact.com/publications/vorotracing/viewer/)

**Local viewer.** Launch the [viser](https://viser.studio/)-based viewer on a local checkpoint:

```bash
uv run python viewer.py --ckpt-dir output/vorotracing-outdoor_20260819_120000/model.pt
```

It serves on `http://localhost:7007` and exposes RGB and depth render modes, the primal point cloud, and per-cell inspection overlays. The config YAML next to the checkpoint is picked up automatically.

## Citation

If you find this work useful, please consider citing:

```bibtex
@article{taveira2026vorotracing,
  author  = {Taveira, Bernardo and Lindstr{\"o}m, Carl and Johnander, Joakim and Kahl, Fredrik},
  title   = {Differentiable Voronoi Ray Tracing Beyond Rasterization Speeds},
  journal = {arXiv preprint arXiv:2608.17682},
  year    = {2026}
}
```

## Acknowledgements

This work builds on [Paragram](https://github.com/zenseact/paragram) for GPU Voronoi diagram construction, [cuBQL](https://github.com/NVIDIA/cuBQL) for BVH queries, [Eigen](https://gitlab.com/libeigen/eigen), and [nerfview](https://github.com/nerfstudio-project/nerfview) with [viser](https://github.com/nerfstudio-project/viser) for the interactive viewer.

## License

Apache License 2.0. See [LICENSE](LICENSE).
