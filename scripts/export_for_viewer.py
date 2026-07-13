"""Export a trained VoroTracing scene to the flat .foam binary format (V4)
consumed by the web viewer (web-foam), the Mojo viewer (mojo-foam), and the
native Metal/CUDA viewer (foam-viewer).

Usage:
    python export_for_viewer.py --checkpoint scene.pt --config scene.yaml \\
        --out scene.foam [--no-specular] [--legacy-v3] [--reference-png ref.png]

V4 is V3 minus two redundancies, for ~31% smaller files (lossless):
  * octmaps store tight RGB (3 f16/texel) instead of RGBA — the 4th channel was
    always zero padding the renderer never reads. The loader re-pads to vec4 on
    upload. Saves ~21% of the file.
  * adjacent_diff is dropped entirely — it is exactly points[neighbor] -
    points[self] (the Voronoi face normal), recomputable on load from
    point_adjacency in one O(M) pass. Saves ~10%. adjacent_diff_size is set 0.

V4 layout (all little-endian, header is 120 B):

    magic:               u32 = 0x464F414D ("FOAM")
    version:             u32 = 4    // V3 = legacy (vec4 octmaps + adjacent_diff)
    num_points:          u32
    num_poses:           u32   // training/test camera poses bundled for nav
    adjacency_size:      u32   // total length of point_adjacency
    adjacent_diff_size:  u32   // 0 in V4 (blob dropped); == adjacency_size in V3
    octmap_res_diff:     u32   // typically 8
    octmap_res_spec:     u32   // 0 when has_specular == 0
    weight_threshold:    f32   // inference defaults baked into the file
    cell_skip_threshold: f32
    max_intersections:   u32
    default_pose:        [12]f32   // T_c2w 3x4 row-major == poses[0]
    aabb_min:            [3]f32
    aabb_max:            [3]f32
    has_specular:        u32       // V3 addition (offset 116)

    --- raw blobs, in declaration order ---
    points:                  [num_points][3]f32
    density:                 [num_points]f32
    diffuse:                 [num_points][R_diff*R_diff*C]f16  // C = 3 (V4) | 4 (V3)
    specular:                [num_points][R_spec*R_spec*C]f16  // omitted if !has_specular
    point_adjacency:         [adjacency_size]u32
    point_adjacency_offsets: [num_points+1]u32
    adjacent_diff:           [adjacent_diff_size][4]f16        // V3 only; absent in V4
    poses:                   [num_poses][12]f32   // T_c2w 3x4 row-major

Float16 is stored as raw u16 little-endian (matches x86/ARM native layout).

`--no-specular` writes has_specular=0, sets octmap_res_spec=0, and omits the
specular blob entirely (the file shrinks by roughly half on typical scenes).
`--legacy-v3` emits the older V3 layout (vec4 octmaps + adjacent_diff) for
consumers that don't yet read V4.
"""

import argparse
import struct
from pathlib import Path

import numpy as np
import torch
import yaml

from vorotracing.datasets.datasets import ColmapConfig
from vorotracing.vorotracing import VoroTracingConfig, VoroTracingInfer
from vorotracing.utils import generate_camera_rays


MAGIC = 0x464F414D
VERSION = 4  # current default; --legacy-v3 still emits 3


def _octmap_to_rgb(oct_rgba: np.ndarray, n_points: int, res: int) -> np.ndarray:
    """Drop the always-zero 4th channel: (N, R*R*4) f16 -> (N, R*R*3) f16."""
    return oct_rgba.reshape(n_points, res * res, 4)[:, :, :3].reshape(n_points, -1)


def load_config(path: Path) -> tuple[VoroTracingConfig, dict]:
    with open(path) as f:
        raw = yaml.safe_load(f)
    # Tolerate either a flat VoroTracingConfig dump or a full training-config dump
    # that nests the model fields under `model_config:`.
    cfg_dict = raw.get("model_config", raw)
    fields = {f.name for f in VoroTracingConfig.__dataclass_fields__.values()}
    model_cfg = VoroTracingConfig(**{k: v for k, v in cfg_dict.items() if k in fields})
    return model_cfg, raw


def _remap_data_path(data_path: str) -> str:
    """Server paths like /proj/.../mipnerf360 get remapped to local data/mipnerf360
    if the original doesn't exist locally (mirrors benchmark_eval.py)."""
    if Path(data_path).exists():
        return data_path
    local = Path("data") / Path(data_path).name
    return str(local) if local.exists() else data_path


def load_poses(
    raw_config: dict,
    data_path_override: str | None,
    scene_override: str | None,
    split: str,
    downsample: int,
) -> torch.Tensor:
    """Load the actual training/test camera poses for this scene.
    Returns a (N, 3, 4) float32 tensor of T_c2w matrices."""
    ds_raw = raw_config.get("dataset_config", {})
    data_path = data_path_override or ds_raw.get("data_path", "data/mipnerf360")
    scene = scene_override or ds_raw.get("scene", "garden")
    data_path = _remap_data_path(data_path)
    print(f"Loading {split!r} poses from {data_path}/{scene} (downsample={downsample})")

    def _one(split_name):
        ds = ColmapConfig(data_path=data_path, scene=scene).load(
            split=split_name, downsample_factor=downsample
        )
        return ds.poses.detach().cpu().float()

    if split == "all":
        return torch.cat([_one("train"), _one("test")], dim=0)
    return _one(split)


def render_reference(model: VoroTracingInfer, T_c2w_3x4, width, height, fov_deg):
    """Render one frame at the default pose for regression-testing the viewer."""
    device = model.device
    fy = 0.5 * height / np.tan(np.deg2rad(0.5 * fov_deg))
    fx = fy
    cx, cy = 0.5 * (width - 1), 0.5 * (height - 1)
    K = torch.tensor(
        [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
        dtype=torch.float32,
        device=device,
    )
    T = torch.eye(4, dtype=torch.float32, device=device)
    T[:3, :4] = torch.from_numpy(T_c2w_3x4).to(device)

    rays = generate_camera_rays(K, T, width, height, device=device)
    rgba = model.render(rays).reshape(height, width, 4).float()
    # The model is trained against sRGB JPGs, so its output is already in sRGB
    # space — no linear→sRGB curve here. Composite premultiplied RGB onto a
    # white background to match benchmark_eval.py's output_to_rgb().
    opacity = rgba[..., 3:4]
    rgb = (rgba[..., :3] + (1.0 - opacity)).clamp(0.0, 1.0)
    out = torch.cat([rgb, torch.ones_like(opacity)], dim=-1).cpu().numpy()
    return (out * 255 + 0.5).astype(np.uint8)


def write_foam(out_path: Path, model: VoroTracingInfer, config: VoroTracingConfig,
               poses: np.ndarray, aabb_min, aabb_max,
               weight_threshold, cell_skip_threshold, max_intersections,
               include_specular: bool = True, version: int = VERSION):
    pts = model.primal_points.detach().cpu().contiguous().numpy().astype(np.float32)
    density = model.density.detach().cpu().contiguous().numpy().astype(np.float32).reshape(-1)
    diffuse = model.diffuse_fp16.detach().cpu().contiguous().numpy().astype(np.float16)
    adj = model.point_adjacency.detach().cpu().contiguous().numpy().astype(np.uint32)
    adj_off = model.point_adjacency_offsets.detach().cpu().contiguous().numpy().astype(np.uint32)
    adj_diff = model.adjacent_diff.detach().cpu().contiguous().numpy().astype(np.float16)

    n_points = pts.shape[0]
    assert pts.shape == (n_points, 3)
    assert density.shape == (n_points,)
    assert adj_off.shape == (n_points + 1,)
    R_d = config.oct_map_res
    assert diffuse.shape == (n_points, R_d * R_d * 4), \
        f"diffuse {diffuse.shape} != ({n_points}, {R_d*R_d*4})"

    if include_specular:
        specular = model.specular_fp16.detach().cpu().contiguous().numpy().astype(np.float16)
        R_s = config.spec_oct_map_res
        assert specular.shape == (n_points, R_s * R_s * 4)
        has_specular = 1
    else:
        specular = None
        R_s = 0
        has_specular = 0

    # V4: drop the zero 4th octmap channel and the recomputable adjacent_diff.
    if version >= 4:
        diffuse = _octmap_to_rgb(diffuse, n_points, R_d)
        if include_specular:
            specular = _octmap_to_rgb(specular, n_points, R_s)
        adj_diff_size = 0  # blob omitted; loader recomputes from adjacency
    else:
        adj_diff_size = adj_diff.shape[0]

    poses = np.ascontiguousarray(poses, dtype=np.float32)
    n_poses = poses.shape[0]
    assert poses.shape == (n_poses, 3, 4), f"poses shape {poses.shape}"
    default_pose = poses[0]

    # 11 leading u32/f32 fields (44 B) + 12-f32 default_pose (48 B)
    # + 3-f32 aabb_min (12 B) + 3-f32 aabb_max (12 B) + has_specular u32 (4 B)
    # = 120 B header.
    header = struct.pack(
        "<IIIIIIIIffI",
        MAGIC, version,
        n_points, n_poses, adj.shape[0], adj_diff_size,
        R_d, R_s,
        float(weight_threshold), float(cell_skip_threshold),
        int(max_intersections),
    )

    with open(out_path, "wb") as f:
        f.write(header)
        f.write(default_pose.tobytes(order="C"))
        f.write(np.asarray(aabb_min, dtype=np.float32).tobytes(order="C"))
        f.write(np.asarray(aabb_max, dtype=np.float32).tobytes(order="C"))
        f.write(struct.pack("<I", has_specular))
        f.write(pts.tobytes(order="C"))
        f.write(density.tobytes(order="C"))
        f.write(diffuse.tobytes(order="C"))
        if include_specular:
            f.write(specular.tobytes(order="C"))
        f.write(adj.tobytes(order="C"))
        f.write(adj_off.tobytes(order="C"))
        if version < 4:
            f.write(adj_diff.tobytes(order="C"))
        f.write(poses.tobytes(order="C"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, type=Path)
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--reference-png", type=Path, default=None,
                    help="If set, also render one frame at the default pose to this PNG for regression testing.")
    ap.add_argument("--ref-width", type=int, default=1280)
    ap.add_argument("--ref-height", type=int, default=720)
    ap.add_argument("--fov", type=float, default=60.0)
    ap.add_argument("--weight-threshold", type=float, default=0.001)
    ap.add_argument("--cell-skip-threshold", type=float, default=0.0)
    ap.add_argument("--max-intersections", type=int, default=1024)
    ap.add_argument("--pose-split", choices=["train", "test", "all"], default="train",
                    help="Which dataset split's poses to bundle (default: train).")
    ap.add_argument("--pose-downsample", type=int, default=8,
                    help="Downsample factor used when loading the dataset for poses. "
                         "Only affects how fast images load; pose values are identical.")
    ap.add_argument("--data-path", type=str, default=None,
                    help="Override dataset root (default: read from checkpoint's training config).")
    ap.add_argument("--scene", type=str, default=None,
                    help="Override scene name (default: read from checkpoint's training config).")
    ap.add_argument("--no-specular", action="store_true",
                    help="Drop the specular octmap (~half the file size). "
                         "Diffuse-only rendering — view-dependent colour comes from the diffuse "
                         "octmap alone. Sets has_specular=0 and R_s=0 in the header.")
    ap.add_argument("--legacy-v3", action="store_true",
                    help="Emit the older V3 layout (vec4 octmaps + adjacent_diff) instead of "
                         "the default V4 (~31%% larger). For consumers that don't read V4 yet.")
    args = ap.parse_args()
    version = 3 if args.legacy_v3 else 4

    config, raw_config = load_config(args.config)
    print(f"Loading {args.checkpoint} with oct_map_res={config.oct_map_res}, spec_oct_map_res={config.spec_oct_map_res}")

    model = VoroTracingInfer.from_pretrained(args.checkpoint, config, quantize="fp16")

    pts_np = model.primal_points.detach().cpu().numpy().astype(np.float32)
    aabb_min = pts_np.min(axis=0)
    aabb_max = pts_np.max(axis=0)

    poses = load_poses(
        raw_config, args.data_path, args.scene, args.pose_split, args.pose_downsample,
    ).numpy()
    print(f"  {poses.shape[0]} poses; using poses[0] as default opening view")

    spec_note = "diffuse-only" if args.no_specular else "with specular"
    print(f"Writing {args.out} V{version} ({spec_note}, {pts_np.shape[0]} points, "
          f"adjacency={model.point_adjacency.shape[0]}, "
          f"adjacent_diff={model.adjacent_diff.shape[0]})")
    write_foam(
        args.out, model, config, poses, aabb_min, aabb_max,
        args.weight_threshold, args.cell_skip_threshold, args.max_intersections,
        include_specular=not args.no_specular, version=version,
    )
    print(f"  written {args.out.stat().st_size / 1e6:.1f} MB")

    if args.reference_png is not None:
        print(f"Rendering reference frame to {args.reference_png}")
        img = render_reference(model, poses[0], args.ref_width, args.ref_height, args.fov)
        try:
            from PIL import Image
            Image.fromarray(img, mode="RGBA").save(args.reference_png)
        except ImportError:
            # Fallback: write a minimal raw RGBA PPM-style header if PIL isn't around.
            import imageio.v3 as iio
            iio.imwrite(args.reference_png, img)


if __name__ == "__main__":
    main()
