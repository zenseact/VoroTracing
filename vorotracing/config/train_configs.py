from copy import deepcopy
from typing import Dict

import tyro

from vorotracing.datasets.datasets import ColmapConfig
from vorotracing.trainer import VoroOptimizerConfig, VoroTrainerConfig
from vorotracing.vorotracing import VoroTracingConfig

_train_configs: Dict[str, VoroTrainerConfig] = {}

# Used in Mip-NeRF 360 scenes: garden, bicycle, stump
_train_configs["vorotracing-outdoor"] = VoroTrainerConfig(
    model_config=VoroTracingConfig(
        oct_map_res=8,
        activation_scale=1.0,
    ),
    dataset_config=ColmapConfig(
        data_path="data/mipnerf360",
        scene="garden",
    ),
    optimizer_config=VoroOptimizerConfig(
        points_lr_init=2e-4,
        points_lr_final=5e-6,
        density_lr_init=1e-1,
        density_lr_final=1e-2,
        attributes_lr_init=2e-2,
        attributes_lr_final=5e-4,
        freeze_points=18000,
        specular_start=4000,
    ),
    iterations=20000,
    rays_per_batch=1000000,
    val_interval=1000,
    train_log_interval=100,
    iter2downsample={0: 4},
    num_init_points=2_000_000,
    density_warmup_steps=2000,
    white_background=True,
    distortion_weight=2e-3,
    specular_reg_weight=1e-2,
    diffuse_mean_pull_weight=5e-3,
    experiment_name="vorotracing-outdoor",
)

# Used in Mip-NeRF 360 scenes: bonsai, counter, kitchen, room
_train_configs["vorotracing-indoor"] = deepcopy(_train_configs["vorotracing-outdoor"])
_train_configs["vorotracing-indoor"].iter2downsample = {0: 2}
_train_configs["vorotracing-indoor"].num_init_points = 2_000_000
_train_configs["vorotracing-indoor"].specular_reg_weight = 1e-4
_train_configs["vorotracing-indoor"].diffuse_mean_pull_weight = 1e-4
_train_configs["vorotracing-indoor"].optimizer_config.specular_start = 0
_train_configs["vorotracing-indoor"].dataset_config.scene = "bonsai"

TrainConfigList = (
    tyro.conf.SuppressFixed[
        tyro.extras.subcommand_type_from_defaults(defaults=_train_configs)
    ]
)
