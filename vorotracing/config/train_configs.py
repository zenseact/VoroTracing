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
    num_final_points=2_000_000,
    densify_from=2000,
    densify_until=11000,
    densify_factor=1.15,
    white_background=True,
    quantile_weight=0.0,
    contribution_weight=0.0,
    distortion_weight=2e-3,
    specular_reg_weight=1e-2,
    diffuse_tv_weight=0.0,
    diffuse_mean_pull_weight=5e-3,
    experiment_name="vorotracing-outdoor",
)

# Used in Mip-NeRF 360 scenes: bonsai, counter, kitchen, room
_train_configs["vorotracing-indoor"] = deepcopy(_train_configs["vorotracing-outdoor"])
_train_configs["vorotracing-indoor"].iter2downsample = {0: 2}
_train_configs["vorotracing-indoor"].num_init_points = 2_000_000
_train_configs["vorotracing-indoor"].num_final_points = 2_000_000
_train_configs["vorotracing-indoor"].specular_reg_weight = 1e-4
_train_configs["vorotracing-indoor"].diffuse_mean_pull_weight = 1e-4
_train_configs["vorotracing-indoor"].optimizer_config.specular_start = 0
_train_configs["vorotracing-indoor"].dataset_config.scene = "bonsai"

# Used in Deep Blending dataset: drjohnson, playroom
_train_configs["vorotracing-db"] = deepcopy(_train_configs["vorotracing-indoor"])
_train_configs["vorotracing-db"].iter2downsample = {0: 1}
_train_configs["vorotracing-db"].specular_reg_weight = 1e-1
_train_configs["vorotracing-db"].dataset_config.data_path = "data/db"
_train_configs["vorotracing-db"].dataset_config.scene = "drjohnson"

# Used in Tanks & Temples dataset: train, truck
_train_configs["vorotracing-tandt"] = deepcopy(_train_configs["vorotracing-outdoor"])
_train_configs["vorotracing-tandt"].iter2downsample = {0: 1}
_train_configs["vorotracing-tandt"].specular_reg_weight = 5e-4
_train_configs["vorotracing-tandt"].dataset_config.data_path = "data/tandt"
_train_configs["vorotracing-tandt"].dataset_config.scene = "train"

# Used in DL3DV dataset; we download the 2K resolution (1920x1080) and only have one image set
_train_configs["vorotracing-dl3dv"] = deepcopy(_train_configs["vorotracing-indoor"])
_train_configs["vorotracing-dl3dv"].iter2downsample = {0: 1}
_train_configs["vorotracing-dl3dv"].dataset_config.data_path = "data/dl3dv"
_train_configs[
    "vorotracing-dl3dv"
].dataset_config.scene = (
    "41566d172f25ef7e3841ac4fbcbc83cace2df1614d699547051df02f567b8101"
)

TrainConfigList = (
    tyro.conf.SuppressFixed[
        tyro.extras.subcommand_type_from_defaults(defaults=_train_configs)
    ]
)
