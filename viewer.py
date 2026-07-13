import tyro
import time
from pathlib import Path

import torch
import yaml

from vorotracing.vorotracing import VoroTracing, VoroTracingConfig
from vorotracing.viewer import VoroTracingViewer


@torch.no_grad()
def run_viewer(ckpt_dir: Path = Path("data/garden/model.pt")):
    device = torch.device("cuda")

    # find .yaml file in the directory
    yaml_files = list(ckpt_dir.parent.glob("*.yaml"))
    print(yaml_files)
    if yaml_files:
        with open(yaml_files[0], "r") as f:
            config_data = yaml.safe_load(f)
        model_fields = config_data.get("model_config", config_data)
        try:
            model_config = VoroTracingConfig(
                **{
                    k: v
                    for k, v in model_fields.items()
                    if k in VoroTracingConfig.__dataclass_fields__
                }
            )
        except TypeError:
            model_config = VoroTracingConfig()
    else:
        model_config = VoroTracingConfig()

    model = VoroTracing.from_pretrained(ckpt_dir, model_config, device=device)

    print(model_config.up_direction)

    VoroTracingViewer(model=model, mode="rendering", up_direction=[0, 0, -1])
    print("Viewer running... Ctrl+C to exit.")
    time.sleep(100000)


if __name__ == "__main__":
    tyro.cli(run_viewer)
