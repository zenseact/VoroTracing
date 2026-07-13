import numpy as np
import torch
import tyro

from vorotracing.config.train_configs import TrainConfigList

seed = 42
torch.random.manual_seed(seed)
np.random.seed(seed)


def main():
    config = tyro.cli(TrainConfigList)
    trainer = config.setup()
    trainer.train()


if __name__ == "__main__":
    main()
