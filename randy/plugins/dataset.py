import os
from pathlib import Path
from omegaconf import OmegaConf
from swift.dataset.register import _register_d_info


PROJ_DIR = Path(__file__).parents[2].resolve()
WORK_DIR = PROJ_DIR.parents[1].resolve()
os.environ['WORK_DIR'] = str(WORK_DIR)

dataset_info = OmegaConf.load(PROJ_DIR.joinpath('randy/dataset_info.json'))
dataset_info = OmegaConf.to_container(dataset_info, resolve=True)

for info in dataset_info:
    _register_d_info(info)
