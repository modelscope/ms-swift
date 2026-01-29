import os
from pathlib import Path
from omegaconf import OmegaConf
from swift.llm.dataset.register import _register_d_info
from swift.llm import register_dataset, DatasetMeta, ResponsePreprocessor


PROJ_DIR = Path(__file__).parents[1].resolve()
os.environ['PROJ_DIR'] = str(PROJ_DIR)

dataset_info = OmegaConf.load(PROJ_DIR.joinpath('randy/dataset_info.json'))
dataset_info = OmegaConf.to_container(dataset_info, resolve=True)

for info in dataset_info:
    _register_d_info(info)
