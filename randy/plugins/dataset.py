import os
from pathlib import Path
from omegaconf import OmegaConf
from swift.dataset import DatasetMeta, ResponsePreprocessor, register_dataset


class CaptionPreprocessor(ResponsePreprocessor):
    def preprocess(self, row):
        query = '<image>'
        response = row.pop('caption', None)
        row.update({
            'query': query,
            'response': response,
        })
        return super().preprocess(row)


PREPROCESSOR_MAP = {
    'CaptionPreprocessor': CaptionPreprocessor,
}

PROJ_DIR = Path(__file__).parents[2].resolve()
WORK_DIR = PROJ_DIR.parents[1].resolve()
os.environ['WORK_DIR'] = str(WORK_DIR)

dataset_info = OmegaConf.load(PROJ_DIR.joinpath('randy/dataset_info.json'))
dataset_info = OmegaConf.to_container(dataset_info, resolve=True)

for info in dataset_info:
    if 'preprocess_func' in info:
        info['preprocess_func'] = PREPROCESSOR_MAP[info['preprocess_func']]()
    register_dataset(DatasetMeta(**info))
