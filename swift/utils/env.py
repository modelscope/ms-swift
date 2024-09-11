import os

from transformers.utils import strtobool


def use_hf_hub():
    return strtobool(os.environ.get('USE_HF', 'False'))
