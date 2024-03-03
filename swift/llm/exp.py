# Copyright (c) Alibaba, Inc. and its affiliates.
import os.path

from swift import snapshot_download
from swift.llm.utils import *
from swift.llm.utils.exp_utils import ExpManager
from swift.utils import *

logger = get_logger()


def find_all_config(dir_or_file: str):
    if os.path.isfile(dir_or_file) and dir_or_file.endswith('.json'):
        return [dir_or_file]
    elif os.path.isdir(dir_or_file):
        configs = []
        for dirpath, dirnames, filenames in os.walk(dir_or_file):
            for name in filenames:
                if name.endswith('.json'):
                    configs.append(os.path.join(dirpath, name))
        return configs
    else:
        dir_or_file = snapshot_download(dir_or_file)
        return [os.path.join(dir_or_file, 'experiment', 'config.json')]


def llm_exp(args: ExpArguments):
    config = args.config
    all_configs = []
    if not isinstance(config, list):
        config = [config]
    for dir_or_file in config:
        all_configs.extend(find_all_config(dir_or_file))
    args.config = all_configs
    exp_manager = ExpManager()
    exp_manager.begin(args)


exp_main = get_main(ExpArguments, llm_exp)
