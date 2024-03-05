# Copyright (c) Alibaba, Inc. and its affiliates.
import os.path
import os
from swift.llm.utils import *
from swift.llm.utils.exp_utils import ExpManager
from swift.utils import *

logger = get_logger()


def find_all_config(dir_or_file: str):
    if os.path.isfile(dir_or_file):
        return [dir_or_file]
    else:
        configs = []
        for dirpath, dirnames, filenames in os.walk(dir_or_file):
            for name in filenames:
                if name.endswith('.json'):
                    configs.append(os.path.join(dirpath, name))
        return configs


def llm_exp(args: ExpArguments):
    config = args.config
    os.makedirs(args.save_dir, exist_ok=True)
    all_configs = []
    if not isinstance(config, list):
        config = [config]
    for dir_or_file in config:
        all_configs.extend(find_all_config(dir_or_file))
    args.config = all_configs
    exp_manager = ExpManager()
    exp_manager.begin(args)


exp_main = get_main(ExpArguments, llm_exp)
