# Copyright (c) Alibaba, Inc. and its affiliates.
import os.path

from swift import snapshot_download
from swift.llm.utils import *
from swift.llm.utils.exp_utils import ExpManager
from swift.utils import *

logger = get_logger()


def llm_exp(args: ExpArguments):
    exp_config = args.exp_config
    if not os.path.exists(exp_config):
        exp_config = snapshot_download(exp_config)
    if os.path.isfile(exp_config):
        exp_config = [exp_config]
    else:
        configs = []
        for dirpath, dirnames, filenames in os.walk(exp_config):
            for name in filenames:
                if name.endswith('.json'):
                    configs.append(os.path.join(dirpath, name))
        exp_config = configs
    exp_manager = ExpManager()
    exp_manager.begin(exp_config)


exp_main = get_main(ExpArguments, llm_exp)
