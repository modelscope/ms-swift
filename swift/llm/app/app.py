# Copyright (c) Alibaba, Inc. and its affiliates.
import os
from contextlib import nullcontext
from typing import List, Union

from swift.utils import get_logger
from ..argument import AppArguments
from ..base import SwiftPipeline
from ..infer import run_deploy

logger = get_logger()


class SwiftApp(SwiftPipeline):
    args_class = AppArguments
    args: args_class

    def run(self):
        args = self.args
        deploy_context = nullcontext() if args.eval_url else run_deploy(self.args, return_url=True)
