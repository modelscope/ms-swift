# Copyright (c) Alibaba, Inc. and its affiliates.
from contextlib import nullcontext
from typing import List, Union

import gradio
from packaging import version

from swift.utils import get_logger
from ..argument import AppArguments
from ..base import SwiftPipeline
from ..infer import run_deploy
from .build_ui import build_ui

logger = get_logger()


class SwiftApp(SwiftPipeline):
    args_class = AppArguments
    args: args_class

    def run(self):
        args = self.args
        deploy_context = nullcontext() if args.base_url else run_deploy(args, return_url=True)
        with deploy_context as base_url:
            base_url = base_url or args.base_url
            demo = build_ui(
                base_url,
                args.model_suffix,
                request_config=args.get_request_config(),
                is_multimodal=args.is_multimodal,
                studio_title=args.studio_title,
                lang=args.lang,
                default_system=args.system)
            concurrency_count = 1 if args.infer_backend == 'pt' else 16
            if version.parse(gradio.__version__) < version.parse('4'):
                queue_kwargs = {'concurrency_count': concurrency_count}
            else:
                queue_kwargs = {'default_concurrency_limit': concurrency_count}
            demo.queue(**queue_kwargs).launch(
                server_name=args.server_name, server_port=args.server_port, share=args.share)


def app_main(args: Union[List[str], AppArguments, None] = None):
    return SwiftApp(args).main()
