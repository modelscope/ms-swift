# Copyright (c) Alibaba, Inc. and its affiliates.
from contextlib import nullcontext
from typing import List, Union

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
            demo.queue().launch(server_name=args.server_name, server_port=args.server_port, share=args.share)


def app_main(args: Union[List[str], AppArguments, None] = None):
    return SwiftApp(args).main()
