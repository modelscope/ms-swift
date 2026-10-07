# Copyright (c) ModelScope Contributors. All rights reserved.
import gradio
import sys
from contextlib import nullcontext
from packaging import version
from typing import List, Optional, Union

from swift.arguments import AppArguments
from swift.utils import get_logger
from ..base import SwiftPipeline
from ..infer import run_deploy
from .build_ui import build_ui

logger = get_logger()

# Addresses that expose the Web UI to the network, making it accessible to
# anyone who can reach the server. Binding to these addresses without
# authentication is dangerous — see GHSA-9g2v-fgfh-65rx.
_UNSAFE_BIND_ADDRESSES = {'0.0.0.0', '::', '[::]'}

_SECURITY_WARNING = ('⚠️ SECURITY WARNING: The Web UI is bound to {addr!r} and will be accessible to '
                     'anyone on the network. The ms-swift Web UI has no built-in authentication and allows '
                     'executing arbitrary commands on the server. This can lead to Remote Code Execution (RCE). '
                     'If you do not need external access, use --server_name 127.0.0.1 (the default). '
                     'If you must expose the Web UI, protect it with a reverse proxy, VPN, or firewall rules.')


class SwiftApp(SwiftPipeline):
    args_class = AppArguments
    args: args_class

    def run(self):
        args = self.args
        if args.server_name in _UNSAFE_BIND_ADDRESSES:
            logger.warning(_SECURITY_WARNING.format(addr=args.server_name))
            print(_SECURITY_WARNING.format(addr=args.server_name), file=sys.stderr)
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
            concurrency_count = 1 if args.infer_backend == 'transformers' else 16
            if version.parse(gradio.__version__) < version.parse('4'):
                queue_kwargs = {'concurrency_count': concurrency_count}
            else:
                queue_kwargs = {'default_concurrency_limit': concurrency_count}
            demo.queue(**queue_kwargs).launch(
                server_name=args.server_name, server_port=args.server_port, share=args.share)


def app_main(args: Optional[Union[List[str], AppArguments]] = None):
    return SwiftApp(args).main()
