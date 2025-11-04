# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import Dict

from swift.utils import get_logger
from ..main import cli_main as swift_cli_main

logger = get_logger()

ROUTE_MAPPING: Dict[str, str] = {
    'pt': 'swift.cli._megatron.pt',
    'sft': 'swift.cli._megatron.sft',
    'rlhf': 'swift.cli._megatron.rlhf',
    'export': 'swift.cli._megatron.export',
}


def cli_main():
    return swift_cli_main(ROUTE_MAPPING, is_megatron=True)


if __name__ == '__main__':
    cli_main()
