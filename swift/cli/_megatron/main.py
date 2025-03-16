# Copyright (c) Alibaba, Inc. and its affiliates.
import importlib.util
import os
import subprocess
import sys
from typing import Dict, List, Optional

from swift.utils import get_logger
from ..main import cli_main as swift_cli_main

logger = get_logger()

ROUTE_MAPPING: Dict[str, str] = {
    'sft': 'swift.cli._megatron.sft',
    'pt': 'swift.cli._megatron.pt',
}


def cli_main():
    return swift_cli_main(ROUTE_MAPPING)


if __name__ == '__main__':
    cli_main()
