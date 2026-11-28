# Copyright (c) ModelScope Contributors. All rights reserved.
from typing import Dict

from swift.cli.main import cli_main as swift_cli_main

ROUTE_MAPPING: Dict[str, str] = {
    'sft': 'swift.cli._fsdpturbo.sft',
}


def cli_main():
    return swift_cli_main(ROUTE_MAPPING, torchrun_all_methods=True)


if __name__ == '__main__':
    cli_main()
