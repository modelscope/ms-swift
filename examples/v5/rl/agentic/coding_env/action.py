# Copyright (c) ModelScope Contributors. All rights reserved.
"""The Action the coding environment's ``run_command`` tool maps onto.

OpenEnv's default action handling passes the tool-call arguments straight to this
class's constructor, so the field names match the advertised tool schema.
"""
from dataclasses import dataclass
from typing import Optional


@dataclass
class RunCommandAction:
    """Execute ``command`` in the episode workspace (optionally under ``cwd``)."""

    command: str
    cwd: Optional[str] = None
