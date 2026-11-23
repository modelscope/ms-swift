# Copyright (c) ModelScope Contributors. All rights reserved.
"""A minimal OpenEnv environment package for code-training rollouts.

Discovered by ``twinkle_agentic.envs.OpenEnv(env_name='coding_env')``: the package
top level exports the Action class, and ``server/terminal_environment.py`` carries
the Environment class (the two shapes OpenEnv's discovery looks for). The
environment gives the policy one tool -- ``run_command`` -- against a persistent
per-episode workspace, which is all a coding rollout needs (files are written
with heredocs and run with ``python file.py``).

Run a GRPO training with it by putting this directory on PYTHONPATH and passing
``--sandbox_kind openenv --sandbox_env_name coding_env``.
"""
from .action import RunCommandAction
from .server.terminal_environment import TerminalEnvironment

__all__ = ['RunCommandAction', 'TerminalEnvironment']
