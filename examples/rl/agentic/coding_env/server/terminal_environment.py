# Copyright (c) ModelScope Contributors. All rights reserved.
"""The coding environment: a persistent per-episode workspace plus ``run_command``.

A subprocess under ``bash -c`` with a timeout and an address-space cap, mirroring
``LocalEnv``'s isolation posture (a cap on runaway scripts, not on hostile ones --
trusted-code policy lives with the run, not the env). ``reset`` clears the
workspace so each episode starts clean; ``step`` runs one command and returns its
combined output, truncated to keep one verbose command from eating the episode's
context. The episode never ends from the environment side (``done=False``): turn
count is the rollout's ``max_turns`` knob.
"""
import os
import shlex
import subprocess
import tempfile
from typing import Any, Dict, List, Optional

MAX_OBSERVATION_CHARS = 8 * 1024

#: The one tool this environment advertises; names and argument spellings follow
#: ``twinkle_agentic.envs.base.DEFAULT_TOOLS`` so a trajectory built against the
#: sandbox replays here unchanged.
RUN_COMMAND_TOOL = {
    'type': 'function',
    'function': {
        'name': 'run_command',
        'description': 'Run a shell command in the episode workspace and return its output.',
        'parameters': {
            'type': 'object',
            'properties': {
                'command': {'type': 'string', 'description': 'The shell command to execute.'},
                'cwd': {'type': 'string', 'description': 'Working directory (optional).'},
            },
            'required': ['command'],
        },
    },
}


class TerminalEnvironment:
    """The Environment class OpenEnv's discovery resolves for ``coding_env``."""

    def __init__(self, workspace: Optional[str] = None, command_timeout: int = 60,
                 memory_limit_gb: Optional[float] = 2.0):
        self._workspace = workspace or tempfile.mkdtemp(prefix='coding_env_')
        os.makedirs(self._workspace, exist_ok=True)
        self._command_timeout = command_timeout
        self._memory_limit_gb = memory_limit_gb

    @property
    def workspace(self) -> str:
        return self._workspace

    def reset(self) -> Dict[str, Any]:
        for name in os.listdir(self._workspace):
            path = os.path.join(self._workspace, name)
            if os.path.isdir(path) and not os.path.islink(path):
                subprocess.run(['rm', '-rf', '--', path], check=False)
            else:
                os.remove(path)
        return {
            'observation': f'Workspace ready: {self._workspace}',
            'reward': 0.0,
            'done': False,
        }

    def step(self, action) -> Dict[str, Any]:
        if isinstance(action, dict):
            command = action.get('command') or ''
            cwd = action.get('cwd')
        else:
            command = getattr(action, 'command', '') or ''
            cwd = getattr(action, 'cwd', None)
        if not command:
            return {'observation': 'No command given.', 'reward': 0.0, 'done': False}
        cwd = os.path.join(self._workspace, cwd) if cwd else self._workspace
        preexec = _address_space_cap(self._memory_limit_gb)
        try:
            proc = subprocess.run(
                ['bash', '-c', command], cwd=cwd, capture_output=True, text=True,
                timeout=self._command_timeout, preexec_fn=preexec)
            observation = (proc.stdout or '') + (('\n[stderr]\n' + proc.stderr) if proc.stderr else '')
            observation = observation.strip()[:MAX_OBSERVATION_CHARS] or \
                f'(no output, exit code {proc.returncode})'
        except subprocess.TimeoutExpired:
            observation = f'Command timed out after {self._command_timeout}s.'
        except Exception as exc:  # noqa -- the observation is the feedback, not a crash
            observation = f'Command failed to start: {exc}'
        return {'observation': observation, 'reward': 0.0, 'done': False}

    def tools(self) -> List[Dict[str, Any]]:
        return [RUN_COMMAND_TOOL]

    def close(self) -> None:
        pass

    def __repr__(self) -> str:
        return f'TerminalEnvironment(workspace={shlex.quote(self._workspace)!r})'


def _address_space_cap(memory_limit_gb: Optional[float]):
    if memory_limit_gb is None:
        return None

    def cap() -> None:
        import resource
        limit = int(memory_limit_gb * 1024 ** 3)
        resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
    return cap
