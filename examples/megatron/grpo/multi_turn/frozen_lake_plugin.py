# Copyright (c) ModelScope Contributors. All rights reserved.
"""Text-based FrozenLake env for multi-turn GRPO training.

Each turn the LLM sees an ASCII grid (S/G/H/F/P) and must reply with a
single move inside ``<action>...</action>``. Reaching G yields reward 1.0;
stepping into H ends the episode with 0. The outer step budget comes from
swift's ``--max_turns`` flag. Per-row ``env_config.seed`` controls map
generation so all ``num_generations`` rollouts of a row share the same map.

Register via ``--external_plugins`` and select with ``--gym_env frozen_lake``.
"""
# code borrowed from ROLL/roll/pipeline/agentic/env/frozen_lake

from __future__ import annotations

import random
import re
from typing import Any, Dict, List, Optional, Tuple

from swift.infer_engine.protocol import RolloutInferRequest
from swift.rollout.gym_env import Env, envs
from swift.template import Messages
from swift.utils import get_logger

logger = get_logger()


def _is_valid(board: List[List[str]], size: int) -> bool:
    """BFS from S; return True iff G is reachable through non-hole cells."""
    start = None
    for r in range(size):
        for c in range(size):
            if board[r][c] == 'S':
                start = (r, c)
                break
        if start is not None:
            break
    if start is None:
        return False

    frontier = [start]
    discovered = set()
    while frontier:
        r, c = frontier.pop()
        if (r, c) in discovered:
            continue
        discovered.add((r, c))
        for dr, dc in ((1, 0), (0, 1), (-1, 0), (0, -1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < size and 0 <= nc < size:
                cell = board[nr][nc]
                if cell == 'G':
                    return True
                if cell != 'H':
                    frontier.append((nr, nc))
    return False


def generate_random_map(size: int = 4, p: float = 0.8, seed: Optional[int] = None) -> List[str]:
    """Generate a random solvable FrozenLake map.

    Args:
        size: side length of the square grid.
        p: probability of a non-hole (F) cell — 1 - p of any cell being H.
        seed: RNG seed; same seed always yields the same map.

    Returns:
        List of `size` strings, each of length `size`, using chars S/G/F/H.
        Both S and G positions are randomised (unlike gym's ``FrozenLake-v1``
        which pins S to top-left and G to bottom-right); BFS-validated.
    """
    rng = random.Random(seed)
    while True:
        board = [['F' if rng.random() < p else 'H' for _ in range(size)] for _ in range(size)]
        start_r, start_c = rng.randrange(size), rng.randrange(size)
        goal_r, goal_c = rng.randrange(size), rng.randrange(size)
        if (start_r, start_c) == (goal_r, goal_c):
            continue
        board[start_r][start_c] = 'S'
        board[goal_r][goal_c] = 'G'
        if _is_valid(board, size):
            return [''.join(row) for row in board]


# (row_delta, col_delta) for each canonical action token.
ACTIONS: Dict[str, Tuple[int, int]] = {
    'up': (-1, 0),
    'down': (1, 0),
    'left': (0, -1),
    'right': (0, 1),
}

SYSTEM_PROMPT = ('You are playing FrozenLake. You see a grid where:\n'
                 '  P = your current position, S = start, G = goal, H = hole, F = safe ice.\n'
                 'Move one cell per turn. Reach G to win (+1 reward). Stepping into H ends '
                 'the episode with 0 reward. Moves that would go off the grid leave you in '
                 'place.\n\n'
                 'On every turn, output your move inside <action>...</action>. '
                 'The action must be exactly one of: up, down, left, right.\n\n'
                 'Example: <action>down</action>')

_ACTION_TAG_RE = re.compile(r'<action>\s*(up|down|left|right)\s*</action>', re.IGNORECASE)
_BARE_ACTION_RE = re.compile(r'\b(up|down|left|right)\b', re.IGNORECASE)


def _render(grid: List[str], row: int, col: int) -> str:
    """Render the grid with the player position marked as 'P'."""
    rendered = []
    for r, line in enumerate(grid):
        if r == row:
            chars = list(line)
            chars[col] = 'P' if chars[col] != 'G' else '*'  # '*' = player on goal
            rendered.append(' '.join(chars))
        else:
            rendered.append(' '.join(line))
    return '\n'.join(rendered)


def _parse_action(completion: str) -> Optional[str]:
    """Extract the action from the assistant message. Returns None if missing."""
    m = _ACTION_TAG_RE.search(completion)
    if m:
        return m.group(1).lower()
    matches = _BARE_ACTION_RE.findall(completion)
    if matches:
        return matches[-1].lower()
    return None


class FrozenLakeEnv(Env):

    def __init__(self, env_config: Dict[str, Any]):
        super().__init__(env_config)
        self.size: int = int(env_config.get('size', 4))
        self.p: float = float(env_config.get('p', 0.8))
        seed = env_config.get('seed')
        self.seed: Optional[int] = int(seed) if seed is not None else None
        self.grid: List[str] = []
        self.row: int = 0
        self.col: int = 0
        self.steps: int = 0

    async def reset(self, config: RolloutInferRequest) -> Tuple[str, Dict[str, Any], str]:
        self.grid = generate_random_map(size=self.size, p=self.p, seed=self.seed)
        for r, line in enumerate(self.grid):
            if 'S' in line:
                self.row, self.col = r, line.index('S')
                break
        self.steps = 0

        observation = (f'FrozenLake {self.size}x{self.size}:\n'
                       f'{_render(self.grid, self.row, self.col)}\n\n'
                       f'You are at row {self.row}, col {self.col}. Output your first move.')
        info = {'seed': self.seed, 'size': self.size}
        return observation, info, SYSTEM_PROMPT

    async def step(self, action: Messages) -> Tuple[str, float, bool, Dict[str, Any]]:
        completion = action[-1].get('content', '') if action else ''
        move = _parse_action(completion)
        self.steps += 1

        info: Dict[str, Any] = {'seed': self.seed, 'step': self.steps, 'parsed_action': move}

        if move is None:
            obs = (f'Invalid response: could not find a move. Reply with '
                   f'<action>up|down|left|right</action>.\n\n'
                   f'{_render(self.grid, self.row, self.col)}')
            return obs, 0.0, False, {**info, 'status': 'invalid_action'}

        dr, dc = ACTIONS[move]
        new_row = max(0, min(len(self.grid) - 1, self.row + dr))
        new_col = max(0, min(len(self.grid[0]) - 1, self.col + dc))
        cell = self.grid[new_row][new_col]
        self.row, self.col = new_row, new_col

        if cell == 'G':
            obs = (f'You moved {move} and reached the goal!\n'
                   f'{_render(self.grid, self.row, self.col)}')
            return obs, 1.0, True, {**info, 'status': 'goal'}
        if cell == 'H':
            obs = (f'You moved {move} and fell into a hole. Episode over.\n'
                   f'{_render(self.grid, self.row, self.col)}')
            return obs, 0.0, True, {**info, 'status': 'hole'}

        obs = (f'You moved {move}. Now at row {self.row}, col {self.col} (step {self.steps}).\n'
               f'{_render(self.grid, self.row, self.col)}\n'
               f'Output your next move.')
        return obs, 0.0, False, {**info, 'status': 'ok'}

    async def close(self):
        pass


envs['frozen_lake'] = FrozenLakeEnv
