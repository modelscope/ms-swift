"""Sudoku scheduler for OpenEnv TextArena Sudoku environment.

Reference: TRL openenv_sudoku_grpo.ipynb
Key features:
1. Multiple reward functions (empty_cell, valid_move, repetition, progress, correct)
2. Hints system: parse board, provide guaranteed moves and candidates
3. Board state tracking with content diff for bounded context
"""
import re
from collections import defaultdict
from typing import Any, Dict, List, Union

from swift.rollout.multi_turn import OpenEnvScheduler


SUDOKU_SYSTEM_PROMPT = """You are an expert Sudoku player with deep knowledge of logical deduction strategies.

## GAME RULES
1. The puzzle is a 9x9 grid divided into nine 3x3 subgrids (boxes)
2. Some cells are pre-filled with numbers 1-9
3. Fill empty cells ('.') with numbers 1-9
4. Each row, column, and 3x3 box must contain 1-9 without repetition
5. Cannot overwrite pre-filled cells
6. Invalid moves result in penalties

## HOW TO PLAY
Output your move in this format: [row col number]
Example: [3 5 7] means place 7 at row 3, column 5.
You may reason before your move, but always end with [row col number].

## STRATEGIC APPROACH
- Naked Singles: If a cell has only one possible candidate, fill it immediately.
- Hidden Singles: If a number can only go in one cell within a row/column/box, place it there.
- Scanning: Look at each row, column, and box to find where numbers can go.

## COMMON PITFALLS
- Don't guess randomly - Sudoku is pure logic
- Don't overwrite pre-filled cells
- Don't repeat a move that was already made
- Coordinates are 1-indexed (1-9)

## BOARD READING
- Rows labeled R1-R9 (top to bottom)
- Columns labeled C1-C9 (left to right)
- Empty cells shown as '.'"""


# ── Board parsing utilities (from TRL example) ──

def _is_valid_board_state(board_str: str) -> bool:
    return "R1" in board_str and "R9" in board_str and "|" in board_str


def _parse_board(board_str: str) -> list:
    grid = [[0] * 9 for _ in range(9)]
    if not _is_valid_board_state(board_str):
        return grid
    for line in board_str.split("\n"):
        line_stripped = line.strip()
        if line_stripped and line_stripped[0] == "R" and len(line_stripped) > 1 and line_stripped[1].isdigit():
            row = int(line_stripped[1]) - 1
            col = 0
            for char in line_stripped[2:]:
                if char == ".":
                    grid[row][col] = 0
                    col += 1
                elif char.isdigit():
                    grid[row][col] = int(char)
                    col += 1
    return grid


def _count_filled_cells(board_str: str) -> int:
    grid = _parse_board(board_str)
    return sum(1 for row in grid for cell in row if cell != 0)


def _get_valid_numbers(grid: list, row: int, col: int) -> set:
    if grid[row][col] != 0:
        return set()
    used = set()
    for c in range(9):
        if grid[row][c] != 0:
            used.add(grid[row][c])
    for r in range(9):
        if grid[r][col] != 0:
            used.add(grid[r][col])
    box_row, box_col = 3 * (row // 3), 3 * (col // 3)
    for r in range(box_row, box_row + 3):
        for c in range(box_col, box_col + 3):
            if grid[r][c] != 0:
                used.add(grid[r][c])
    return set(range(1, 10)) - used


def _extract_empty_cells_with_candidates(board_str: str, sort_by_difficulty: bool = True):
    grid = _parse_board(board_str)
    cells = []
    for row in range(9):
        for col in range(9):
            if grid[row][col] == 0:
                candidates = _get_valid_numbers(grid, row, col)
                cells.append((row + 1, col + 1, candidates))
    if sort_by_difficulty:
        cells.sort(key=lambda x: len(x[2]))
    return cells


def _extract_empty_cells(board_str: str) -> list:
    empty_cells = []
    if not _is_valid_board_state(board_str):
        return empty_cells
    for line in board_str.split("\n"):
        line_stripped = line.strip()
        if line_stripped and line_stripped[0] == "R" and len(line_stripped) > 1 and line_stripped[1].isdigit():
            row = int(line_stripped[1])
            col = 0
            for char in line_stripped[2:]:
                if char == ".":
                    col += 1
                    empty_cells.append((row, col))
                elif char.isdigit():
                    col += 1
    return empty_cells


def _extract_board_only(text: str) -> str:
    if not text:
        return ""
    lines = text.split("\n")
    board_lines = []
    in_board = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("C1") or (
            stripped and stripped[0] == "R" and len(stripped) > 1 and stripped[1].isdigit()
        ):
            in_board = True
        if in_board and (stripped.startswith("-") or stripped.startswith("R") or stripped.startswith("C1")):
            board_lines.append(line)
        elif (
            in_board
            and stripped
            and not stripped.startswith("-")
            and not (stripped[0] == "R" and len(stripped) > 1 and stripped[1].isdigit())
        ):
            break
    return "\n".join(board_lines) if board_lines else ""


def _make_hints(board_str: str, successful_moves: list, failed_moves: list, difficulty: str = "easy") -> str:
    """Generate hint text for the model (from TRL example)."""
    parts = []
    all_tried = successful_moves + failed_moves
    if all_tried:
        parts.append(f"\nMOVES ALREADY TRIED (do not repeat): {', '.join(all_tried[:10])}")
    if not board_str or not _is_valid_board_state(board_str):
        return "\n".join(parts)

    cells = _extract_empty_cells_with_candidates(board_str, sort_by_difficulty=True)
    if cells:
        guaranteed = []
        other = []
        for r, c, candidates in cells[:10]:
            if len(candidates) == 1:
                guaranteed.append(f"[{r} {c} {list(candidates)[0]}]")
            elif len(candidates) <= 3:
                nums = ",".join(str(n) for n in sorted(candidates))
                other.append(f"({r},{c})->{nums}")
        if guaranteed:
            parts.append(f"\nGUARANTEED MOVES (only one option): {', '.join(guaranteed[:5])}")
        if other:
            parts.append(f"Other options: {' | '.join(other[:5])}")

    return "\n".join(parts)


class SudokuScheduler(OpenEnvScheduler):
    """Sudoku scheduler with multi-reward and hints system.

    Tracks 5 reward components per trajectory:
    - empty_cell_reward: Did the model target empty cells? (+1/-1)
    - valid_move_reward: Were moves accepted by env? (1.0/-0.5/0.0)
    - repetition_reward: Penalty for repeating moves (exponential)
    - progress_reward: How much of the puzzle was filled (0-1)
    - correct_reward: Environment's reward (0 or 1)

    Combined reward = sum of all components.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._last_content_len: Dict[str, int] = {}
        # Per-uuid state tracking
        self._board_states: Dict[str, str] = {}
        self._move_counts: Dict[str, defaultdict] = {}
        self._successful_moves: Dict[str, list] = {}
        self._failed_moves: Dict[str, list] = {}
        self._valid_move_scores: Dict[str, list] = {}
        self._empty_cell_scores: Dict[str, list] = {}
        self._correct_scores: Dict[str, list] = {}
        self._repetition_scores: Dict[str, list] = {}
        self._initial_filled: Dict[str, int] = {}
        self._max_filled: Dict[str, int] = {}

    async def on_trajectory_start(self, requests):
        """Initialize env, parse board, compute hints."""
        import asyncio
        semaphore = asyncio.Semaphore(getattr(self, 'max_concurrent_envs', 4))

        async def _init_single(req):
            async with semaphore:
                uuid = req.uuid
                if uuid in self._envs:
                    await self._close_and_remove(uuid)

                row_env_config = (req.data_dict or {}).get('env_config', {}) if hasattr(req, 'data_dict') else {}
                env_config = {**getattr(self, 'env_config_defaults', {}), **row_env_config}
                wrapper = self._create_env(env_config)

                obs, metadata = wrapper.reset()
                system_message = env_config.get('system_message', SUDOKU_SYSTEM_PROMPT)

                content = self._extract_content(obs)
                self._last_content_len[uuid] = len(content)

                # Parse initial board state
                board = _extract_board_only(content) if _is_valid_board_state(content) else content
                self._board_states[uuid] = content if _is_valid_board_state(content) else ""
                initial_filled = _count_filled_cells(self._board_states[uuid]) if self._board_states[uuid] else 0

                # Initialize tracking state
                self._move_counts[uuid] = defaultdict(int)
                self._successful_moves[uuid] = []
                self._failed_moves[uuid] = []
                self._valid_move_scores[uuid] = []
                self._empty_cell_scores[uuid] = []
                self._correct_scores[uuid] = []
                self._repetition_scores[uuid] = []
                self._initial_filled[uuid] = initial_filled
                self._max_filled[uuid] = initial_filled

                # Build initial message with board + hints
                hints = _make_hints(self._board_states[uuid], [], [])
                user_content = f"{board}{hints}" if board else content

                from swift.rollout.multi_turn import Messages
                messages = []
                if system_message:
                    messages.append({'role': 'system', 'content': system_message})
                messages.append({'role': 'user', 'content': user_content})
                req.messages = messages

                self._envs[uuid] = wrapper
                self._total_rewards[uuid] = 0.0
                self._step_rewards[uuid] = []
                self._pending_obs[uuid] = None

        await asyncio.gather(*[_init_single(req) for req in requests])

    async def _close_and_remove(self, uuid):
        """Override to clean up all tracking state."""
        await super()._close_and_remove(uuid)
        self._last_content_len.pop(uuid, None)
        self._board_states.pop(uuid, None)
        self._move_counts.pop(uuid, None)
        self._successful_moves.pop(uuid, None)
        self._failed_moves.pop(uuid, None)
        self._valid_move_scores.pop(uuid, None)
        self._empty_cell_scores.pop(uuid, None)
        self._correct_scores.pop(uuid, None)
        self._repetition_scores.pop(uuid, None)
        self._initial_filled.pop(uuid, None)
        self._max_filled.pop(uuid, None)

    def _extract_content(self, observation: Any) -> str:
        if isinstance(observation, dict):
            messages = observation.get('messages', [])
            if messages:
                return messages[0].get('content', '')
            prompt = observation.get('prompt', '')
            if prompt:
                return prompt
        return str(observation)

    async def on_turn_end(self, infer_request, response_choice, current_turn):
        """Parse move, step env, compute multi-reward, generate hints."""
        uuid = infer_request.uuid
        wrapper = self._envs.get(uuid)
        if wrapper is None:
            return {'done': True, 'rollout_infos': {}}

        action_text = response_choice.message.content
        action_dict = self.parse_action(action_text)
        move = action_dict.get('message', '[1 1 1]')

        # Step environment
        obs, env_reward, done, metadata = wrapper.step(action_dict)
        correct_score = float(env_reward or 0.0)

        # Extract new content (diff from last seen)
        full_content = self._extract_content(obs)
        last_len = self._last_content_len.get(uuid, 0)
        new_content = full_content[last_len:] if len(full_content) > last_len else full_content
        self._last_content_len[uuid] = len(full_content)

        # ── Compute reward components (from TRL example) ──

        # Check if env says invalid
        new_content_lower = new_content.lower()
        env_says_invalid = any(
            kw in new_content_lower for kw in ["invalid", "error", "cannot", "already", "violation", "lost"]
        )

        # Check if move targets an empty cell
        if self._board_states.get(uuid):
            empty_cells = _extract_empty_cells(self._board_states[uuid])
            targets_empty = tuple(int(x) for x in re.findall(r'\d+', move)[:3]) in [(r, c) for r, c in empty_cells] if len(re.findall(r'\d+', move)) >= 3 else True
        else:
            targets_empty = True

        # Empty cell reward: +1 if targeted empty, -1 if tried to overwrite
        empty_cell_score = 1.0 if targets_empty else -1.0

        # Repetition tracking
        is_new_move = self._move_counts[uuid][move] == 0
        repetition_count = self._move_counts[uuid][move]
        self._move_counts[uuid][move] += 1
        repetition_score = -min(2 ** repetition_count, 10.0) if repetition_count > 0 else 0.0

        # Valid move score
        is_valid = not env_says_invalid and targets_empty
        if is_valid and is_new_move:
            valid_move_score = 1.0
            self._successful_moves[uuid].append(move)
        elif "please resubmit" in new_content_lower or "avoid penalties" in new_content_lower:
            valid_move_score = -0.5
            self._failed_moves[uuid].append(move)
        else:
            valid_move_score = 0.0
            if not is_valid:
                self._failed_moves[uuid].append(move)

        # Update board state if valid and new content has board
        if is_valid and _is_valid_board_state(new_content):
            self._board_states[uuid] = new_content
            current_filled = _count_filled_cells(new_content)
            if current_filled > self._max_filled[uuid]:
                self._max_filled[uuid] = current_filled

        # Progress reward
        remaining = 81 - self._initial_filled[uuid]
        if remaining > 0:
            progress_score = (self._max_filled[uuid] - self._initial_filled[uuid]) / remaining
        else:
            progress_score = 1.0

        # Track all scores
        self._valid_move_scores[uuid].append(valid_move_score)
        self._empty_cell_scores[uuid].append(empty_cell_score)
        self._correct_scores[uuid].append(correct_score)
        self._repetition_scores[uuid].append(repetition_score)

        # Combined reward (sum of all components, matching TRL's nansum approach)
        combined_reward = (
            sum(self._empty_cell_scores[uuid]) / max(len(self._empty_cell_scores[uuid]), 1)
            + sum(self._valid_move_scores[uuid]) / max(len(self._valid_move_scores[uuid]), 1)
            + sum(self._repetition_scores[uuid]) / max(len(self._repetition_scores[uuid]), 1)
            + progress_score
            + correct_score
        )

        self._total_rewards[uuid] = combined_reward
        self._step_rewards.setdefault(uuid, []).append(combined_reward)

        # Build next observation with board + hints
        if not done:
            board_str = self._board_states.get(uuid, "")
            board = _extract_board_only(board_str) if board_str else ""
            hints = _make_hints(
                board_str,
                self._successful_moves[uuid],
                self._failed_moves[uuid],
            )
            step_num = len(self._successful_moves[uuid])
            next_obs = f"Step {step_num}. Progress: {step_num} cells filled.\n\nBoard:\n{board}{hints}"
        else:
            next_obs = None

        self._pending_obs[uuid] = next_obs

        rollout_infos = {
            'total_reward': self._total_rewards[uuid],
            'step_rewards': list(self._step_rewards.get(uuid, [])),
            'gym_done': done,
            'empty_cell_reward': sum(self._empty_cell_scores[uuid]) / max(len(self._empty_cell_scores[uuid]), 1),
            'valid_move_reward': sum(self._valid_move_scores[uuid]) / max(len(self._valid_move_scores[uuid]), 1),
            'repetition_reward': sum(self._repetition_scores[uuid]) / max(len(self._repetition_scores[uuid]), 1),
            'progress_reward': progress_score,
            'correct_reward': correct_score,
        }
        if done:
            await self._close_and_remove(uuid)

        return {'done': done, 'rollout_infos': rollout_infos}

    def parse_action(self, text: str) -> Dict[str, Any]:
        """Extract [row col number] from model output."""
        match = re.search(r'\[\s*(\d+)\s+(\d+)\s+(\d+)\s*\]', text)
        if match:
            row, col, num = match.groups()
            return {"message": f"[{row} {col} {num}]"}

        numbers = re.findall(r'\d+', text)
        if len(numbers) >= 3:
            return {"message": f"[{numbers[0]} {numbers[1]} {numbers[2]}]"}

        return {"message": "[1 1 1]"}

    def format_observation(self, observation: Any) -> Union[str, List[Dict]]:
        return self._extract_content(observation)
"""Sudoku scheduler for OpenEnv TextArena sudoku environment.

The model sees the Sudoku board and outputs moves in [row col number] format.
Multi-turn: each turn the model places one number on the board.

Optimization: only sends the latest board state + feedback to the model,
not the full cumulative history, to keep context length manageable.
"""
import re
from typing import Any, Dict, Union, List

from swift.rollout.multi_turn import OpenEnvScheduler


SUDOKU_SYSTEM_PROMPT = """You are an expert Sudoku solver. You play Sudoku by placing numbers on the board.

Rules:
- The board is a 9x9 grid. Rows 1-9, Columns 1-9.
- Empty cells are '.', filled cells contain digits 1-9.
- Each row, column, and 3x3 box must contain digits 1-9 without repetition.

To make a move, output ONLY the move in this exact format:
[row col number]

Example: [3 5 7] means place 7 at row 3, column 5.
Do not output anything else. Just the move in brackets."""


class SudokuScheduler(OpenEnvScheduler):
    """Scheduler for OpenEnv TextArena Sudoku environment.

    Tracks the last seen content length per uuid to extract only new feedback
    from the cumulative messages, keeping context length bounded.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._last_content_len: Dict[str, int] = {}

    async def on_trajectory_start(self, requests):
        """Reset tracking state for each trajectory."""
        import asyncio
        semaphore = asyncio.Semaphore(getattr(self, 'max_concurrent_envs', 4))

        async def _init_single(req):
            async with semaphore:
                uuid = req.uuid
                if uuid in self._envs:
                    await self._close_and_remove(uuid)

                row_env_config = (req.data_dict or {}).get('env_config', {}) if hasattr(req, 'data_dict') else {}
                env_config = {**getattr(self, 'env_config_defaults', {}), **row_env_config}
                wrapper = self._create_env(env_config)

                obs, metadata = wrapper.reset()
                system_message = env_config.get('system_message', '')

                # Track initial content length
                content = self._extract_content(obs)
                self._last_content_len[uuid] = len(content)

                from swift.rollout.multi_turn import Messages
                messages = []
                if system_message:
                    messages.append({'role': 'system', 'content': system_message})
                messages.append({'role': 'user', 'content': content})
                req.messages = messages

                self._envs[uuid] = wrapper
                self._total_rewards[uuid] = 0.0
                self._step_rewards[uuid] = []
                self._pending_obs[uuid] = None

        await asyncio.gather(*[_init_single(req) for req in requests])

    async def _close_and_remove(self, uuid):
        """Override to also clean up tracking state."""
        await super()._close_and_remove(uuid)
        self._last_content_len.pop(uuid, None)

    def _extract_content(self, observation: Any) -> str:
        """Extract the text content from a TextArena observation."""
        if isinstance(observation, dict):
            messages = observation.get('messages', [])
            if messages:
                return messages[0].get('content', '')
            prompt = observation.get('prompt', '')
            if prompt:
                return prompt
        return str(observation)

    async def on_turn_end(self, infer_request, response_choice, current_turn):
        """Parse model output, step env, and track content for diff."""
        uuid = infer_request.uuid
        wrapper = self._envs.get(uuid)
        if wrapper is None:
            return {'done': True, 'rollout_infos': {}}

        action_text = response_choice.message.content
        action_dict = self.parse_action(action_text)
        obs, reward, done, metadata = wrapper.step(action_dict)

        self._total_rewards[uuid] = self._total_rewards.get(uuid, 0.0) + float(reward)
        self._step_rewards.setdefault(uuid, []).append(float(reward))

        # Extract only NEW content (diff from last seen)
        full_content = self._extract_content(obs)
        last_len = self._last_content_len.get(uuid, 0)
        new_content = full_content[last_len:] if len(full_content) > last_len else full_content
        self._last_content_len[uuid] = len(full_content)

        next_obs = None if done else new_content
        self._pending_obs[uuid] = next_obs

        rollout_infos = {
            'total_reward': self._total_rewards[uuid],
            'step_rewards': list(self._step_rewards.get(uuid, [])),
            'gym_done': done,
        }
        if done:
            await self._close_and_remove(uuid)

        return {'done': done, 'rollout_infos': rollout_infos}

    def parse_action(self, text: str) -> Dict[str, Any]:
        """Extract [row col number] from model output."""
        match = re.search(r'\[\s*(\d+)\s+(\d+)\s+(\d+)\s*\]', text)
        if match:
            row, col, num = match.groups()
            return {"message": f"[{row} {col} {num}]"}

        numbers = re.findall(r'\d+', text)
        if len(numbers) >= 3:
            return {"message": f"[{numbers[0]} {numbers[1]} {numbers[2]}]"}

        return {"message": "[1 1 1]"}

    def format_observation(self, observation: Any) -> Union[str, List[Dict]]:
        """Format observation (used only for initial state in on_trajectory_start)."""
        return self._extract_content(observation)


# Register scheduler so --external_plugins can load it
from swift.rollout.multi_turn import multi_turns
multi_turns['sudoku_scheduler'] = SudokuScheduler
