# Copyright (c) ModelScope Contributors. All rights reserved.
"""Interactive terminal chat for ``swift infer`` -- the no-dataset REPL, with optional tool calling.

This is the dev counterpart of legacy ``--eval_human true``. Two generation paths share one loop:

* No tools: each human turn is one ``sampler`` call, streamed token-by-token when ``stream`` is on
  (``sample_stream``) or printed whole otherwise. This is the fast, low-latency chat path.
* Tools configured (``RolloutConfig.tools`` plus ``max_turns``): each human turn drives one
  ``RolloutEngine.generate``, and twinkle's ``MultiTurnRollout`` runs the whole intra-turn loop -- the
  model calls tools, reads their results, and continues until it stops answering with tools or hits
  ``max_turns``. That loop is batch-oriented, so it returns the finished trajectory rather than streaming
  tokens; the response is printed whole and the tools it invoked are traced afterwards.

The UX (colors, the ``You:``/``Agent:`` prompts, the dim tool trace, bounded history) borrows
``twinkle_client/auto``; the tool execution does NOT -- it runs dev-native through the sandbox env pool
and swift's tool plugins, not auto's OpenAI-compatible training-control tools.
"""
from __future__ import annotations
from dataclasses import asdict
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, Sequence

if TYPE_CHECKING:
    from swift.dev.config import (
        GenerationConfig,
        ModelConfig,
        QuantizeConfig,
        RLHFConfig,
        RolloutConfig,
        TemplateConfig,
    )

# ANSI colors, mirroring auto/app.py so the two terminal tools feel alike.
_CYAN = '\033[1;36m'
_GREEN = '\033[1;32m'
_YELLOW = '\033[1;33m'
_DIM = '\033[2m'
_RESET = '\033[0m'

#: Keep the conversation bounded; pruning cuts at a user boundary so a turn is never left half-dropped.
_MAX_HISTORY_MESSAGES = 50

#: Sentinel returned by ``_CliState.read_query`` for 'quit', kept distinct from an empty line (which means
#: "reprompt") and from None (a command that was already handled).
_QUIT = object()


def infer_cli(
    model_config: ModelConfig,
    template_config: TemplateConfig,
    generation_config: Optional[GenerationConfig] = None,
    *,
    backend: Literal['vllm', 'sglang', 'transformers'] = 'vllm',
    engine_args: Optional[Dict[str, Any]] = None,
    adapters: Optional[List[str]] = None,
    quantize_config: Optional[QuantizeConfig] = None,
    multi_round: bool = True,
    rollout_config: Optional[RolloutConfig] = None,
    multi_turn_config: Optional[RLHFConfig] = None,
) -> None:
    """Interactive REPL. Commands: ``clear`` / ``reset-system`` / ``multi-line`` / ``single-line`` / ``quit``.

    History is kept across turns unless ``multi_round`` is False. When ``rollout_config.tools`` and
    ``multi_turn_config.max_turns`` are both set, each turn runs the tool-calling rollout; otherwise it is
    a plain chat turn (streamed when ``generation_config.stream``). ``prompt_media`` is available for a
    template that asks for multimodal inputs, matching legacy's ``input_mm_data``.
    """
    from swift.dev.builders import build_sampler, build_template, load_model_processor, to_sampling_params

    _, processor = load_model_processor(model_config)
    template = build_template(template_config, processor)
    sampler = build_sampler(
        model_config,
        backend=backend,
        engine_args=engine_args,
        template=template,
        adapters=adapters,
        quantize_config=quantize_config)
    adapter_path = adapters[0] if adapters else None
    params = to_sampling_params(generation_config)
    params_dict = asdict(params)
    stream = bool(generation_config is not None and generation_config.stream)

    # Tools need twinkle's multi-turn engine, which requires max_turns; validate already rejects tools
    # without it, so this gate is just "did the user actually configure a tool rollout".
    tools_on = bool(rollout_config is not None and rollout_config.tools and multi_turn_config is not None
                    and multi_turn_config.max_turns is not None)
    rollout = None
    if tools_on:
        # The injected sampler stays owned by this function, so teardown is close() (the sandbox env pool)
        # followed by sampler.shutdown(); build_tool_sandbox resolves the tool plugins and leases one env
        # per episode, which for the REPL is the single interactive conversation.
        from swift.dev.rollout import RolloutEngine
        from swift.dev.rollout.sandbox import build_tool_sandbox
        rollout = RolloutEngine(sampler=sampler, template=template)
        env_pool, tool_plugins = build_tool_sandbox(rollout_config)
        rollout.configure_multi_turn(
            max_turns=multi_turn_config.max_turns,
            max_trajectory_tokens=multi_turn_config.max_trajectory_tokens,
            env_pool=env_pool,
            tool_plugins=tool_plugins)

    state = _CliState(system=template_config.system)
    _print_welcome(tools_on, stream)
    try:
        while True:
            try:
                query = state.read_query()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if query is None:
                continue
            if query is _QUIT:
                break

            state.add_query(query)
            trajectory = state.to_trajectory()
            if rollout is not None:
                response = _run_tool_turn(rollout, trajectory, params_dict)
            elif stream:
                response = _run_stream_turn(sampler, trajectory, params, adapter_path)
            else:
                response = _run_plain_turn(sampler, trajectory, params, adapter_path)
            if multi_round:
                state.add_response(response)
            else:
                state.clear()
    finally:
        if rollout is not None:
            rollout.close()
        sampler.shutdown()
    print(f'\n{_DIM}Bye.{_RESET}')


def _print_welcome(tools_on: bool, stream: bool) -> None:
    mode = 'tool-calling' if tools_on else ('streaming' if stream else 'single-shot')
    print(f'{_CYAN}swift infer{_RESET} — interactive {mode} chat')
    print(f'{_DIM}Commands: clear | reset-system | multi-line | single-line | quit{_RESET}')
    if tools_on:
        print(f"{_DIM}Tools are on: each turn runs the model's tool loop until it answers or hits max_turns.{_RESET}")
    print()


def _run_stream_turn(sampler: Any, trajectory: Dict[str, Any], params: Any, adapter_path: Optional[str]) -> str:
    """One streamed chat turn: echo tokens as they arrive, return the full text."""
    print(f'{_CYAN}Agent:{_RESET} ', end='', flush=True)
    pieces: List[str] = []
    for delta, _finish_reason in sampler.sample_stream(trajectory, params, adapter_path=adapter_path):
        if delta:
            print(delta, end='', flush=True)
            pieces.append(delta)
    print(flush=True)
    return ''.join(pieces)


def _run_plain_turn(sampler: Any, trajectory: Dict[str, Any], params: Any, adapter_path: Optional[str]) -> str:
    """One non-streamed chat turn."""
    from swift.dev.builders import sampled_texts
    texts = sampled_texts(sampler.sample([trajectory], params, adapter_path=adapter_path))
    response = texts[0][0] if texts and texts[0] else ''
    print(f'{_CYAN}Agent:{_RESET} {response}', flush=True)
    return response


def _run_tool_turn(rollout: Any, trajectory: Dict[str, Any], params_dict: Dict[str, Any]) -> str:
    """One human turn through the tool-calling rollout; trace the tools used, then print the final answer.

    ``MultiTurnRollout`` runs the whole intra-turn loop in a single call and returns the finished
    trajectory, so there is no live token stream to show; the tools it invoked are read back off the new
    messages. Only the final assistant text re-enters the conversation history -- the tool plumbing is
    per-turn and the next human turn starts from the answer, not from the intermediate calls.
    """
    messages = trajectory['messages']
    samples = rollout.generate([messages], num_samples=1, sampling_params=params_dict, force_logprobs=False)
    sample = samples[0] if samples else None
    if sample is None:
        print(f'{_YELLOW}[no response from the rollout]{_RESET}')
        return ''
    response = sample.decoded or ''
    new_messages: Sequence[Dict[str, Any]] = []
    if sample.messages:
        new_messages = sample.messages[len(messages):] if len(sample.messages) >= len(messages) else sample.messages
    tools = _tool_names(new_messages)
    if tools:
        print(f'  {_DIM}↳ called tools: {", ".join(tools)}{_RESET}')
    if sample.truncated:
        print(f'  {_YELLOW}[trajectory truncated at max_turns / max_trajectory_tokens]{_RESET}')
    print(f'{_CYAN}Agent:{_RESET} {response}', flush=True)
    return response


def _tool_names(messages: Sequence[Dict[str, Any]]) -> List[str]:
    """The tool function names invoked across the assistant turns, in call order."""
    names: List[str] = []
    for message in messages:
        if message.get('role') != 'assistant':
            continue
        for tool_call in message.get('tool_calls') or []:
            name = (tool_call.get('function') or {}).get('name')
            if name:
                names.append(name)
    return names


class _CliState:
    """Conversation state for :func:`infer_cli`, i.e. legacy's ``InferCliState``."""

    def __init__(self, system: Optional[str] = None):
        self.system = system
        self.messages: List[Dict[str, Any]] = []
        self.media: Dict[str, List[str]] = {'images': [], 'audios': [], 'videos': []}
        self.multiline = False

    def clear(self) -> None:
        self.messages = []
        self.media = {key: [] for key in self.media}

    def add_query(self, query: str) -> None:
        self.messages.append({'role': 'user', 'content': query})

    def add_response(self, response: str) -> None:
        self.messages.append({'role': 'assistant', 'content': response})
        self._prune()

    def _prune(self) -> None:
        """Bound the history, cutting at a user boundary so an assistant answer is never orphaned."""
        if len(self.messages) <= _MAX_HISTORY_MESSAGES:
            return
        cut = len(self.messages) - _MAX_HISTORY_MESSAGES
        while cut < len(self.messages) and self.messages[cut].get('role') != 'user':
            cut += 1
        if cut < len(self.messages):
            self.messages = self.messages[cut:]

    def to_trajectory(self) -> Dict[str, Any]:
        trajectory: Dict[str, Any] = {'messages': list(self.messages)}
        if self.system:
            trajectory['messages'] = [{'role': 'system', 'content': self.system}] + trajectory['messages']
        for key, values in self.media.items():
            if values:
                trajectory[key] = list(values)
        return trajectory

    def read_query(self):
        """Read one turn, handling the commands. Returns the query, None (handled), or ``_QUIT``."""
        raw = self._read_raw()
        stripped = raw.strip()
        if not stripped:
            return None
        lowered = stripped.lower()
        if lowered in ('quit', 'exit', 'q'):
            return _QUIT
        if lowered == 'clear':
            self.clear()
            print(f'{_DIM}History cleared.{_RESET}')
            return None
        if lowered == 'reset-system':
            self.system = input('Enter the new system prompt: ').strip() or None
            self.clear()
            print(f'{_DIM}System set to {self.system!r}; history cleared (a mid-conversation system swap '
                  f'would leave turns answered under the old one).{_RESET}')
            return None
        if lowered in ('multi-line', 'single-line'):
            self.multiline = lowered == 'multi-line'
            print(f'{_DIM}multi-line mode: {self.multiline}{_RESET}')
            return None
        return stripped

    def _read_raw(self) -> str:
        if not self.multiline:
            return input(f'{_GREEN}You:{_RESET} ')
        print(f'{_GREEN}You:{_RESET} {_DIM}(multi-line; end with a single "#" on its own line){_RESET}')
        lines = []
        while True:
            line = input()
            if line.strip() == '#':
                break
            lines.append(line)
        return '\n'.join(lines)

    def prompt_media(self, kinds: Sequence[str]) -> None:
        """Ask for media paths, blank line to stop -- legacy's ``input_mm_data``."""
        for kind in kinds:
            while True:
                path = input(f'Input a {kind[:-1]} path/url (blank to finish): ').strip()
                if not path:
                    break
                self.media.setdefault(kind, []).append(path)
