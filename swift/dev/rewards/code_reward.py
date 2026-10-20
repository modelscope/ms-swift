# Copyright (c) ModelScope Contributors. All rights reserved.
from __future__ import annotations

import asyncio
import json
import re
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, List, Optional, Sequence, Tuple

from swift.dev.utils import get_logger
from .reward import ORM

logger = get_logger()

__all__ = ['CodeExecutionReward']

#: Markers the wrapper prints around the program's captured stdout, lifted back out of the merged
#: stdout+stderr ``run_script`` returns. Unique enough that a program will not print them by accident.
_STDOUT_BEGIN = '<<<TWINKLE_CODE_STDOUT_BEGIN>>>'
_STDOUT_END = '<<<TWINKLE_CODE_STDOUT_END>>>'

#: A fenced ```lang ... ``` block, the shape code completions usually arrive in. The language tag is
#: matched loosely and the body taken verbatim; a completion with no fence is used whole.
_FENCE_RE = re.compile(r'```[a-zA-Z0-9_+\-#]*[ \t]*\r?\n(.*?)```', re.DOTALL)

#: A leading ```lang opener with no closer -- what a rollout truncated by ``max_tokens`` leaves behind.
_OPEN_FENCE_RE = re.compile(r'^[ \t]*```[a-zA-Z0-9_+\-#]*[ \t]*\r?\n')


@dataclass(frozen=True)
class _ShellLang:
    """How to build and run one shell-expressed language (compiled or interpreted)."""

    ext: str
    #: Compile command with ``{src}`` / ``{exe}`` placeholders, or None for an interpreted language.
    compile_cmd: Optional[str]
    #: Run command with ``{exe}`` / ``{src}`` placeholders; stdin is redirected from the input file.
    run_cmd: str


#: Languages run through a Python wrapper (stdin as ``StringIO``, stdout captured in-process).
_PYTHON_LANGS = frozenset({'python', 'python3', 'py'})

#: Languages run through a shell heredoc + build + run. Restricted to toolchains a training host
#: commonly has; an unlisted language is rejected up front (see _check_language), not at run time.
_SHELL_LANGS = {
    'c': _ShellLang('.c', 'gcc -O2 -std=c11 {src} -o {exe}', './{exe}'),
    'cpp': _ShellLang('.cpp', 'g++ -O2 -std=c++17 {src} -o {exe}', './{exe}'),
    'c++': _ShellLang('.cpp', 'g++ -O2 -std=c++17 {src} -o {exe}', './{exe}'),
    'javascript': _ShellLang('.js', None, 'node {src}'),
    'js': _ShellLang('.js', None, 'node {src}'),
}


def _norm_lang(language: Optional[str]) -> str:
    return (language or 'python').strip().lower()


def _check_language(language: Optional[str]) -> None:
    """Fail loudly on a language this reward cannot run, before any code is executed."""
    lang = _norm_lang(language)
    if lang not in _PYTHON_LANGS and lang not in _SHELL_LANGS:
        supported = sorted(_PYTHON_LANGS | set(_SHELL_LANGS))
        raise ValueError(f'code_execution reward: unsupported language {language!r}; supported: {supported}.')


def _extract_code(completion: Any, language: str) -> str:
    """The program text from a completion: the first fenced block if there is one, else the whole thing.

    A truncated rollout (``max_tokens``) can leave an opener with no closer; rather than run the raw text
    -- whose leading ```` ```lang ```` line is a syntax error and would score a false 0 -- a dangling
    opener and any trailing ```` ``` ```` are stripped so a complete-but-unfenced body still runs.
    """
    text = completion if isinstance(completion, str) else str(completion)
    match = _FENCE_RE.search(text)
    if match:
        return match.group(1)
    stripped = _OPEN_FENCE_RE.sub('', text.strip(), count=1)
    if stripped.endswith('```'):
        stripped = stripped[:-3]
    return stripped.strip()


def _parse_test_cases(raw: Any) -> List[Tuple[str, str]]:
    """A row's ``test_cases`` -> ``[(stdin, expected_stdout), ...]``.

    Accepts a JSON string or an already-decoded ``{"inputs": [...], "outputs": [...]}``. A row that
    cannot be parsed yields ``[]`` (scored 0 with a warning) rather than raising: one bad row is a data
    problem, not a reason to abort a whole scoring batch.
    """
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except (ValueError, TypeError):
            logger.warning(f'code_execution reward: unparseable test_cases JSON, scoring 0: {raw[:120]!r}')
            return []
    if not isinstance(raw, dict):
        logger.warning(f'code_execution reward: test_cases is not a mapping, scoring 0: {type(raw).__name__}')
        return []
    inputs = raw.get('inputs') or []
    outputs = raw.get('outputs') or []
    if not isinstance(inputs, list) or not isinstance(outputs, list):
        logger.warning('code_execution reward: test_cases inputs/outputs must be lists, scoring 0.')
        return []
    if len(inputs) != len(outputs):
        logger.warning(f'code_execution reward: {len(inputs)} inputs vs {len(outputs)} outputs; '
                       'scoring the shorter prefix.')
    return [(str(i), str(o)) for i, o in zip(inputs, outputs)]


def _normalize_output(text: str) -> List[str]:
    """Lines with trailing whitespace stripped and leading/trailing blank lines dropped.

    The whitespace-tolerant comparison a judge uses: a program whose answer is right but whose trailing
    newline or line-ending spaces differ from the reference still passes, while a wrong line break in
    the middle does not.
    """
    lines = [line.rstrip() for line in text.splitlines()]
    while lines and lines[0] == '':
        lines.pop(0)
    while lines and lines[-1] == '':
        lines.pop()
    return lines


def _outputs_match(actual: str, expected: str) -> bool:
    return _normalize_output(actual) == _normalize_output(expected)


def _extract_stdout(merged: Optional[str]) -> Optional[str]:
    """The program's stdout from between the markers, or None when they are absent.

    Absent markers mean the wrapper never completed -- a timeout killed the child, a compiled language
    failed to build, or the program crashed hard -- and every one of those is a wrong answer, so the
    caller scores the case 0.
    """
    if not merged or _STDOUT_BEGIN not in merged or _STDOUT_END not in merged:
        return None
    middle = merged.split(_STDOUT_BEGIN, 1)[1].split(_STDOUT_END, 1)[0]
    return middle


def _python_source(code: str, test_input: str) -> str:
    """A Python wrapper: feed ``test_input`` as stdin, capture the program's stdout between markers.

    The user code is embedded with ``repr`` (a valid Python literal, so no quoting hazard) and run with
    ``exec(compile(...))``; any exception it raises is swallowed, leaving whatever it printed before the
    crash in the capture -- a runtime error is a wrong answer, not a reward failure.
    """
    return ''.join([
        'import sys, io\n',
        'sys.stdin = io.StringIO(', repr(test_input), ')\n',
        '_cap = io.StringIO()\n',
        '_real = sys.stdout\n',
        'sys.stdout = _cap\n',
        'try:\n',
        '    exec(compile(', repr(code), ', "<user_code>", "exec"), {"__name__": "__main__"})\n',
        'except BaseException:\n',
        '    pass\n',
        'finally:\n',
        '    sys.stdout = _real\n',
        'sys.stdout.write(', repr(_STDOUT_BEGIN + '\n'), ')\n',
        'sys.stdout.write(_cap.getvalue())\n',
        'sys.stdout.write(', repr('\n' + _STDOUT_END + '\n'), ')\n',
    ])


def _shell_source(spec: _ShellLang, code: str, test_input: str) -> str:
    """A shell script: write the program and its input via quoted heredocs, build, run, mark stdout.

    Filenames carry a per-call uuid so concurrent runs sharing one ``AgentEnv`` workspace never collide
    (``LocalEnv`` already gives each call its own temp dir). The heredoc delimiter is uuid-suffixed and
    quoted, so the program body is taken literally -- no shell expansion, no early terminator. A compiled
    language builds behind an ``if``, so a build failure prints no markers and scores 0.
    """
    uid = uuid.uuid4().hex
    src = f'prog_{uid}{spec.ext}'
    exe = f'prog_{uid}.out'
    inp = f'in_{uid}.txt'
    sentinel = f'TWINKLE_CODE_EOF_{uid}'
    parts: List[str] = [
        f"cat > {src} <<'{sentinel}'\n", code, '\n', f'{sentinel}\n',
        f"cat > {inp} <<'{sentinel}'\n", test_input, '\n', f'{sentinel}\n',
    ]
    run_block = [
        f"printf '%s\\n' '{_STDOUT_BEGIN}'\n",
        f'{spec.run_cmd.format(exe=exe, src=src)} < {inp}\n',
        f"printf '%s\\n' '{_STDOUT_END}'\n",
    ]
    if spec.compile_cmd:
        parts.append(f'if {spec.compile_cmd.format(src=src, exe=exe)} 2>/dev/null; then\n')
        parts.extend(f'  {line}' for line in run_block)
        parts.append('fi\n')
    else:
        parts.extend(run_block)
    return ''.join(parts)


def _build_run(language: str, code: str, test_input: str) -> Tuple[str, str]:
    """``(interpreter, source)`` for ``env.run_script`` in the given language."""
    lang = _norm_lang(language)
    if lang in _PYTHON_LANGS:
        return 'python', _python_source(code, test_input)
    return 'shell', _shell_source(_SHELL_LANGS[lang], code, test_input)


def _execute_one(env: Any, code: str, test_input: str, language: str, timeout: int) -> Optional[str]:
    """Run one program against one test case in a worker thread; blocking, hence the executor.

    Returns the program's captured stdout, or None when the wrapper never completed -- a timeout, a build
    failure, a hard crash -- every one of which is a wrong answer. Scoring keys off stdout alone, the way
    a judge does, so the exit code ``run_script`` also returns is dropped here.
    """
    interpreter, source = _build_run(language, code, test_input)
    _exit_code, merged = env.run_script(source, interpreter, timeout=timeout)
    return _extract_stdout(merged)


class CodeExecutionReward(ORM):
    """Score generated code by executing it against per-sample test cases.

    Registered as ``code_execution``; knobs read off the run's RLConfig (``self.args``, each with a
    default so ``args=None`` works): ``code_language`` (the default when a row carries no ``language``
    column), ``code_timeout``, ``code_max_concurrency``, ``code_memory_limit_gb``,
    ``code_sandbox_template`` (None -> LocalEnv; a template name -> an AgentEnv microVM).
    """

    def __init__(self, args: Optional[Any] = None, **kwargs):
        super().__init__(args, **kwargs)
        self.language = getattr(args, 'code_language', None) or 'python'
        self.timeout = int(getattr(args, 'code_timeout', None) or 10)
        self.max_concurrency = max(1, int(getattr(args, 'code_max_concurrency', None) or 8))
        self.memory_limit_gb = getattr(args, 'code_memory_limit_gb', 2.0)
        self.sandbox_template = getattr(args, 'code_sandbox_template', None)
        _check_language(self.language)

    def _build_env(self) -> Any:
        """A throwaway execution env: ``LocalEnv`` by default, ``AgentEnv`` when a template is named.

        Built once per scoring batch and shared across its concurrent runs -- safe because ``LocalEnv``
        with ``workspace=None`` gives every ``run_script`` its own temp dir, and the shell wrapper's
        uuid filenames keep concurrent ``AgentEnv`` runs from colliding in its one workspace.
        """
        if self.sandbox_template:
            from twinkle_agentic.envs import AgentEnv
            env = AgentEnv(
                template=self.sandbox_template,
                command_timeout=self.timeout,
                metadata={'run': 'swift', 'reward': 'code_execution'})
            env.clear()  # boots the sandbox; run_script needs one up.
            return env
        from twinkle_agentic.envs import LocalEnv
        return LocalEnv(workspace=None, command_timeout=self.timeout, memory_limit_gb=self.memory_limit_gb)

    @staticmethod
    def _close_env(env: Any) -> None:
        try:
            env.close()
        except Exception as exc:  # noqa: BLE001 -- teardown is best effort; the score is already computed
            logger.warning(f'code_execution reward: env close failed: {type(exc).__name__}: {exc}')

    async def __call__(self, trajectories, **kwargs) -> List[float]:
        """One score per trajectory: the fraction of its test cases the program passes.

        ``test_cases`` and ``language`` ride each trajectory row (the packed ``user_data`` label or
        the flat column); ``language`` may be absent to take ``code_language`` from the config.
        """
        from twinkle.data_format import ground_truth, last_assistant_text

        completions = [last_assistant_text(t) for t in trajectories]
        cases_raw = [ground_truth(t, 'test_cases') for t in trajectories]
        if not any(case is not None for case in cases_raw):
            raise ValueError("the 'code_execution' reward scores code against test cases, but no "
                             "'test_cases' column was provided. Add one to the dataset as JSON "
                             '\'{"inputs": [...], "outputs": [...]}\' (stdin / expected stdout per case).')
        n = len(trajectories)
        lang_col = [ground_truth(t, 'language') or self.language for t in trajectories]
        for lang in set(lang_col):
            _check_language(lang)

        codes = [_extract_code(completion, lang) for completion, lang in zip(completions, lang_col)]
        parsed = [_parse_test_cases(raw) for raw in cases_raw]

        env = None
        executor = None
        try:
            env = self._build_env()
            executor = ThreadPoolExecutor(max_workers=self.max_concurrency, thread_name_prefix='code_reward')
            loop = asyncio.get_running_loop()
            scores = await asyncio.gather(*[
                self._score_one(loop, executor, env, code, cases, lang)
                for code, cases, lang in zip(codes, parsed, lang_col)
            ])
        finally:
            if executor is not None:
                executor.shutdown(wait=False)
            if env is not None:
                self._close_env(env)
        return list(scores)

    async def _score_one(self, loop: Any, executor: ThreadPoolExecutor, env: Any, code: str,
                         cases: List[Tuple[str, str]], language: str) -> float:
        if not cases:
            return 0.0
        passed = 0
        for test_input, expected in cases:
            actual = await loop.run_in_executor(executor, _execute_one, env, code, test_input, language,
                                                self.timeout)
            if actual is not None and _outputs_match(actual, expected):
                passed += 1
        return passed / len(cases)
