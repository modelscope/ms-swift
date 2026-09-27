# Copyright (c) ModelScope Contributors. All rights reserved.
"""An EXTERNAL tool plugin: bring your own tools instead of the built-in ``sandbox``.

``--tools`` selects tool plugins by registered name. ``sandbox`` is the built-in one (it exposes the
leased env's own run_command/write_file/read_file). To use your own, write a ``ToolPlugin`` in a .py
like this, register it under a name, then point the run at the file and the name::

    swift infer ... --external_plugins examples/v5/infer/custom_tools.py --tools calculator

``--external_plugins`` imports this file before anything is built (``PluginRegistry.load_configured``),
so the ``@PluginRegistry.register`` below has run by the time ``--tools calculator`` is resolved. Tools
are a multi-turn feature, so a run that names any tool also needs ``--max_turns``.

Contract (see ``swift.dev.plugin.ToolPlugin`` and ``twinkle_agentic.tools.base.Tool``):
  * ``ToolPlugin.build(env)`` is called once per episode with the sandbox ``Env`` this episode leased,
    and returns the twinkle ``Tool`` objects the model may call. A tool that needs no sandbox can ignore
    ``env`` (the calculator does); one that acts in the workspace wraps ``env`` (see ``EnvTool.from_env``,
    which is exactly what the built-in ``sandbox`` plugin returns).
  * Each ``Tool`` implements ``__call__(tool_name, arguments) -> str`` (the observation fed back to the
    model) and ``tool_info() -> {'type': 'function', 'function': {name, description, parameters}}``
    (the OpenAI-shaped schema advertised in the prompt). ``ToolManager`` indexes tools by that name.
"""
from __future__ import annotations
import ast
import operator
from typing import Any, Dict, List

from swift.dev.plugin import PluginRegistry, ToolPlugin
from twinkle.data_format.message import Tool as ToolInfo

# Only numbers and arithmetic operators are evaluated -- never names, calls or attributes -- so a
# completion cannot run arbitrary code through this tool (no eval/exec anywhere).
_BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARYOPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}


def _safe_arith(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _safe_arith(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _BINOPS:
        return _BINOPS[type(node.op)](_safe_arith(node.left), _safe_arith(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARYOPS:
        return _UNARYOPS[type(node.op)](_safe_arith(node.operand))
    raise ValueError(f'unsupported expression: {ast.dump(node)}')


class _Calculator:
    """A minimal twinkle ``Tool``: evaluate an arithmetic expression, return the result as a string."""

    def __call__(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        expression = str(arguments.get('expression', ''))
        try:
            return str(_safe_arith(ast.parse(expression, mode='eval')))
        except Exception as exc:  # noqa: BLE001 - the observation is whatever the model reads back
            return f'Error: {type(exc).__name__}: {exc}'

    def tool_info(self) -> ToolInfo:
        return {
            'type': 'function',
            'function': {
                'name': 'calculator',
                'description': 'Evaluate an arithmetic expression (+ - * / // % **) and return the number.',
                'parameters': {
                    'type': 'object',
                    'properties': {
                        'expression': {
                            'type': 'string',
                            'description': 'The arithmetic expression to evaluate, e.g. "2*(3+4)".'
                        }
                    },
                    'required': ['expression'],
                },
            },
        }


@PluginRegistry.register('tool', 'calculator')
class CalculatorTools(ToolPlugin):
    """Expose the calculator tool. Selected with ``--tools calculator`` after ``--external_plugins`` this file."""

    def build(self, env: Any) -> List[Any]:
        # This tool is stateless and touches no workspace, so the leased env is ignored. Return one
        # instance per episode; return [] to contribute nothing, or wrap env (EnvTool) to act in it.
        return [_Calculator()]
