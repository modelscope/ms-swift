# Copyright (c) ModelScope Contributors. All rights reserved.

import ast
from pathlib import Path


def test_ipc_bucket_ack_happens_after_tp_barrier():
    """The shared transfer buffer must not be reusable before every TP rank copied it."""
    source_path = Path(__file__).parents[2] / 'swift' / 'pipelines' / 'infer' / 'rollout.py'
    tree = ast.parse(source_path.read_text(encoding='utf-8'))

    update_fn = next(
        node for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == 'update_weights_from_ipc'
    )
    source_lines = source_path.read_text(encoding='utf-8').splitlines()
    function_source = '\n'.join(source_lines[update_fn.lineno - 1:update_fn.end_lineno])

    barrier_pos = function_source.find('_dist.barrier(group=cpu_group)')
    ack_pos = function_source.find("socket.send(b'')", barrier_pos)

    assert barrier_pos >= 0, 'TP barrier is required before acknowledging a reusable bucket'
    assert ack_pos > barrier_pos, 'ACK must be sent only after all TP ranks finish copying the bucket'
