# Copyright (c) ModelScope Contributors. All rights reserved.
"""Model-node adapter for the weightless structure inspector."""
from __future__ import annotations
import sys
from typing import Any, Mapping

import json

from . import model_inspection


def _typed_configs(params: Mapping[str, Any], node_id: str):
    """Panel params -> typed configs, mirroring ``ModelComponent.build``."""
    from .components.resources import _configs, _fill  # resources pulls twinkle.loss (torch)

    config_pkg = _configs()
    model_kv = dict(params.get('model_config') or {})
    model_kv['model'] = params.get('model')
    model_config = _fill(config_pkg.ModelConfig, model_kv, node_id)
    train_type = str(params.get('train_type') or 'lora')
    tuner_kv = dict(params.get('tuner_config') or {})
    if train_type != 'full':
        tuner_kv['tuner'] = train_type
    tuner_config = _fill(config_pkg.TunerConfig, tuner_kv, node_id) if tuner_kv else None
    train_kv = dict(params.get('train_config') or {})
    train_config = _fill(config_pkg.TrainConfig, train_kv, node_id) if train_kv else None
    return model_config, tuner_config, train_config


def inspect_model_node(node_id: str,
                       params: Mapping[str, Any],
                       *,
                       detail: str = 'config',
                       timeout_s: float = 60.0) -> dict:
    model_config, tuner_config, train_config = _typed_configs(params, node_id)
    return model_inspection.inspect_model(
        model_config,
        detail=detail,
        training=True,
        tuner_config=tuner_config,
        train_config=train_config,
        frozen=bool(params.get('freeze')),
        backend=str(params.get('backend') or 'hf'),
        parallel=params.get('parallel') or params.get('parallel_spec') or {},
        timeout_s=timeout_s)


def main(*, warmup: bool = True) -> int:
    """Serve newline-delimited inspection requests without starting a training session."""
    if warmup:
        from .components.resources import _configs  # heavy: pay it once before reporting ready
        _configs()
        print('[model-inspection] {"event":"ready"}', flush=True)
    for line in sys.stdin:
        request_id = ''
        try:
            request = json.loads(line)
            request_id = str(request.get('request_id') or '')
            result = inspect_model_node(
                str(request.get('node_id') or 'model'),
                request.get('params') or {},
                detail=str(request.get('detail') or 'config'),
                timeout_s=float(request.get('timeout_s') or 60.0))
            reply = {'request_id': request_id, 'ok': True, 'result': result}
        except Exception as exc:
            reply = {'request_id': request_id, 'ok': False, 'error': f'{type(exc).__name__}: {exc}'}
        print(f'[model-inspection] {json.dumps(reply, ensure_ascii=False)}', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
