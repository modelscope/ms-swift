# Copyright (c) ModelScope Contributors. All rights reserved.
"""Shared fixtures for the ``swift deploy`` suite.

Two tiers, mirroring ``swift/dev/tests/infer``:

- **fast** (no GPU, no download): call ``build_server_config`` / ``parse_deploy_configs`` directly and
  assert on the assembled ``ServerConfig`` / parsed Configs. Nothing here imports a model at module scope.
- **slow** (``@pytest.mark.slow`` + ``@pytest.mark.accel(1)``): :func:`run_deploy_process` really spawns
  ``run_deploy`` in a subprocess -- a local Ray head + Ray Serve + a vLLM engine on the real
  ``Qwen/Qwen3.5-4B`` -- and yields the gateway base URL once ``{route_prefix}/models`` answers 200. The
  tests then drive it over HTTP (OpenAI surface) or through ``twinkle_client`` (token-in-token-out), which
  is the same path ``examples/v5/deploy`` uses, so a green suite means the examples work as written.

``Qwen3.5-4B`` is a unified vision-language model, so one checkpoint covers both the text and the
multimodal serving tests.
"""
from __future__ import annotations

import base64
import io
import os
import socket
from contextlib import contextmanager
from typing import Any, Dict, Optional

import pytest

#: One real checkpoint for the whole suite: swift registers it as the multimodal ``qwen3_5`` type and
#: twinkle maps ``Qwen3.5`` -> ``Qwen3_5Template``, so text and image-text both resolve correctly.
MODEL = os.environ.get('DEPLOY_TEST_MODEL', 'Qwen/Qwen3.5-4B')
MODEL_TYPE = 'qwen3_5'
TEMPLATE = 'qwen3_5'
API_KEY = 'EMPTY'
SERVED_NAME = 'policy'
#: vLLM engine args shared by every slow server: eager mode skips CUDA-graph capture (faster bring-up),
#: a bounded context and a modest utilisation keep the replica well inside a single card's free memory
#: (0.5, not 0.9, so a previous module's server that has not fully released the GPU cannot OOM this one).
ENGINE_ARGS: Dict[str, Any] = {
    'gpu_memory_utilization': 0.5,
    'enforce_eager': True,
    'max_model_len': 8192,
}
#: Bringing up Ray Serve + vLLM + a 4B checkpoint is minutes, not seconds; give the readiness probe room.
SERVER_TIMEOUT = float(os.environ.get('DEPLOY_TEST_TIMEOUT', '900'))


def free_port() -> int:
    """An OS-assigned ephemeral port, closed again before the server binds it."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('127.0.0.1', 0))
        return s.getsockname()[1]


def port_is_closed(port: int, host: str = '127.0.0.1') -> bool:
    """True when nothing accepts a TCP connection on ``port`` (the server really went down)."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(1.0)
        return s.connect_ex((host, port)) != 0


def image_data_url(size: int = 64, seed: int = 0) -> str:
    """A tiny random RGB image as a ``data:image/png;base64,...`` URL for a multimodal chat request."""
    import numpy as np
    from PIL import Image
    rng = np.random.default_rng(seed)
    arr = (rng.random((size, size, 3)) * 255).astype('uint8')
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format='PNG')
    return 'data:image/png;base64,' + base64.b64encode(buf.getvalue()).decode()


_TEMPLATE_CACHE: Dict[str, Any] = {}


def build_template():
    """The model's real twinkle chat template (tokenizer only, no weights), cached per process.

    Built exactly as the served sampler builds it -- ``get_template_for_model`` picks the class and
    ``construct_class`` instantiates it with ``model_id`` -- so a token-in-token-out test encodes its
    prompt with the same template the server has pinned, without loading the model a second time.
    """
    if 'template' not in _TEMPLATE_CACHE:
        import twinkle.template
        from twinkle.server.utils import get_template_for_model
        from twinkle.template import Template
        from twinkle.utils import construct_class
        _TEMPLATE_CACHE['template'] = construct_class(
            get_template_for_model(MODEL), Template, twinkle.template, model_id=MODEL)
    return _TEMPLATE_CACHE['template']


def encode_prompt_ids(messages) -> list:
    """Encode a chat prompt to ``input_ids`` with the real template -- what a token-in-token-out loop feeds.

    ``add_generation_prompt=True`` appends the assistant turn opener, so the ids are a ready-to-continue
    prompt exactly like the one the server would build from the same messages.
    """
    from twinkle.data_format import Trajectory
    encoded = build_template().encode(Trajectory(messages=list(messages)), add_generation_prompt=True)
    input_ids = encoded['input_ids']
    return input_ids.tolist() if hasattr(input_ids, 'tolist') else list(input_ids)


def build_configs(*, deploy_overrides: Optional[Dict[str, Any]] = None, port: int = 8000):
    """The dev Configs ``run_deploy`` needs, with ``DeployConfig`` overridden per test.

    ``served_model_name`` defaults to ``SERVED_NAME`` so the sampler is mounted at a stable, short name
    (the client's ``model`` must equal it). ``api_key`` defaults to ``API_KEY`` so the auth path is on.
    """
    from swift.dev.config import DeployConfig, ModelConfig, TemplateConfig
    model_config = ModelConfig(model=MODEL, model_type=MODEL_TYPE, torch_dtype='bfloat16')
    template_config = TemplateConfig(template=TEMPLATE)
    overrides = {
        'served_model_name': SERVED_NAME,
        'api_key': API_KEY,
        'host': '127.0.0.1',
        'route_prefix': '/v1',
    }
    overrides.update(deploy_overrides or {})
    deploy_config = DeployConfig(port=port, **overrides)
    return model_config, template_config, deploy_config


@pytest.fixture(scope='session')
def live_server():
    """Factory: spawn a real ``swift deploy`` server and yield its base URL, then tear it down.

    Returns a callable used as a context manager so a test (or a module-scoped fixture) controls the
    server's lifetime::

        with live_server(backend='vllm', deploy_overrides={'enable_data_plane': True}) as srv:
            srv.get('/models')

    The yielded object exposes ``base_url`` (``{scheme}://127.0.0.1:{port}{route_prefix}``), ``port``,
    ``deploy_config`` and thin ``get`` / ``post`` / ``stream`` HTTP helpers that inject the Bearer key.
    """
    from swift.dev.recipe.run_deploy import run_deploy_process

    @contextmanager
    def _launch(*, backend: str = 'vllm', engine_args: Optional[Dict[str, Any]] = None,
                deploy_overrides: Optional[Dict[str, Any]] = None, adapter_mapping: Optional[Dict[str,
                                                                                                 str]] = None,
                timeout: float = SERVER_TIMEOUT, port: Optional[int] = None):
        port = port or free_port()
        model_config, template_config, deploy_config = build_configs(deploy_overrides=deploy_overrides, port=port)
        cm = run_deploy_process(
            model_config,
            template_config,
            None,
            port=port,
            timeout=timeout,
            backend=backend,
            engine_args=dict(ENGINE_ARGS if engine_args is None else engine_args),
            adapter_mapping=adapter_mapping,
            deploy_config=deploy_config,
        )
        with cm as base_url:
            yield LiveServer(base_url, port, deploy_config)

    return _launch


class LiveServer:
    """A running deployment: its base URL plus HTTP helpers that carry the Bearer key."""

    def __init__(self, base_url: str, port: int, deploy_config: Any):
        self.base_url = base_url.rstrip('/')
        self.port = port
        self.deploy_config = deploy_config
        self.api_key = deploy_config.api_key

    def url(self, path: str) -> str:
        return f'{self.base_url}/{path.lstrip("/")}'

    def _headers(self, auth: bool = True, extra: Optional[Dict[str, str]] = None) -> Dict[str, str]:
        headers = {'Content-Type': 'application/json'}
        if auth and self.api_key:
            headers['Authorization'] = f'Bearer {self.api_key}'
        if extra:
            headers.update(extra)
        return headers

    def get(self, path: str, *, auth: bool = True, timeout: float = 60.0):
        import httpx
        return httpx.get(self.url(path), headers=self._headers(auth), timeout=timeout)

    def post(self, path: str, json: Any, *, auth: bool = True, timeout: float = 300.0):
        import httpx
        return httpx.post(self.url(path), json=json, headers=self._headers(auth), timeout=timeout)

    def stream(self, path: str, json: Any, *, auth: bool = True, timeout: float = 300.0):
        """POST and yield the raw SSE lines (``data: ...``) as the server produces them."""
        import httpx
        with httpx.stream('POST', self.url(path), json=json, headers=self._headers(auth), timeout=timeout) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if line:
                    yield line

    def chat(self, messages, *, model: str = SERVED_NAME, path: str = '/chat/completions', **params):
        """A non-streaming chat completion, returned as the parsed OpenAI response dict."""
        body = {'model': model, 'messages': messages, 'stream': False, **params}
        resp = self.post(path, body)
        resp.raise_for_status()
        return resp.json()
