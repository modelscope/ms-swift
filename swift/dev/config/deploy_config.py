"""OpenAI-compatible server configuration."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Literal, Optional


@dataclass
class DeployConfig:
    """How the server starts, listens, authenticates, and identifies itself.

    What the server generates with is GenerationConfig, and which engine backs it is InferConfig.
    """

    # === Model startup ===
    #: Merge a single adapter into the base weights before starting the server.
    merge_lora: bool = False

    # === Address ===
    host: str = '0.0.0.0'
    port: int = 8000
    #: Bearer token required on every request. None serves without authentication, which is only safe
    #: when the port is not reachable from outside the host.
    api_key: Optional[str] = None

    # === TLS ===
    #: Both are needed for https; setting one alone leaves the server on plain http.
    ssl_keyfile: Optional[str] = None
    ssl_certfile: Optional[str] = None

    # === Identity in the OpenAI API ===
    #: The name clients pass as ``model`` and see in ``/v1/models``. None falls back to the loaded
    #: model's own name, so an alias here is what lets a client keep working after the weights change.
    served_model_name: Optional[str] = None
    owned_by: str = 'swift'

    # === Response detail ===
    #: Ceiling on a request's ``top_logprobs``. Bounded because each extra entry is paid for per token.
    max_logprobs: int = 20
    max_concurrency: int = 64

    # === Logging ===
    #: Log every request, including its prompt and completion.
    verbose: bool = True
    #: Requests between throughput summaries. Only consulted when ``verbose`` is off, where it is the
    #: sole indication the server is alive.
    log_interval: int = 20
    request_log_path: Optional[str] = None
    log_level: Literal['critical', 'error', 'warning', 'info', 'debug', 'trace'] = 'info'

    # === Ray Serve runtime ===
    #: twinkle-server runs on Ray Serve. ``None`` or ``'local'`` starts an in-process local Ray head, so
    #: a single ``swift deploy`` command serves without a pre-existing cluster; ``'auto'`` attaches to a
    #: cluster already reachable at the default address (the ``DistributedConfig.mode='ray'`` case).
    ray_address: Optional[str] = None
    #: Ray namespace the deployments and their named actors are registered under.
    ray_namespace: str = 'twinkle_cluster'
    #: Mount prefix of the gateway app. ``'/v1'`` reproduces the legacy OpenAI paths, so clients hit
    #: ``/v1/chat/completions``, ``/v1/completions``, ``/v1/embeddings`` and ``/v1/models``.
    route_prefix: str = '/v1'
    #: Sampler deployment kind. ``None`` derives it from the backend and ``enable_data_plane``: vLLM
    #: becomes ``'vllm_async'`` with the data plane else ``'vllm'``, sglang becomes ``'sglang_async'``
    #: with the data plane else ``'sglang'``, transformers becomes ``'torch'``. The ``*_async`` variants
    #: are the only ones that can hand a rollout a ``DataRef``, so an explicit non-async ``sampler_type``
    #: together with ``enable_data_plane`` is rejected at config-build time.
    sampler_type: Optional[Literal['mock', 'vllm', 'vllm_async', 'sglang', 'sglang_async', 'torch']] = None
    #: Serve token-in-token-out over the data plane: adds a ``data_plane`` application and points the
    #: sampler at it, so a rollout can ``asample_to_data_plane`` and train on the returned ``DataRef``.
    enable_data_plane: bool = False

    # === Persistence (forwarded to ServerConfig.persistence) ===
    persistence_mode: Optional[str] = None
    persistence_file_path: Optional[str] = None

    # === Scaling (forwarded to each application's deployment) ===
    #: Replica count per application. ``None`` keeps Ray Serve's single-replica default.
    num_replicas: Optional[int] = None
    #: Enable Ray Serve autoscaling instead of a fixed ``num_replicas``.
    autoscaling: bool = False
