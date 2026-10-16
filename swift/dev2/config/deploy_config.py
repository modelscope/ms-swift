# Copyright (c) ModelScope Contributors. All rights reserved.
from dataclasses import dataclass
from typing import Literal, Optional


@dataclass
class DeployConfig:
    """How the server starts, listens, authenticates, and identifies itself (generation is
    GenerationConfig's job, engine choice InferConfig's)."""

    # === Model startup ===
    merge_lora: bool = False  # merge a single adapter into the base weights before serving

    # === Address ===
    host: str = '0.0.0.0'
    port: int = 8000
    # Bearer token per request; None serves unauthenticated (only safe off-network).
    api_key: Optional[str] = None

    # === TLS ===
    # Both are needed for https; one alone leaves the server on plain http.
    ssl_keyfile: Optional[str] = None
    ssl_certfile: Optional[str] = None

    # === Identity in the OpenAI API ===
    # The name clients pass as ``model``; None falls back to the loaded model's own name.
    served_model_name: Optional[str] = None
    owned_by: str = 'swift'

    # === Response detail ===
    max_logprobs: int = 20  # ceiling on a request's top_logprobs
    max_concurrency: int = 64

    # === Logging ===
    verbose: bool = True  # log every request, including prompt and completion
    # Throughput summaries between quiet periods; only read when verbose is off.
    log_interval: int = 20
    log_level: Literal['critical', 'error', 'warning', 'info', 'debug', 'trace'] = 'info'

    # === Ray Serve runtime ===
    # None/'local' starts an in-process local Ray head; 'auto' attaches to a reachable cluster.
    ray_address: Optional[str] = None
    ray_namespace: str = 'swift'
    # '/v1' reproduces the legacy OpenAI paths (/v1/chat/completions, /v1/completions, ...).
    route_prefix: str = '/v1'
    # None derives from backend + enable_data_plane; only the *_async kinds can hand a rollout a
    # DataRef, so an explicit non-async sampler_type with enable_data_plane is rejected.
    sampler_type: Optional[Literal['mock', 'vllm', 'vllm_async', 'sglang', 'sglang_async', 'torch']] = None
    # Token-in-token-out over the data plane: adds a data_plane app the rollout can stream into.
    enable_data_plane: bool = False

    # === Persistence (forwarded to ServerConfig.persistence) ===
    # None keeps the in-memory default; 'redis' needs a redis_url the deploy surface
    # does not carry, so it is left to twinkle-server's own config.
    persistence_mode: Optional[Literal['memory', 'file']] = None
    persistence_file_path: Optional[str] = None

    # === Scaling (forwarded to each application's deployment) ===
    num_replicas: Optional[int] = None  # None keeps Ray Serve's single-replica default
    autoscaling: bool = False
