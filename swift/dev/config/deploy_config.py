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

    # === Extension ===
    #: Registered name of a context manager wrapped around each request, for callers that need
    #: per-request setup or teardown the server itself does not provide.
    context_manager: Optional[str] = None
