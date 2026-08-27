"""OpenAI-compatible server configuration."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional


@dataclass
class DeployConfig:
    """Where the server listens, how it authenticates, and what it calls itself.

    Only the serving surface lives here. What the server generates with is GenerationConfig, and which
    engine backs it is InferConfig -- a deployment swaps engines without touching any of these fields.
    """

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

    # === Logging ===
    #: Log every request, including its prompt and completion.
    verbose: bool = True
    #: Requests between throughput summaries. Only consulted when ``verbose`` is off, where it is the
    #: sole indication the server is alive.
    log_interval: int = 20
    log_level: Literal['critical', 'error', 'warning', 'info', 'debug', 'trace'] = 'info'

    # === Extension ===
    #: Registered name of a context manager wrapped around each request, for callers that need
    #: per-request setup or teardown the server itself does not provide.
    context_manager: Optional[str] = None
