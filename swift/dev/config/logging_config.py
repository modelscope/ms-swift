"""Logging and experiment tracking configuration."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Literal, Optional


# TODO: integrate it
@dataclass
class LoggingConfig:
    """SwanLab, TensorBoard, and general logging settings."""

    # === SwanLab ===
    swanlab_token: Optional[str] = None
    swanlab_project: str = 'ms-swift'
    swanlab_workspace: Optional[str] = None
    swanlab_exp_name: Optional[str] = None
    swanlab_notification_method: Optional[str] = None
    swanlab_webhook_url: Optional[str] = None
    swanlab_secret: Optional[str] = None
    swanlab_mode: Literal['cloud', 'local'] = 'cloud'

    # === SwanLab email notification ===
    # Only read when swanlab_notification_method selects email. Kept apart from the webhook route above
    # because email needs a relay the process can reach, which the webhook does not.
    swanlab_smtp_server: Optional[str] = None
    swanlab_smtp_port: Optional[int] = None
    swanlab_sender_email: Optional[str] = None
    swanlab_receiver_email: Optional[str] = None
    swanlab_email_language: Optional[str] = 'zh'

    # === Weights & Biases ===
    # Naming 'wandb' in report_to is what enables it; these only say which project and run it lands in.
    # Credentials are not here -- wandb reads WANDB_API_KEY from the environment itself.
    wandb_project: str = 'ms-swift'
    wandb_exp_name: Optional[str] = None
    #: Log each distinct prompt once instead of once per generation. Megatron RLHF only, where the same
    #: prompt is sampled repeatedly and the repeats otherwise drown the table.
    wandb_log_unique_prompts: Optional[bool] = None

    # === TensorBoard ===
    #: Where event files are written. Distinct from ``logging_dir``, which HF's own TensorBoard
    #: integration uses; this one is Megatron's, and the two backends write through different writers.
    tensorboard_dir: Optional[str] = None
    #: Events buffered before a flush. Larger costs less I/O and loses more on a crash.
    tensorboard_queue_size: int = 50

    # === General Logging ===
    #: Trackers the run reports to: 'tensorboard', 'swanlab', 'wandb', or 'none'. This is the switch;
    #: the per-tracker blocks above only say where the run lands once its tracker is named here.
    report_to: List[str] = field(default_factory=lambda: ['tensorboard'])
    logging_dir: Optional[str] = None
    #: Name this run shows under in every tracker. None lets each tracker invent one, which is how runs
    #: end up indistinguishable in a shared project.
    run_name: Optional[str] = None
    logging_steps: int = 5
    logging_first_step: bool = True
    #: Replace a nan or inf loss with the running average instead of logging it. Off by default so the
    #: divergence is visible rather than smoothed away.
    logging_nan_inf_filter: bool = False
    logging_strategy: Literal['steps', 'epoch', 'no'] = 'steps'
    disable_tqdm: Optional[bool] = None

    # Not migrated: HF's `log_level`, which sets the trainer's own logger verbosity. DeployConfig
    # already has a `log_level` meaning the server's, and the two would collide the moment a CLI
    # flattens every config into one flag namespace. The server one is the one users actually pass.
