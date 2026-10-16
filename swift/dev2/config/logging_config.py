# Copyright (c) ModelScope Contributors. All rights reserved.
from dataclasses import dataclass, field
from typing import List, Literal, Optional


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

    # Email notification only; needs a reachable relay the webhook route does not.
    swanlab_smtp_server: Optional[str] = None
    swanlab_smtp_port: Optional[int] = None
    swanlab_sender_email: Optional[str] = None
    swanlab_receiver_email: Optional[str] = None
    swanlab_email_language: Optional[str] = 'zh'

    # === Weights & Biases ===
    # Naming 'wandb' in report_to enables it; credentials come from WANDB_API_KEY in the env.
    wandb_project: str = 'ms-swift'
    wandb_exp_name: Optional[str] = None
    # Log each distinct prompt once (Megatron RLHF repeats would otherwise drown the table).
    wandb_log_unique_prompts: Optional[bool] = None

    # === TensorBoard ===
    # Distinct from logging_dir (HF's own TB integration): this one is Megatron's writer.
    tensorboard_dir: Optional[str] = None
    tensorboard_queue_size: int = 50  # events buffered before a flush

    # === General Logging ===
    # The switch: 'tensorboard' / 'swanlab' / 'wandb' / 'none'; per-tracker blocks only say where.
    report_to: List[str] = field(default_factory=lambda: ['tensorboard'])
    logging_dir: Optional[str] = None
    run_name: Optional[str] = None
    logging_steps: int = 5
    logging_first_step: bool = True
    # Off by default so a nan/inf loss stays visible instead of being smoothed away.
    logging_nan_inf_filter: bool = False
    logging_strategy: Literal['steps', 'epoch', 'no'] = 'steps'

    # HF's log_level is not migrated: DeployConfig.log_level owns that name for the server.
