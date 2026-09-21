"""Lightweight experiment tracking shared by all dev training loops."""
from __future__ import annotations
import math
import os
from dataclasses import asdict
from typing import TYPE_CHECKING, Any, Dict, Iterable, Optional

import json

if TYPE_CHECKING:
    from swift.dev.config import LoggingConfig

_SUPPORTED_REPORTERS = frozenset({'tensorboard', 'wandb', 'swanlab'})
_SECRET_FIELDS = {'swanlab_token', 'swanlab_secret', 'swanlab_webhook_url'}


class RunTracker:
    """Own tracker setup, metric writes, and teardown for one training loop."""

    def __init__(self, config: Optional['LoggingConfig'], output_dir: str):
        self.config = config
        self.output_dir = output_dir
        self._writers: Dict[str, Any] = {}
        self._loss_total = 0.0
        self._loss_count = 0
        self._logged_prompts = set()
        if config is None or not _is_main_process():
            return
        reporters = {name.lower() for name in config.report_to}
        reporters.discard('none')
        unknown = reporters - _SUPPORTED_REPORTERS
        if unknown:
            raise ValueError(f'Unsupported report_to values: {sorted(unknown)}. Supported: '
                             f'{sorted(_SUPPORTED_REPORTERS)} or "none".')
        for reporter in sorted(reporters):
            getattr(self, f'_setup_{reporter}')()

    def should_log(self, step: int, *, epoch_end: bool = False) -> bool:
        if self.config is None or self.config.logging_strategy == 'no':
            return False
        if self.config.logging_strategy == 'epoch':
            return epoch_end
        return bool((self.config.logging_first_step and step == 1)
                    or (self.config.logging_steps and step % self.config.logging_steps == 0))

    def log(self, metrics: Dict[str, Any], step: int, *, epoch_end: bool = False) -> Dict[str, Any]:
        """Filter and write one metric record; returns the record actually emitted."""
        filtered = self._filter_metrics(metrics)
        if not self.should_log(step, epoch_end=epoch_end):
            return filtered
        scalars = {key: value for key, value in filtered.items() if key != 'step' and isinstance(value, (int, float))}
        writer = self._writers.get('tensorboard')
        if writer is not None:
            for key, value in scalars.items():
                writer.add_scalar(key, value, step)
        writer = self._writers.get('wandb')
        if writer is not None:
            writer.log(scalars, step=step)
        writer = self._writers.get('swanlab')
        if writer is not None:
            writer.log(scalars, step=step)
        return filtered

    def log_prompts(self, prompts: Iterable[Any], samples: Iterable[Any], step: int, num_generations: int = 1) -> None:
        """Log each distinct prompt/completion pair once when W&B prompt logging is enabled."""
        if not self.config or not self.config.wandb_log_unique_prompts:
            return
        writer = self._writers.get('wandb')
        if writer is None:
            return
        prompts = list(prompts)
        rows = []
        for index, sample in enumerate(samples):
            if not prompts:
                break
            prompt = prompts[min(index // max(1, num_generations), len(prompts) - 1)]
            prompt_key = json.dumps(prompt, ensure_ascii=False, sort_keys=True, default=str)
            if prompt_key in self._logged_prompts:
                continue
            self._logged_prompts.add(prompt_key)
            rows.append([prompt_key, getattr(sample, 'decoded', str(sample))])
        if rows:
            writer.log({'completions': writer.Table(columns=['prompt', 'completion'], data=rows)}, step=step)

    def close(self) -> None:
        writer = self._writers.get('tensorboard')
        if writer is not None:
            writer.close()
        writer = self._writers.get('wandb')
        if writer is not None:
            writer.finish()
        writer = self._writers.get('swanlab')
        if writer is not None:
            finish = getattr(writer, 'finish', None)
            if finish is not None:
                finish()
        self._writers.clear()

    def _filter_metrics(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        result = dict(metrics)
        loss = result.get('loss')
        if isinstance(loss, (int, float)):
            if math.isfinite(loss):
                self._loss_total += float(loss)
                self._loss_count += 1
            elif self.config is not None and self.config.logging_nan_inf_filter and self._loss_count:
                result['loss'] = self._loss_total / self._loss_count
        return result

    def _setup_tensorboard(self) -> None:
        try:
            from torch.utils.tensorboard import SummaryWriter
        except ImportError as exc:
            raise ImportError('report_to includes tensorboard; install tensorboard to enable it.') from exc
        config = self.config
        log_dir = config.tensorboard_dir or config.logging_dir or os.path.join(self.output_dir, 'runs')
        self._writers['tensorboard'] = SummaryWriter(log_dir=log_dir, max_queue=config.tensorboard_queue_size)

    def _setup_wandb(self) -> None:
        try:
            import wandb
        except ImportError as exc:
            raise ImportError('report_to includes wandb; install wandb to enable it.') from exc
        config = self.config
        wandb.init(
            dir=config.logging_dir or self.output_dir,
            name=config.wandb_exp_name or config.run_name,
            project=config.wandb_project,
            config=_public_config(config),
        )
        self._writers['wandb'] = wandb

    def _setup_swanlab(self) -> None:
        try:
            import swanlab
        except ImportError as exc:
            raise ImportError('report_to includes swanlab; install swanlab to enable it.') from exc
        config = self.config
        if config.swanlab_token:
            swanlab.login(config.swanlab_token)
        _register_swanlab_notification(swanlab, config)
        swanlab.init(
            project=config.swanlab_project,
            workspace=config.swanlab_workspace,
            experiment_name=config.swanlab_exp_name or config.run_name,
            logdir=config.logging_dir or self.output_dir,
            mode=config.swanlab_mode,
            config=_public_config(config),
        )
        self._writers['swanlab'] = swanlab


def _is_main_process() -> bool:
    return int(os.environ.get('RANK', '0')) == 0


def _public_config(config: 'LoggingConfig') -> Dict[str, Any]:
    return {key: value for key, value in asdict(config).items() if key not in _SECRET_FIELDS}


def _register_swanlab_notification(swanlab: Any, config: 'LoggingConfig') -> None:
    method = config.swanlab_notification_method
    if method is None:
        return
    from swanlab.plugin.notification import (
        DingTalkCallback,
        DiscordCallback,
        EmailCallback,
        LarkCallback,
        SlackCallback,
        WXWorkCallback,
    )
    callbacks = {
        'dingtalk': DingTalkCallback,
        'discord': DiscordCallback,
        'email': EmailCallback,
        'lark': LarkCallback,
        'slack': SlackCallback,
        'wxwork': WXWorkCallback,
    }
    callback_cls = callbacks.get(method)
    if callback_cls is None:
        raise ValueError(f'Unsupported swanlab_notification_method: {method!r}. Supported: {sorted(callbacks)}.')
    if method == 'email':
        required: Iterable[Optional[Any]] = (
            config.swanlab_sender_email,
            config.swanlab_receiver_email,
            config.swanlab_smtp_server,
            config.swanlab_smtp_port,
        )
        if not all(required):
            raise ValueError('SwanLab email notification requires sender_email, receiver_email, smtp_server, and '
                             'smtp_port.')
        callback = callback_cls(
            sender_email=config.swanlab_sender_email,
            receiver_email=config.swanlab_receiver_email,
            password=config.swanlab_secret,
            smtp_server=config.swanlab_smtp_server,
            port=config.swanlab_smtp_port,
            language=config.swanlab_email_language,
        )
    else:
        callback = callback_cls(webhook_url=config.swanlab_webhook_url, secret=config.swanlab_secret)
    swanlab.register_callbacks([callback])
