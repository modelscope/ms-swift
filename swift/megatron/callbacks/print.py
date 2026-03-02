# Copyright (c) ModelScope Contributors. All rights reserved.
import os
import time

import torch
from tqdm import tqdm

from swift.megatron.utils import reduce_max_stat_across_model_parallel_group
from swift.utils import JsonlWriter, format_time, get_logger, is_last_rank
from .base import MegatronCallback

logger = get_logger()


class PrintCallback(MegatronCallback):

    def __init__(self, trainer):
        super().__init__(trainer)
        self.training_bar = None
        self.eval_bar = None
        self.jsonl_writer = None
        self.throughput_writer = None
        self.is_write_rank = is_last_rank()

    def on_train_begin(self):
        self.training_bar = tqdm(
            total=self.args.train_iters, dynamic_ncols=True, disable=not self.is_write_rank, desc='Train: ')
        self.start_step = self.state.iteration
        self.training_bar.update(self.state.iteration)
        self.current_step = self.state.iteration
        self.start_time = time.time()
        logging_path = os.path.join(self.args.output_dir, 'logging.jsonl')
        throughput_path = os.environ.get('SWIFT_THROUGHPUT_JSONL') or os.path.join(
            self.args.output_dir, 'throughput_rank0.jsonl')
        logger.info(f'logging_path: {logging_path}')
        logger.info(f'throughput_path: {throughput_path}')
        self.jsonl_writer = JsonlWriter(logging_path, enable_async=True, write_on_rank='last')
        self.throughput_writer = JsonlWriter(throughput_path, enable_async=True, write_on_rank='master')

    def on_train_end(self):
        self.training_bar.close()
        self.training_bar = None

    def on_step_end(self):
        n_step = self.state.iteration - self.current_step
        self.current_step = self.state.iteration
        self.training_bar.update(n_step)

    def on_eval_begin(self):
        self.eval_bar = tqdm(
            total=self.args.eval_iters, dynamic_ncols=True, disable=not self.is_write_rank, desc='Evaluate: ')

    def on_eval_end(self):
        self.eval_bar.close()
        self.eval_bar = None

    def on_eval_step(self):
        self.eval_bar.update()

    def on_log(self, logs):
        state = self.state
        args = self.args
        logs['iteration'] = f'{state.iteration}/{args.train_iters}'
        elapsed = time.time() - self.start_time
        logs['elapsed_time'] = format_time(elapsed)
        n_steps = state.iteration - self.start_step
        train_speed = elapsed / n_steps if n_steps > 0 else 0.0
        logs['remaining_time'] = format_time((args.train_iters - state.iteration) * train_speed)
        memory = reduce_max_stat_across_model_parallel_group(torch.cuda.max_memory_reserved() / 1024**3)
        logs['memory(GiB)'] = round(memory, 2)
        logs['train_speed(s/it)'] = round(train_speed, 6)

        active_tokens_per_step = logs.get('tokens_active_per_step')
        total_tokens_per_step = logs.get('tokens_total_per_step')
        if isinstance(active_tokens_per_step, (int, float)) and isinstance(total_tokens_per_step, (int, float)):
            active_tokens_per_step = float(active_tokens_per_step)
            total_tokens_per_step = float(total_tokens_per_step)
            if train_speed > 0:
                logs['active_tps'] = active_tokens_per_step / train_speed
                logs['total_tps'] = total_tokens_per_step / train_speed
            if total_tokens_per_step > 0:
                logs['mask_ratio'] = 1.0 - (active_tokens_per_step / total_tokens_per_step)
            if self.throughput_writer is not None:
                throughput_event = {
                    'step': state.iteration,
                    'elapsed_s': elapsed,
                    'window_steps': n_steps,
                    'train_speed_s_per_it': train_speed,
                    'tokens_active_per_step': active_tokens_per_step,
                    'tokens_total_per_step': total_tokens_per_step,
                    'active_tps': logs.get('active_tps'),
                    'total_tps': logs.get('total_tps'),
                    'mask_ratio': logs.get('mask_ratio'),
                }
                self.throughput_writer.append(throughput_event)

        logs = {k: round(v, 8) if isinstance(v, float) else v for k, v in logs.items()}
        self.jsonl_writer.append(logs)
        if self.is_write_rank:
            self.training_bar.write(str(logs))
