# Copyright (c) ModelScope Contributors. All rights reserved.
import os
import time
import torch
from tqdm import tqdm

from swift.megatron.utils import reduce_max_stat_across_model_parallel_group
from swift.utils import JsonlWriter, format_time, get_env_args, get_logger, is_last_rank
from .base import MegatronCallback

logger = get_logger()


class PrintCallback(MegatronCallback):

    def __init__(self, trainer):
        super().__init__(trainer)
        self.training_bar = None
        self.eval_bar = None
        self.jsonl_writer = None
        self.is_write_rank = is_last_rank()
        self.device_peak_tflops = get_env_args('DEVICE_TFLOPS', float, None)
        if self.device_peak_tflops is not None:
            logger.info(
                f"Specify theoretical max TFLOPS through ENV 'DEVICE_TFLOPS'. [{self.device_peak_tflops} TFLOPS]")
            self.device_peak_tflops = float(self.device_peak_tflops)

    def on_train_begin(self):
        self.training_bar = tqdm(
            total=self.args.train_iters, dynamic_ncols=True, disable=not self.is_write_rank, desc='Train: ')
        self.start_step = self.state.iteration
        self.training_bar.update(self.state.iteration)
        self.current_step = self.state.iteration
        self.start_time = time.time()
        logging_path = os.path.join(self.args.output_dir, 'logging.jsonl')
        logger.info(f'logging_path: {logging_path}')
        self.jsonl_writer = JsonlWriter(logging_path, enable_async=True, write_on_rank='last')

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
        seq_length = getattr(args, 'seq_length', None)
        world_size = getattr(args, 'world_size', None) or 1
        if train_speed > 0 and seq_length:
            logs['tgs'] = round(
                args.global_batch_size * args.seq_length / train_speed / world_size, 3)
            if self.device_peak_tflops:
                throughput = self._get_throughput_tflops_per_gpu(train_speed)
                if throughput is not None:
                    logs['throughput(TFLOP/s/GPU)'] = round(throughput, 3)
                    logs['MFU'] = round(throughput / self.device_peak_tflops, 6)
        logs['remaining_time'] = format_time((args.train_iters - state.iteration) * train_speed)
        memory = reduce_max_stat_across_model_parallel_group(torch.cuda.max_memory_reserved() / 1024**3)
        logs['memory(GiB)'] = round(memory, 2)
        logs['train_speed(s/it)'] = round(train_speed, 6)
        logs = {k: round(v, 8) if isinstance(v, float) else v for k, v in logs.items()}
        self.jsonl_writer.append(logs)
        if self.is_write_rank:
            self.training_bar.write(str(logs))

    def _get_throughput_tflops_per_gpu(self, train_speed):
        if train_speed <= 0:
            return None
        world_size = getattr(self.args, 'world_size', None) or 1
        num_flops = self._num_floating_point_operations(self.args.global_batch_size)
        if num_flops is None:
            return None
        return num_flops / (train_speed * 10**12 * world_size)

    def _num_floating_point_operations(self, batch_size):
        seq_length = getattr(self.args, 'seq_length', None)
        if seq_length is None:
            return None
        config = self.trainer.config
        hidden_size = getattr(config, 'hidden_size', None)
        num_layers = getattr(config, 'num_layers', None)
        num_attention_heads = getattr(config, 'num_attention_heads', None)
        ffn_hidden_size = getattr(config, 'ffn_hidden_size', None)
        if None in {hidden_size, num_layers, num_attention_heads, ffn_hidden_size}:
            return None

        kv_channels = getattr(config, 'kv_channels', None) or hidden_size // num_attention_heads
        num_query_groups = getattr(config, 'num_query_groups', None) or num_attention_heads
        padded_vocab_size = getattr(config, 'padded_vocab_size', None) or getattr(config, 'vocab_size', None)
        if padded_vocab_size is None:
            return None

        query_projection_size = kv_channels * num_attention_heads
        query_projection_to_hidden_size_ratio = query_projection_size / hidden_size
        num_experts_routed_to = 1 if getattr(config, 'num_moe_experts', None) is None else getattr(
            config, 'moe_router_topk', 1)
        gated_linear_multiplier = 1.5 if getattr(config, 'swiglu', False) else 1.0

        return (
            12 * batch_size * seq_length * num_layers * hidden_size * hidden_size * (
                (1 + (num_query_groups / num_attention_heads) + (seq_length / hidden_size))
                * query_projection_to_hidden_size_ratio
                + (ffn_hidden_size / hidden_size) * num_experts_routed_to * gated_linear_multiplier
                + padded_vocab_size / (2 * num_layers * hidden_size)))
