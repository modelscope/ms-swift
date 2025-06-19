# Copyright (c) Alibaba, Inc. and its affiliates.
import os
import sys
from datetime import datetime

import torch

from swift.llm import git_clone_github
from swift.utils import get_logger, is_megatron_available, safe_ddp_context, subprocess_run

logger = get_logger()


def _patch_transformer_engine():
    import transformer_engine
    try:
        from transformer_engine.pytorch.attention import apply_rotary_pos_emb
    except ImportError:
        try:
            transformer_engine.pytorch.attention.apply_rotary_pos_emb = (
                transformer_engine.pytorch.attention.rope.apply_rotary_pos_emb)
            logger.info('Patch apply_rotary_pos_emb successfully applied.')
        except (ImportError, AttributeError):
            pass
    try:
        from transformer_engine.pytorch.attention import _SplitAlongDim
    except ImportError:
        try:
            transformer_engine.pytorch.attention._SplitAlongDim = (transformer_engine.pytorch.utils.SplitAlongDim)
            logger.info('Patch _SplitAlongDim successfully applied.')
        except (ImportError, AttributeError):
            pass


def _patch__batched_p2p_ops():
    from megatron.core.pipeline_parallel import p2p_communication

    _batched_p2p_ops_origin = p2p_communication._batched_p2p_ops

    def _batched_p2p_ops(**kwargs):
        kwargs['group'] = None
        return _batched_p2p_ops_origin(**kwargs)

    p2p_communication._batched_p2p_ops = _batched_p2p_ops


def _patch_training_log():
    # TODO: support swanlab
    from megatron.core import mpu
    from megatron.core.transformer.moe.moe_utils import track_moe_metrics
    from megatron.training.theoretical_memory_usage import report_theoretical_memory
    from megatron.core.transformer.multi_token_prediction import MTPLossLoggingHelper
    from megatron.training import (training, get_args, get_timers, get_tensorboard_writer, get_wandb_writer,
                                   get_one_logger, one_logger_utils, is_last_rank, print_rank_last)
    from megatron.training.training import num_floating_point_operations
    from megatron.core.num_microbatches_calculator import get_num_microbatches
    from megatron.training.utils import reduce_max_stat_across_model_parallel_group, report_memory

    # Code borrowed from NVIDIA/Megatron-LM
    def training_log(loss_dict, total_loss_dict, learning_rate, decoupled_learning_rate, iteration, loss_scale,
                     report_memory_flag, skipped_iter, grad_norm, params_norm, num_zeros_in_grad):
        """Log training information such as losses, timing, ...."""
        args = get_args()
        timers = get_timers()
        writer = get_tensorboard_writer()
        wandb_writer = get_wandb_writer()

        # Advanced, skipped, and Nan iterations.
        advanced_iters_key = 'advanced iterations'
        skipped_iters_key = 'skipped iterations'
        nan_iters_key = 'nan iterations'
        # Advanced iterations.
        if not skipped_iter:
            total_loss_dict[advanced_iters_key] = total_loss_dict.get(advanced_iters_key, 0) + 1
        else:
            if advanced_iters_key not in total_loss_dict:
                total_loss_dict[advanced_iters_key] = 0
        # Skipped iterations.
        total_loss_dict[skipped_iters_key] = total_loss_dict.get(skipped_iters_key, 0) + skipped_iter
        # Update losses and set nan iterations
        got_nan = False
        for key in loss_dict:
            if not skipped_iter:
                total_loss_dict[key] = total_loss_dict.get(key, torch.tensor([0.0], dtype=torch.float,
                                                                             device='cuda')) + loss_dict[key]
            else:
                value = loss_dict[key].float().sum().item()
                is_nan = value == float('inf') or value == -float('inf') or value != value
                got_nan = got_nan or is_nan
        total_loss_dict[nan_iters_key] = total_loss_dict.get(nan_iters_key, 0) + int(got_nan)

        # Logging.
        timers_to_log = [
            'forward-backward', 'forward-compute', 'backward-compute', 'batch-generator', 'forward-recv',
            'forward-send', 'backward-recv', 'backward-send', 'forward-send-forward-recv', 'forward-send-backward-recv',
            'backward-send-forward-recv', 'backward-send-backward-recv', 'forward-backward-send-forward-backward-recv',
            'layernorm-grads-all-reduce', 'embedding-grads-all-reduce', 'all-grads-sync', 'params-all-gather',
            'optimizer-copy-to-main-grad', 'optimizer-unscale-and-check-inf', 'optimizer-clip-main-grad',
            'optimizer-count-zeros', 'optimizer-inner-step', 'optimizer-copy-main-to-model-params', 'optimizer'
        ]

        # Calculate batch size.
        batch_size = args.micro_batch_size * args.data_parallel_size * get_num_microbatches()

        # Track app tag & app tag ID
        one_logger_utils.track_app_tag(batch_size, args.world_size, args.seq_length)

        total_iterations = total_loss_dict[advanced_iters_key] + total_loss_dict[skipped_iters_key]

        # learning rate will be None on ranks without trainable params, so we must gather across mp ranks
        learning_rate = reduce_max_stat_across_model_parallel_group(learning_rate)
        # Tensorboard values.
        # Timer requires all the ranks to call.
        if args.log_timers_to_tensorboard and (iteration % args.tensorboard_log_interval == 0):
            timers.write(timers_to_log, writer, iteration, normalizer=total_iterations)
        if writer and (iteration % args.tensorboard_log_interval == 0):
            if wandb_writer:
                wandb_writer.log({'samples vs steps': args.consumed_train_samples}, iteration)
            writer.add_scalar('learning-rate', learning_rate, iteration)
            writer.add_scalar('learning-rate vs samples', learning_rate, args.consumed_train_samples)
            if wandb_writer:
                wandb_writer.log({'learning-rate': learning_rate}, iteration)
            if args.decoupled_lr is not None:
                writer.add_scalar('decoupled-learning-rate', decoupled_learning_rate, iteration)
            if args.skipped_train_samples > 0:
                writer.add_scalar('skipped-train-samples', args.skipped_train_samples, iteration)
                if wandb_writer:
                    wandb_writer.log({'skipped-train-samples': args.skipped_train_samples}, iteration)
            writer.add_scalar('batch-size', batch_size, iteration)
            writer.add_scalar('batch-size vs samples', batch_size, args.consumed_train_samples)
            if wandb_writer:
                wandb_writer.log({'batch-size': batch_size}, iteration)
            for key in loss_dict:
                writer.add_scalar(key, loss_dict[key], iteration)
                writer.add_scalar(key + ' vs samples', loss_dict[key], args.consumed_train_samples)
                if wandb_writer:
                    wandb_writer.log({key: loss_dict[key]}, iteration)
            if args.log_loss_scale_to_tensorboard:
                writer.add_scalar('loss-scale', loss_scale, iteration)
                writer.add_scalar('loss-scale vs samples', loss_scale, args.consumed_train_samples)
                if wandb_writer:
                    wandb_writer.log({'loss-scale': loss_scale}, iteration)
            if args.log_world_size_to_tensorboard:
                writer.add_scalar('world-size', args.world_size, iteration)
                writer.add_scalar('world-size vs samples', args.world_size, args.consumed_train_samples)
                if wandb_writer:
                    wandb_writer.log({'world-size': args.world_size}, iteration)
            if grad_norm is not None:
                writer.add_scalar('grad-norm', grad_norm, iteration)
                writer.add_scalar('grad-norm vs samples', grad_norm, args.consumed_train_samples)
                if wandb_writer:
                    wandb_writer.log({'grad-norm': grad_norm}, iteration)
            if num_zeros_in_grad is not None:
                writer.add_scalar('num-zeros', num_zeros_in_grad, iteration)
                writer.add_scalar('num-zeros vs samples', num_zeros_in_grad, args.consumed_train_samples)
                if wandb_writer:
                    wandb_writer.log({'num-zeros': num_zeros_in_grad}, iteration)
            if params_norm is not None:
                writer.add_scalar('params-norm', params_norm, iteration)
                writer.add_scalar('params-norm vs samples', params_norm, args.consumed_train_samples)
                if wandb_writer:
                    wandb_writer.log({'params-norm': params_norm}, iteration)
            if args.log_memory_to_tensorboard:
                mem_stats = torch.cuda.memory_stats()
                writer.add_scalar(
                    'mem-reserved-bytes',
                    mem_stats['reserved_bytes.all.current'],
                    iteration,
                )
                writer.add_scalar(
                    'mem-allocated-bytes',
                    mem_stats['allocated_bytes.all.current'],
                    iteration,
                )
                writer.add_scalar(
                    'mem-max-allocated-bytes',
                    mem_stats['allocated_bytes.all.peak'],
                    iteration,
                )
                writer.add_scalar(
                    'mem-allocated-count',
                    mem_stats['allocation.all.current'],
                    iteration,
                )
        if args.num_experts is not None:
            moe_loss_scale = 1 / get_num_microbatches()
            track_names = []
            if args.moe_router_load_balancing_type in ['aux_loss', 'seq_aux_loss']:
                track_names.append('load_balancing_loss')
            if args.moe_z_loss_coeff is not None:
                track_names.append('z_loss')
            track_moe_metrics(
                loss_scale=moe_loss_scale,
                iteration=iteration,
                writer=writer,
                wandb_writer=wandb_writer,
                total_loss_dict=total_loss_dict,
                per_layer_logging=args.moe_per_layer_logging,
                force_initialize=True,
                track_names=track_names,
                num_layers=args.num_layers,
                moe_layer_freq=args.moe_layer_freq)
        if args.mtp_num_layers is not None:
            mtp_loss_scale = 1 / get_num_microbatches()
            MTPLossLoggingHelper.track_mtp_metrics(mtp_loss_scale, iteration, writer, wandb_writer, total_loss_dict)
        if iteration % args.log_interval == 0 or iteration == 1:
            if args.record_memory_history and is_last_rank():
                snapshot = torch.cuda.memory._snapshot()
                from pickle import dump
                with open(args.memory_snapshot_path, 'wb') as f:
                    dump(snapshot, f)

            elapsed_time = timers('interval-time').elapsed(barrier=True)
            elapsed_time_per_iteration = elapsed_time / total_iterations

            throughput = num_floating_point_operations(args, batch_size) / (
                elapsed_time_per_iteration * 10**12 * args.world_size)

            one_logger_utils.track_e2e_metrics(args.log_throughput, throughput)

            if args.log_timers_to_tensorboard:
                if writer:
                    writer.add_scalar('iteration-time', elapsed_time_per_iteration, iteration)
                if wandb_writer:
                    wandb_writer.log({'iteration-time': elapsed_time_per_iteration}, iteration)
            log_string = f" [{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]"
            log_string += ' iteration {:8d}/{:8d} |'.format(iteration, args.train_iters)
            log_string += ' consumed samples: {:12d} |'.format(args.consumed_train_samples)
            if args.skipped_train_samples > 0:
                log_string += ' skipped samples: {:12d} |'.format(args.skipped_train_samples)
            log_string += ' elapsed time per iteration (ms): {:.1f} |'.format(elapsed_time_per_iteration * 1000.0)
            if args.log_throughput:
                log_string += f' throughput per GPU (TFLOP/s/GPU): {throughput:.1f} |'
                if args.log_timers_to_tensorboard:
                    if writer:
                        writer.add_scalar('throughput', throughput, iteration)
                    if wandb_writer:
                        wandb_writer.log({'throughput': throughput}, iteration)
            # Decoupled_learning_rate should be not None only on first and last pipeline stage.
            log_string += f' learning rate: {learning_rate:.6E} |'
            if args.decoupled_lr is not None and (mpu.is_pipeline_first_stage(ignore_virtual=True)
                                                  or mpu.is_pipeline_last_stage(ignore_virtual=True)):
                assert decoupled_learning_rate is not None
                log_string += f' decoupled learning rate: {decoupled_learning_rate:.6E} |'
            else:
                assert decoupled_learning_rate is None
            log_string += f' global batch size: {batch_size:5d} |'
            for key in total_loss_dict:
                if key not in [advanced_iters_key, skipped_iters_key, nan_iters_key]:
                    avg = total_loss_dict[key].item() / float(max(1, total_loss_dict[advanced_iters_key]))
                    log_string += ' {}: {:.6E} |'.format(key, avg)
                    total_loss_dict[key] = torch.tensor([0.0], dtype=torch.float, device='cuda')
            log_string += f' loss scale: {loss_scale:.1f} |'
            if grad_norm is not None:
                log_string += f' grad norm: {grad_norm:.3f} |'
            if num_zeros_in_grad is not None:
                log_string += f' num zeros: {num_zeros_in_grad} |'
            if params_norm is not None:
                log_string += f' params norm: {params_norm:.3f} |'
            log_string += ' number of skipped iterations: {:3d} |'.format(total_loss_dict[skipped_iters_key])
            log_string += ' number of nan iterations: {:3d} |'.format(total_loss_dict[nan_iters_key])
            total_loss_dict[advanced_iters_key] = 0
            total_loss_dict[skipped_iters_key] = 0
            total_loss_dict[nan_iters_key] = 0
            print_rank_last(log_string)
            if report_memory_flag:
                # Report memory after optimizer state has been initialized.
                if torch.distributed.get_rank() == 0:
                    num_microbatches = get_num_microbatches()
                    report_theoretical_memory(args, num_microbatches=num_microbatches, verbose=True)
                report_memory(f'(after {iteration} iterations)')
                report_memory_flag = False
            timers.log(timers_to_log, normalizer=args.log_interval)

        return report_memory_flag

    training.training_log = training_log


def _patch_megatron():
    _patch_transformer_engine()
    _patch__batched_p2p_ops()
    try:
        _patch_training_log()
        logger.info('Patch training_log successfully applied.')
    except Exception:
        pass


def init_megatron_env() -> None:
    if 'MEGATRON_LM_PATH' not in os.environ:
        os.environ['MEGATRON_LM_PATH'] = git_clone_github(
            'https://github.com/NVIDIA/Megatron-LM', branch='core_r0.12.0')
    with safe_ddp_context(hash_id='megatron-lm'):
        if not is_megatron_available():
            subprocess_run([sys.executable, '-m', 'pip', 'install', '-e', os.environ['MEGATRON_LM_PATH']])
    sys.path.insert(0, os.environ['MEGATRON_LM_PATH'])
    _patch_megatron()
