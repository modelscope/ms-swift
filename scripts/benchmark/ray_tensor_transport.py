# Copyright (c) ModelScope Contributors. All rights reserved.
"""Native Ray CPU tensor replay; set RAY_ENABLE_ZERO_COPY_TORCH_TENSORS before launch."""
import argparse
import json
import os
import statistics
import tempfile
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--payload-mib', type=int, default=64)
    parser.add_argument('--rounds', type=int, default=6)
    parser.add_argument('--gpu', action='store_true', help='Also verify native accelerator-preparation ownership')
    args = parser.parse_args()
    if args.payload_mib < 1 or args.rounds < 1:
        parser.error('--payload-mib and --rounds must be positive')
    os.environ.setdefault('RAY_USAGE_STATS_ENABLED', '0')
    import ray
    import torch
    from ray._private import ray_constants

    from swift.ray.megatron.grpo_trainer import GRPOTrainer
    from swift.rl_core.data import GRPOBatch
    from swift.rlhf_trainers.gkd_loss import TeacherOutput

    flag = 'RAY_ENABLE_ZERO_COPY_TORCH_TENSORS'
    if not hasattr(ray_constants, flag):
        parser.error('This replay requires Ray >= 2.53 with native zero-copy tensor support')
    enabled = getattr(ray_constants, flag)
    value = os.environ.get(flag, '0')
    if args.gpu and not torch.cuda.is_available():
        parser.error('--gpu requires an available CUDA device and the Megatron runtime dependencies')
    torch.set_num_threads(1)

    @ray.remote(num_cpus=1)
    class Consumer:

        def __init__(self):
            torch.set_num_threads(1)

        def inspect(self, refs):
            from ray._private import ray_constants

            first = ray.get(refs[0])
            second = ray.get(refs[0])
            shared = first['features'].data_ptr() == second['features'].data_ptr()
            assert shared == enabled, (shared, enabled)
            assert getattr(ray_constants, flag) == enabled
            assert os.environ.get(flag) == value
            assert first['features'].dtype == torch.bfloat16
            assert not first['features'].requires_grad
            assert torch.equal(first['features'], torch.ones_like(first['features']))
            for name, tensor in first['matrix'].items():
                expected = torch.arange(12).reshape(3, 4).to(tensor.dtype)
                assert torch.equal(tensor, expected), name
                assert not tensor.requires_grad
            assert first['noncontiguous'].shape == (4, 3)
            assert torch.equal(first['noncontiguous'], torch.arange(12).reshape(3, 4).t())
            assert first['scalar'].shape == () and first['scalar'].item() == 1.5
            assert first['empty'].shape == (0, 4)
            assert first['alias'] is first['features']
            owned = first['features'].clone()
            owned.add_(1)
            assert owned.data_ptr() != first['features'].data_ptr()
            assert torch.equal(second['features'], torch.ones_like(second['features']))
            return {'shared_cpu_storage': shared, 'value': os.environ.get(flag)}

        def consume(self, refs):
            data = ray.get(refs[0])
            return data['features'].sum(dtype=torch.float32).item()

        def prepare_on_gpu(self, refs):
            from types import SimpleNamespace

            from swift.megatron.trainers.utils import get_batch_on_this_pp_rank
            from swift.utils import get_current_device

            data = ray.get(refs[0])
            original_labels = data['labels'].clone()
            batch = data.pop('grpo_batch')
            teacher = data.pop('teacher_output')
            data.pop('matrix')
            data.pop('noncontiguous')
            data.pop('scalar')
            data.pop('empty')
            prepared = get_batch_on_this_pp_rank(
                SimpleNamespace(task_type='causal_lm', pipeline_model_parallel_size=1), data)
            device = get_current_device()
            batch.to_device(device)
            teacher.to_device(device)
            assert prepared['features'].is_cuda and batch.completion_mask.is_cuda and teacher.full_logits.is_cuda
            assert torch.equal(prepared['labels'].cpu(), torch.roll(original_labels, -1, dims=-1))
            prepared['features'].add_(1)
            batch.completion_mask.logical_not_()
            teacher.full_logits.add_(1)
            torch.cuda.synchronize()
            fresh = ray.get(refs[0])
            assert torch.equal(fresh['features'], torch.ones_like(fresh['features']))
            assert fresh['grpo_batch'].completion_mask.all().item()
            assert fresh['teacher_output'].full_logits.eq(0).all().item()
            assert torch.equal(fresh['labels'], original_labels)
            return True

    size = args.payload_mib * (1 << 20) // 2
    features = torch.ones(size, dtype=torch.bfloat16)
    labels = torch.arange(32).reshape(2, 16)
    payload = {
        'features': features,
        'alias': features,
        'labels': labels,
        'matrix': {
            str(dtype): torch.arange(12).reshape(3, 4).to(dtype)
            for dtype in [
                torch.bool, torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64, torch.float16,
                torch.bfloat16, torch.float32, torch.float64
            ]
        },
        'noncontiguous': torch.arange(12).reshape(3, 4).t(),
        'scalar': torch.tensor(1.5, dtype=torch.bfloat16),
        'empty': torch.empty((0, 4), dtype=torch.bfloat16),
    }
    if args.gpu:
        payload['grpo_batch'] = GRPOBatch(
            completion_mask=torch.ones((2, 16), dtype=torch.bool),
            truncated_mask=torch.zeros(2, dtype=torch.bool),
            seq_lengths=torch.tensor([16, 16]))
        payload['teacher_output'] = TeacherOutput(full_logits=torch.zeros((2, 16, 4)), labels=labels)
    actors = []
    with tempfile.TemporaryDirectory(prefix='swift-zt-', dir='/tmp') as temp_dir:
        try:
            ray.init(
                address='local',
                _node_ip_address='127.0.0.1',
                num_cpus=2,
                num_gpus=int(args.gpu),
                include_dashboard=False,
                object_store_memory=max(80 << 20, args.payload_mib * (4 << 20)),
                _temp_dir=temp_dir,
                runtime_env={'env_vars': {
                    flag: value
                }})
            reader = Consumer.options(
                num_gpus=int(args.gpu), runtime_env={
                    'env_vars': {
                        flag: value,
                        'LOCAL_RANK': '0'
                    }
                }).remote()
            observer = Consumer.options(runtime_env={'env_vars': {'TRANSPORT_ROLE': 'observer'}}).remote()
            actors.extend([reader, observer])
            warm_ref = ray.put(payload)
            ownership = ray.get([reader.inspect.remote([warm_ref]), observer.inspect.remote([warm_ref])], timeout=300)
            if args.gpu:
                assert ray.get(reader.prepare_on_gpu.remote([warm_ref]), timeout=300)
                ray.get(observer.inspect.remote([warm_ref]), timeout=300)

            rows = ray.get(ray.put([{'per_token_logps': torch.full((16, ), float(i))} for i in range(2)]))
            batch = GRPOBatch(
                torch.ones((2, 16), dtype=torch.bool), torch.zeros(2, dtype=torch.bool), torch.tensor([16, 16]))
            GRPOTrainer._scatter_logps([batch], rows, 'old_per_token_logps')
            batch.old_per_token_logps.add_(1)
            assert rows[0]['per_token_logps'].eq(0).all().item() and rows[1]['per_token_logps'].eq(1).all().item()

            assert ray.get(reader.consume.remote([warm_ref]), timeout=300) == float(size)
            put_seconds, rpc_seconds = [], []
            for _ in range(args.rounds):
                start = time.perf_counter()
                ref = ray.put(payload)
                put_seconds.append(time.perf_counter() - start)
                start = time.perf_counter()
                checksum = ray.get(reader.consume.remote([ref]), timeout=300)
                rpc_seconds.append(time.perf_counter() - start)
                assert checksum == float(size)
            print(
                'RESULT ' + json.dumps(
                    {
                        'ray': ray.__version__,
                        'torch': torch.__version__,
                        'enabled': enabled,
                        'environment_value': value,
                        'payload_mib': args.payload_mib,
                        'rounds': args.rounds,
                        'ownership': ownership,
                        'native_scatter_isolated': True,
                        'native_gpu_preparation_isolated': args.gpu,
                        'put_seconds': put_seconds,
                        'rpc_seconds': rpc_seconds,
                        'median_put_seconds': statistics.median(put_seconds),
                        'median_rpc_seconds': statistics.median(rpc_seconds),
                        'median_total_seconds': statistics.median([a + b for a, b in zip(put_seconds, rpc_seconds)]),
                    }),
                flush=True)
        finally:
            for actor in actors:
                try:
                    ray.kill(actor, no_restart=True)
                except Exception:
                    pass
            ray.shutdown()


if __name__ == '__main__':
    main()
