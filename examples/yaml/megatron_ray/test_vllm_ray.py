#!/usr/bin/env python
"""Quick integration test for vLLM Ray support.

Tests:
  1. VllmWorker can init and generate text
  2. MegatronWorker + co-located vLLM
  3. Weight sync from Megatron → vLLM

Usage:
  CUDA_VISIBLE_DEVICES=0,1 python examples/yaml/megatron_ray/test_vllm_ray.py
"""
import os
import sys

os.environ.setdefault('CUDA_DEVICE_MAX_CONNECTIONS', '1')


def test_vllm_worker_separated():
    """Test 1: VllmWorker in separated mode."""
    import ray

    from swift.ray.megatron.resource_pool import ResourcePool, ResourcePoolManager
    from swift.ray.megatron.vllm_worker import VllmWorker
    from swift.ray.megatron.worker_group import WorkerGroup

    print('\n' + '=' * 60)
    print('TEST 1: VllmWorker separated mode')
    print('=' * 60)

    ray.init(ignore_reinit_error=True)

    pool = ResourcePool([2])
    pool.create()

    RemoteVllm = ray.remote(num_gpus=1)(VllmWorker)
    wg = WorkerGroup.from_pool('rollout', pool, worker_cls=RemoteVllm)

    argv = [
        '--model',
        'Qwen/Qwen2.5-0.5B',
        '--max_length',
        '512',
        '--max_completion_length',
        '64',
        '--temperature',
        '0.7',
    ]
    wg.broadcast('init_model', argv)
    wg.build_dispatch_info(worker_cls=VllmWorker)

    pings = wg.ping()
    print('Pings:', pings)

    prompts = [
        {
            'messages': [{
                'role': 'user',
                'content': 'Hello, who are you?'
            }]
        },
        {
            'messages': [{
                'role': 'user',
                'content': 'What is 2+2?'
            }]
        },
    ]
    outputs = wg.generate(prompts)
    if outputs:
        print(f'Generated {len(outputs)} outputs:')
        for i, out in enumerate(outputs):
            text = out.response.choices[0].message.content if out.response else '(empty)'
            print(f'  [{i}] {text[:100]}...')
    else:
        print('No outputs (expected on non-collector ranks)')

    pool.destroy()
    ray.shutdown()
    print('TEST 1 PASSED\n')


def test_colocated_vllm():
    """Test 2: MegatronWorker with co-located vLLM engine.

    Uses DPO rlhf_type for simpler training init, while testing
    the vLLM co-locate infrastructure (init_vllm + generate + weight sync).
    """
    import ray

    from swift.ray.megatron.megatron_worker import MegatronWorker
    from swift.ray.megatron.resource_pool import ResourcePool
    from swift.ray.megatron.worker_group import WorkerGroup

    print('\n' + '=' * 60)
    print('TEST 2: Co-located MegatronWorker + vLLM')
    print('=' * 60)

    ray.init(ignore_reinit_error=True)

    pool = ResourcePool([2])
    pool.create()

    wg = WorkerGroup.from_pool('train', pool)

    argv = [
        '--model',
        'Qwen/Qwen2.5-0.5B',
        '--dataset',
        'hjh0119/shareAI-Llama3-DPO-zh-en-emoji',
        '--max_length',
        '512',
        '--max_completion_length',
        '64',
        '--temperature',
        '0.7',
        '--rlhf_type',
        'dpo',
        '--micro_batch_size',
        '1',
        '--global_batch_size',
        '2',
        '--padding_free',
        'false',
        '--finetune',
        'true',
        '--cross_entropy_loss_fusion',
        'true',
        '--lr',
        '1e-5',
        '--attention_backend',
        'flash',
        '--tuner_type',
        'full',
        '--tensor_model_parallel_size',
        '2',
        '--sequence_parallel',
        'true',
    ]
    wg.broadcast('init_model', argv, trainable=True)
    print('Megatron model initialized')

    wg.broadcast('init_vllm')
    print('vLLM engine initialized')

    wg.build_dispatch_info(worker_cls=MegatronWorker)

    pings = wg.ping()
    print('Pings:', pings)

    prompts = [
        {
            'messages': [{
                'role': 'user',
                'content': 'Tell me a joke.'
            }]
        },
    ]
    outputs = wg.generate(prompts)
    if outputs:
        print(f'Generated {len(outputs)} outputs:')
        for i, out in enumerate(outputs):
            text = out.response.choices[0].message.content if out.response else '(empty)'
            print(f'  [{i}] {text[:100]}...')

    result = wg.sync_weights_to_vllm()
    print('Weight sync result:', result)

    pool.destroy()
    ray.shutdown()
    print('TEST 2 PASSED\n')


if __name__ == '__main__':
    test_name = sys.argv[1] if len(sys.argv) > 1 else 'all'

    if test_name in ('all', '1', 'separated'):
        test_vllm_worker_separated()

    if test_name in ('all', '2', 'colocated'):
        test_colocated_vllm()

    print('\nAll tests passed!')
