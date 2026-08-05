import os
import torch.distributed as dist
import torch.multiprocessing as mp
from datasets import Dataset as HfDataset
from torch.utils.data import DataLoader

PACKING_LENGTH = 2048
PACKING_INTERVAL = 128  # IterablePackingDataset default: raw samples consumed per packing round


def _collate(batch):
    return [row['idx'] for row in batch]


def _collate_packed(batch):
    # one packed sample: report which source samples it is made of
    return [row['idx'] for row in batch[0]]


def _build_stream(num_samples: int):
    return HfDataset.from_list([{'idx': i} for i in range(num_samples)]).to_iterable_dataset()


class _StubTemplate:
    """Stand-in for a Template: keeps the source index so packs can be traced back."""

    packing = False
    padding_free = False
    max_length = PACKING_LENGTH

    def encode(self, data, return_length=False):
        # lengths vary so that packs do not line up with the packing rounds
        length = 256 + 128 * (data['idx'] % 5)
        return {'idx': data['idx'], 'input_ids': [0] * length, 'labels': [0] * length, 'length': length}


def test_shard_state_is_bound_lazily():
    """The dataset is built before the data-parallel group is known, so binding must come later."""
    from swift.dataset import shard_streaming_dataset
    dataset, state = shard_streaming_dataset(_build_stream(10))
    assert [row['idx'] for row in dataset] == list(range(10))  # unbound: world_size 1, keeps everything
    state.bind(rank=1, world_size=3, block_size=1)
    assert [row['idx'] for row in dataset] == [1, 4, 7]
    state.bind(rank=0, world_size=2, block_size=3)
    assert [row['idx'] for row in dataset] == [0, 1, 2, 6, 7, 8]


def _worker(rank, world_size, port, cfg, queue):
    from swift.dataloader import DataLoaderDispatcher
    from swift.dataset import shard_streaming_dataset
    from swift.dataset.packing import IterablePackingDataset
    os.environ.update({'MASTER_ADDR': '127.0.0.1', 'MASTER_PORT': str(port)})
    dist.init_process_group('gloo', rank=rank, world_size=world_size)
    try:
        dataset = _build_stream(cfg['num_samples'])
        shard_state = None
        if cfg['shard']:
            dataset, shard_state = shard_streaming_dataset(dataset)
        if cfg['packing']:
            dataset = IterablePackingDataset(
                _StubTemplate(), dataset, num_proc=1, packing_length=PACKING_LENGTH, packing_strategy='sequential')
            base_dataloader = DataLoader(dataset, batch_size=1, collate_fn=_collate_packed)
        else:
            base_dataloader = DataLoader(
                dataset, batch_size=cfg['batch_size'], collate_fn=_collate, num_workers=cfg['num_workers'])
        dispatcher = DataLoaderDispatcher(
            base_dataloader, skip_batches=cfg.get('skip_batches', 0), shard_state=shard_state)
        queue.put((rank, [batch for batch in dispatcher]))
    finally:
        dist.destroy_process_group()


def _run(world_size: int, **cfg):
    cfg.setdefault('shard', True)
    cfg.setdefault('packing', False)
    cfg.setdefault('batch_size', 1)
    cfg.setdefault('num_workers', 0)
    ctx = mp.get_context('spawn')
    queue = ctx.Queue()
    port = 29500 + hash(tuple(sorted(cfg.items())) + (world_size, )) % 2000
    procs = mp.spawn(_worker, args=(world_size, port, cfg, queue), nprocs=world_size, join=False)
    batches = dict(queue.get(timeout=300) for _ in range(world_size))
    procs.join()
    return [batches[rank] for rank in range(world_size)]


def _flat(batches):
    return [idx for batch in batches for idx in batch]


def test_even_shards_cover_the_dataset():
    seen = [_flat(b) for b in _run(4, num_samples=40)]
    assert [len(s) for s in seen] == [10, 10, 10, 10]
    assert sorted(idx for s in seen for idx in s) == list(range(40))


def test_uneven_shards_stop_together():
    # 41 samples over 4 ranks: rank0 gets 11, the others 10. Every rank must yield the same
    # number of batches (and none may hang), so rank0's trailing sample is dropped.
    seen = [_flat(b) for b in _run(4, num_samples=41)]
    assert [len(s) for s in seen] == [10, 10, 10, 10]
    assert 40 not in seen[0]


def test_skip_batches_is_local_to_each_rank():
    seen = [_flat(b) for b in _run(2, num_samples=20, skip_batches=3)]
    assert seen == [[6, 8, 10, 12, 14, 16, 18], [7, 9, 11, 13, 15, 17, 19]]


def test_single_process_is_a_noop():
    assert [_flat(b) for b in _run(1, num_samples=5)] == [[0, 1, 2, 3, 4]]


def test_shard_survives_a_dataloader_worker():
    """The split is bound before the loader starts its worker, so the worker must see it."""
    seen = [_flat(b) for b in _run(2, num_samples=20, num_workers=1)]
    assert seen == [list(range(0, 20, 2)), list(range(1, 20, 2))]


def test_shard_reads_the_same_data_as_dispatch():
    """Sharding must not change which samples a rank trains on, only who prepares them."""
    for world_size, num_samples, batch_size in [(4, 40, 1), (4, 40, 2), (2, 40, 3), (2, 37, 4)]:
        dispatched = _run(world_size, num_samples=num_samples, batch_size=batch_size, shard=False)
        sharded = _run(world_size, num_samples=num_samples, batch_size=batch_size, shard=True)
        assert sharded == dispatched, (world_size, num_samples, batch_size, dispatched, sharded)


def test_packing_shard_partitions_the_samples():
    """With packing each rank rebuilds its own packs, so the grouping changes but the data must not.

    Both modes drop whatever is left when the stream runs out mid-step, so coverage is compared
    with that tail allowed for.
    """
    num_samples = 512
    for shard in (True, False):
        seen = _flat(_flat(_run(2, num_samples=num_samples, packing=True, shard=shard)))
        assert len(seen) == len(set(seen)), f'a sample was packed twice (shard={shard})'
        assert set(seen) <= set(range(num_samples))
        assert len(seen) >= num_samples - PACKING_INTERVAL, f'lost more than a tail round (shard={shard})'


if __name__ == '__main__':
    test_shard_state_is_bound_lazily()
    test_even_shards_cover_the_dataset()
    test_uneven_shards_stop_together()
    test_skip_batches_is_local_to_each_rank()
    test_single_process_is_a_noop()
    test_shard_survives_a_dataloader_worker()
    test_shard_reads_the_same_data_as_dispatch()
    test_packing_shard_partitions_the_samples()
    print('all passed')
