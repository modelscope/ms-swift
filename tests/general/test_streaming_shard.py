import os
import socket
import torch.distributed as dist
import torch.multiprocessing as mp
from datasets import Dataset
from torch.utils.data import DataLoader


def _stream(size):
    return Dataset.from_list([{'idx': i} for i in range(size)]).to_iterable_dataset()


def _collate(batch):
    return [row['idx'] for row in batch]


class _PackingTemplate:
    packing = False
    padding_free = False
    max_length = 128

    def encode(self, row, return_length=False):
        # Deliberately imbalanced: first raw block produces many packs, the second one only one.
        length = 128 if row['idx'] < 128 else 1
        return {'idx': row['idx'], 'input_ids': [0] * length, 'labels': [0] * length}


def test_lazy_block_assignment():
    from swift.dataset import shard_streaming_dataset
    dataset, state = shard_streaming_dataset(_stream(10))
    assert [row['idx'] for row in dataset] == list(range(10))
    state.bind(rank=1, world_size=3, block_size=1)
    assert [row['idx'] for row in dataset] == [1, 4, 7]
    state.bind(rank=0, world_size=2, block_size=3)
    assert [row['idx'] for row in dataset] == [0, 1, 2, 6, 7, 8]


def _distributed_cases(rank, port, queue):
    from swift.dataloader import DataLoaderDispatcher
    from swift.dataset import shard_streaming_dataset
    from swift.dataset.packing import IterablePackingDataset

    os.environ.update({'MASTER_ADDR': '127.0.0.1', 'MASTER_PORT': str(port)})
    dist.init_process_group('gloo', rank=rank, world_size=2)
    try:

        def collect(size, batch_size, shard, *, skip=0, workers=0):
            dataset = _stream(size)
            state = None
            if shard:
                dataset, state = shard_streaming_dataset(dataset)
            loader = DataLoader(dataset, batch_size=batch_size, collate_fn=_collate, num_workers=workers)
            return list(DataLoaderDispatcher(loader, skip_batches=skip, shard_state=state))

        # A deterministic one-to-one stream keeps the dispatch assignment, including its tail.
        dispatched = collect(17, 2, False, workers=1)
        sharded = collect(17, 2, True, workers=1)

        # Resume skips local batches instead of replaying every rank's data on rank 0.
        skipped = collect(20, 1, True, skip=3)

        # Packing is intentionally imbalanced. first-exhausted keeps the ranks in step but can
        # leave substantial data unconsumed on another rank; this is documented behaviour.
        dataset, state = shard_streaming_dataset(_stream(256))
        dataset = IterablePackingDataset(
            _PackingTemplate(),
            dataset,
            num_proc=1,
            packing_interval=128,
            packing_length=128,
            packing_strategy='sequential')
        loader = DataLoader(dataset, batch_size=1, collate_fn=lambda batch: [row['idx'] for row in batch[0]])
        packed = list(DataLoaderDispatcher(loader, shard_state=state))
        queue.put((rank, {'dispatched': dispatched, 'sharded': sharded, 'skipped': skipped, 'packed': packed}))
    finally:
        dist.destroy_process_group()


def test_distributed_streaming_shard():
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
    ctx = mp.get_context('spawn')
    queue = ctx.Queue()
    procs = mp.spawn(_distributed_cases, args=(port, queue), nprocs=2, join=False)
    result = dict(queue.get(timeout=300) for _ in range(2))
    procs.join()

    assert [result[r]['sharded'] for r in range(2)] == [result[r]['dispatched'] for r in range(2)]
    assert [[idx for batch in result[r]['skipped'] for idx in batch] for r in range(2)] == [
        [6, 8, 10, 12, 14, 16, 18],
        [7, 9, 11, 13, 15, 17, 19],
    ]
    assert [len(result[r]['packed']) for r in range(2)] == [1, 1]
    assert [sum(len(batch) for batch in result[r]['packed']) for r in range(2)] == [1, 128]
