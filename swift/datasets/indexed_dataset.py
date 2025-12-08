# Copyright (c) Alibaba, Inc. and its affiliates.
import bisect
import mmap
import os
import pickle
import threading
from queue import Queue
from typing import Any, List

from modelscope.hub.utils.utils import get_cache_dir
from torch.utils.data import Dataset


class IndexedDatasetBuilder:
    CHUNK_SIZE = 1e10

    def __init__(self, dataset_name: str):
        self.cache_dir = IndexedDataset.get_cache_dir(dataset_name)
        self.n_shard = 1
        self.bin_path = os.path.join(self.cache_dir, IndexedDataset.BIN_FNAME.format(0))
        self.idx_path = os.path.join(self.cache_dir, IndexedDataset.IDX_FNAME)
        if os.path.exists(self.bin_path):
            os.remove(self.bin_path)
        self.bin_file = open(self.bin_path, 'ab')
        self.length_list = []
        self.idx_list = [0]
        self.shard_offset = [0]
        self._thread = None
        self._queue = Queue(maxsize=1000)

    def _write_worker(self):
        while True:
            items = self._queue.get()
            if items is None:
                break
            bin_buffer = []
            for item in items:
                item_buffer = pickle.dumps(item)
                bin_buffer.append(item_buffer)
                self.idx_list.append(self.idx_list[-1] + len(item_buffer))
                self.length_list.append(item['length'])
            self.bin_file.write(b''.join(bin_buffer))
            offset = self.idx_list[-1] - self.shard_offset[-1]
            if offset >= self.CHUNK_SIZE:
                self.bin_file.close()
                self.bin_path = os.path.join(self.cache_dir, IndexedDataset.BIN_FNAME.format(self.n_shard))
                self.shard_offset.append(self.shard_offset[-1] + offset)
                self.n_shard += 1
                if os.path.exists(self.bin_path):
                    os.remove(self.bin_path)
                self.bin_file = open(self.bin_path, 'ab')

    def add_items(self, items: List[Any]) -> None:
        if self._thread is None:
            self._thread = threading.Thread(target=self._write_worker, daemon=True)
            self._thread.start()
        self._queue.put(items)

    def finalize(self):
        if self._thread is not None:
            self._queue.put(None)
            self._thread.join()
        self.bin_file.close()
        idx_obj = {
            'idx': self.idx_list,
            'length': self.length_list,
            'n_shard': self.n_shard,
            'shard_offset': self.shard_offset,
        }
        with open(self.idx_path, 'wb') as f:
            pickle.dump(idx_obj, f)


class BinReader:

    def __init__(self, bin_path: str):
        self.bin_path = bin_path
        self.file = open(bin_path, 'rb')
        try:
            self.mm = mmap.mmap(self.file.fileno(), 0, access=mmap.ACCESS_READ)
        except ValueError:
            # For example, self.file is an empty file.
            self.mm = None

    def read_buffer(self, offset: int, size: int) -> bytes:
        if offset < 0 or size < 0 or offset + size > len(self.mm):
            raise ValueError('Invalid offset or size')
        return self.mm[offset:offset + size]

    def __del__(self):
        if self.mm is not None:
            self.mm.close()
        self.file.close()


class IndexedDataset(Dataset):
    BIN_FNAME = 'data-{:05d}.bin'
    IDX_FNAME = 'data.idx'

    @staticmethod
    def get_cache_dir(dataset_name: str):
        cache_dir = os.getenv('PACKING_CACHE') or os.path.join(get_cache_dir(), 'tmp')
        cache_dir = os.path.join(cache_dir, dataset_name)
        os.makedirs(cache_dir, exist_ok=True)
        assert dataset_name is not None, f'dataset_name: {dataset_name}'
        return cache_dir

    def __init__(self, dataset_name: str):
        self.dataset_name = dataset_name
        cache_dir = self.get_cache_dir(dataset_name)
        self.idx_path = os.path.join(cache_dir, self.IDX_FNAME)
        with open(self.idx_path, 'rb') as f:
            idx_obj = pickle.load(f)
        self.idx_list = idx_obj['idx']
        self.length_list = idx_obj['length']
        self.n_shard = idx_obj['n_shard']
        self.shard_offset = idx_obj['shard_offset']
        self.bin_readers = []
        for i in range(self.n_shard):
            bin_path = os.path.join(cache_dir, self.BIN_FNAME.format(i))
            self.bin_readers.append(BinReader(bin_path))

    def __getitem__(self, index: int):
        if index < 0:
            index = index % len(self)
        idx, idx_next = self.idx_list[index], self.idx_list[index + 1]
        num_shard = bisect.bisect_right(self.shard_offset, idx)
        offset = self.shard_offset[num_shard - 1]
        buffer = self.bin_readers[num_shard - 1].read_buffer(idx - offset, idx_next - idx)
        return pickle.loads(buffer)

    def __len__(self):
        return len(self.idx_list) - 1
