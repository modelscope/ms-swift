# Copyright (c) ModelScope Contributors. All rights reserved.
import torch
import torch.distributed as dist
from tqdm import tqdm

from swift.utils import to_device


class _TensorMeta:
    """Sentinel replacing a tensor in the schema, carrying metadata for buffer allocation."""
    __slots__ = ('idx', 'shape', 'dtype')

    def __init__(self, idx, shape, dtype):
        self.idx = idx
        self.shape = shape
        self.dtype = dtype


def _flatten_for_scatter(obj, tensors):
    """Recursively separate tensors from a nested structure.

    Tensors are appended to `tensors` and replaced by _TensorMeta sentinels.
    The returned schema is lightweight and can be pickled efficiently.
    """
    if torch.is_tensor(obj):
        idx = len(tensors)
        tensors.append(obj)
        return _TensorMeta(idx, tuple(obj.shape), obj.dtype)
    elif isinstance(obj, dict):
        return {k: _flatten_for_scatter(v, tensors) for k, v in obj.items()}
    elif isinstance(obj, (tuple, list)):
        return type(obj)(_flatten_for_scatter(v, tensors) for v in obj)
    else:
        return obj


def _unflatten_from_scatter(schema, tensors):
    """Reconstruct the original nested structure from schema and flat tensors list."""
    if isinstance(schema, _TensorMeta):
        return tensors[schema.idx]
    elif isinstance(schema, dict):
        return {k: _unflatten_from_scatter(v, tensors) for k, v in schema.items()}
    elif isinstance(schema, (tuple, list)):
        return type(schema)(_unflatten_from_scatter(v, tensors) for v in schema)
    else:
        return schema


def _collect_tensor_metas(schema, metas):
    """Collect _TensorMeta from schema in DFS order (same order as flatten)."""
    if isinstance(schema, _TensorMeta):
        metas.append(schema)
    elif isinstance(schema, dict):
        for v in schema.values():
            _collect_tensor_metas(v, metas)
    elif isinstance(schema, (tuple, list)):
        for v in schema:
            _collect_tensor_metas(v, metas)


class DataLoaderDispatcher:

    def __init__(self, base_dataloader, device=None, skip_batches: int = 0):
        self.base_dataloader = base_dataloader
        self.device = device
        self.skip_batches = skip_batches

    @property
    def rank(self):
        return dist.get_rank(self.group) if dist.is_initialized() else 0

    @property
    def world_size(self):
        return dist.get_world_size(self.group) if dist.is_initialized() else 1

    @property
    def group(self):
        return dist.group.WORLD if dist.is_initialized() else 1

    @property
    def _scatter_device(self):
        """Determine the correct device for dist.scatter based on backend."""
        backend = dist.get_backend(self.group)
        if backend == 'nccl':
            return torch.device('cuda', torch.cuda.current_device())
        elif backend == 'hccl':
            return torch.device('npu', torch.npu.current_device())
        return None  # keep tensors on their original device

    def _scatter_object_list(self, inputs):
        """Scatter data from rank 0 to all ranks.

        Optimization: separates tensors from non-tensor structure (schema) so that
        schemas are scattered via pickle (lightweight) and tensors are transferred
        via P2P isend/irecv (efficient NCCL/Gloo tensor transfer, zero padding waste).
        Naturally handles variable-size tensors across ranks.
        """
        if not dist.is_initialized():
            return inputs[0]

        global_src_rank = dist.get_global_rank(self.group, 0)
        scatter_device = self._scatter_device

        if self.rank == 0:
            # Flatten each rank's data: separate tensors from schema
            schemas = []
            per_rank_tensors = []
            for item in inputs:
                if item is None:
                    schemas.append(None)
                    per_rank_tensors.append([])
                else:
                    tensors = []
                    schema = _flatten_for_scatter(item, tensors)
                    schemas.append(schema)
                    per_rank_tensors.append(tensors)

            # Scatter lightweight schemas (no tensor payload, fast pickle)
            schema_out = [None]
            dist.scatter_object_list(schema_out, schemas, global_src_rank, group=self.group)
            my_schema = schema_out[0]

            # Send tensors to other ranks via async P2P
            handles = []
            send_bufs = []  # keep tensors alive until sends complete
            for r in range(1, self.world_size):
                dst_rank = dist.get_global_rank(self.group, r)
                for t in per_rank_tensors[r]:
                    tensor = t.contiguous()
                    if scatter_device is not None:
                        tensor = tensor.to(scatter_device)
                    send_bufs.append(tensor)
                    handles.append(dist.isend(tensor, dst=dst_rank, group=self.group))

            # Rank 0 keeps its own tensors (move to device if needed)
            my_tensors = per_rank_tensors[0]
            if scatter_device is not None:
                my_tensors = [t.contiguous().to(scatter_device) for t in my_tensors]

            # Wait for all sends to complete
            for h in handles:
                h.wait()
            del send_bufs  # safe to release after all sends finished
        else:
            # Receive schema (lightweight)
            schema_out = [None]
            dist.scatter_object_list(schema_out, None, global_src_rank, group=self.group)
            my_schema = schema_out[0]

            if my_schema is None:
                return None

            # Receive tensors via async P2P (shape/dtype from _TensorMeta in schema)
            metas = []
            _collect_tensor_metas(my_schema, metas)
            metas.sort(key=lambda m: m.idx)
            device = scatter_device if scatter_device is not None else 'cpu'
            my_tensors = []
            handles = []
            for meta in metas:
                recv_buf = torch.empty(meta.shape, dtype=meta.dtype, device=device)
                handles.append(dist.irecv(recv_buf, src=global_src_rank, group=self.group))
                my_tensors.append(recv_buf)

            # Wait for all receives to complete
            for h in handles:
                h.wait()

        if my_schema is None:
            return None
        return _unflatten_from_scatter(my_schema, my_tensors)

    def _skip_batches(self, base_iter):
        if self.rank == 0 and self.skip_batches > 0:
            for _ in tqdm(range(self.skip_batches), dynamic_ncols=True, desc='Skip Batches: '):
                [next(base_iter) for _ in range(self.world_size)]

    def __iter__(self):
        base_iter = iter(self.base_dataloader)
        self._skip_batches(base_iter)
        while True:
            if self.rank == 0:
                try:
                    data = [next(base_iter) for _ in range(self.world_size)]
                except StopIteration:
                    data = [None] * self.world_size
                data = self._scatter_object_list(data)
            else:
                data = self._scatter_object_list(None)
            if data is None:
                break
            if self.device:
                data = to_device(data, self.device)
            yield data
