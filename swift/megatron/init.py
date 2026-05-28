# Copyright (c) ModelScope Contributors. All rights reserved.
import concurrent.futures
import importlib.metadata
import logging
import os
import torch
import torch.distributed as dist
from contextlib import contextmanager
from copy import copy
from packaging import version
from tqdm import tqdm
from transformers.modeling_utils import custom_object_save
from transformers.utils import is_torch_npu_available
from transformers.utils.versions import require_version

from swift.model import get_model_processor, save_checkpoint
from swift.utils import (HfConfigFactory, disable_safe_ddp_context_use_barrier, get_logger, get_modules_to_not_convert,
                         get_multimodal_target_regex, is_master, split_list)

logger = get_logger()

_SWIFT_BRIDGE_GLOO_GROUP_CACHE = {}
_SWIFT_BRIDGE_GLOO_GROUP_BY_SOURCE_GROUP = {}


def _patch__batched_p2p_ops():
    from megatron.core.pipeline_parallel import p2p_communication

    _batched_p2p_ops_origin = p2p_communication._batched_p2p_ops

    def _batched_p2p_ops(**kwargs):
        kwargs['group'] = None
        return _batched_p2p_ops_origin(**kwargs)

    p2p_communication._batched_p2p_ops = _batched_p2p_ops


def _clear_npu_default_pg_bound_device_id_for_gloo():
    if not is_torch_npu_available() or not dist.is_initialized():
        return
    try:
        default_pg = dist.distributed_c10d._get_default_group()
    except Exception:
        return
    if getattr(default_pg, 'bound_device_id', None) is not None:
        default_pg.bound_device_id = None


def _get_bridge_group_specs(bridge, is_expert):
    from megatron.core import mpu

    world_size = dist.get_world_size()
    if is_expert:
        group_unit = bridge.etp_size * bridge.ep_size * bridge.pp_size
        if world_size % group_unit != 0:
            raise RuntimeError(
                f'Cannot build mcore_bridge EP-PP Gloo groups: world_size={world_size}, '
                f'etp={bridge.etp_size}, ep={bridge.ep_size}, pp={bridge.pp_size}.')
        rank_generator = mpu.RankGenerator(
            tp=bridge.etp_size,
            ep=bridge.ep_size,
            dp=world_size // group_unit,
            pp=bridge.pp_size,
            cp=1,
            order='tp-cp-ep-dp-pp',
            rank_offset=0,
        )
        token = 'ep-pp'
    else:
        cp_size = int(getattr(bridge.config, 'context_parallel_size', 1) or 1)
        group_unit = bridge.tp_size * bridge.pp_size * cp_size
        if world_size % group_unit != 0:
            raise RuntimeError(
                f'Cannot build mcore_bridge PP Gloo groups: world_size={world_size}, '
                f'tp={bridge.tp_size}, pp={bridge.pp_size}, cp={cp_size}.')
        rank_generator = mpu.RankGenerator(
            tp=bridge.tp_size,
            ep=1,
            dp=world_size // group_unit,
            pp=bridge.pp_size,
            cp=cp_size,
            order='tp-cp-ep-dp-pp',
            rank_offset=0,
        )
        token = 'pp'
    return [tuple(int(rank) for rank in ranks) for ranks in rank_generator.get_ranks(token)]


def _get_or_create_bridge_gloo_group(bridge, is_expert):
    pp_group = bridge.ep_pp_group if is_expert else bridge.pp_group
    try:
        own_ranks = tuple(int(rank) for rank in dist.get_process_group_ranks(pp_group))
    except Exception as e:
        raise RuntimeError(f'Cannot inspect mcore_bridge PP/EP-PP group ranks: is_expert={is_expert}.') from e

    group_specs = _get_bridge_group_specs(bridge, is_expert)
    if own_ranks not in group_specs:
        raise RuntimeError(
            f'mcore_bridge PP/EP-PP group ranks are inconsistent with reconstructed specs: '
            f'is_expert={is_expert}, own_ranks={own_ranks}, specs={group_specs}.')

    cache_key = ('ep_pp' if is_expert else 'pp', tuple(group_specs))
    if cache_key not in _SWIFT_BRIDGE_GLOO_GROUP_CACHE:
        _clear_npu_default_pg_bound_device_id_for_gloo()
        device = torch.device('npu', torch.npu.current_device())
        marker = torch.ones((), dtype=torch.int32, device=device)
        dist.all_reduce(marker)
        own_groups = {}
        for ranks in group_specs:
            group = dist.new_group(list(ranks), backend='gloo')
            if dist.get_rank() in ranks:
                own_groups[ranks] = group
        marker = torch.ones((), dtype=torch.int32, device=device)
        dist.all_reduce(marker)
        _SWIFT_BRIDGE_GLOO_GROUP_CACHE[cache_key] = own_groups
        logger.warning_once(
            f'Created mcore_bridge {cache_key[0]} Gloo metadata groups for NPU weight export: ranks={group_specs}')

    gloo_group = _SWIFT_BRIDGE_GLOO_GROUP_CACHE[cache_key].get(own_ranks)
    if gloo_group is None:
        raise RuntimeError(
            f'Current rank is not in cached mcore_bridge Gloo group: is_expert={is_expert}, own_ranks={own_ranks}.')
    _SWIFT_BRIDGE_GLOO_GROUP_BY_SOURCE_GROUP[id(pp_group)] = (pp_group, gloo_group)
    return gloo_group


def _get_cached_bridge_gloo_group(group):
    source_group, gloo_group = _SWIFT_BRIDGE_GLOO_GROUP_BY_SOURCE_GROUP.get(id(group), (None, None))
    return gloo_group if source_group is group else None


def _patch_torch_FileSystemReader():
    from torch.distributed.checkpoint.filesystem import FileSystemReader
    from torch.futures import Future
    _origin_read_data = FileSystemReader.read_data
    _origin__slice_file = FileSystemReader._slice_file
    READER_MAX_WORKERS = int(os.environ.get('MCORE_READER_MAX_WORKERS', '16'))

    @contextmanager
    def _patch__slice_file(prog_bar):

        def _slice_file(self, *args, **kwargs):
            prog_bar.update()
            return _origin__slice_file(self, *args, **kwargs)

        FileSystemReader._slice_file = _slice_file
        try:
            yield
        finally:
            FileSystemReader._slice_file = _origin__slice_file

    def read_data(self, plan, planner):

        def _worker(plan_shard):
            _origin_read_data(self, plan_shard, planner)

        prog_bar = tqdm(total=len(plan.items), dynamic_ncols=True, desc='Loading: ')
        plan_shards = split_list(plan.items, READER_MAX_WORKERS, contiguous=False)
        with _patch__slice_file(prog_bar):
            with concurrent.futures.ThreadPoolExecutor(max_workers=READER_MAX_WORKERS) as pool:
                futures = []
                for i in range(READER_MAX_WORKERS):
                    plan_shard = copy(plan)
                    plan_shard.items = plan_shards[i]
                    futures.append(pool.submit(_worker, plan_shard))
                concurrent.futures.wait(futures)
        prog_bar.close()
        fut: Future = Future()
        fut.set_result(None)
        return fut

    FileSystemReader.read_data = read_data


def _patch_validate_non_overlapping_shards_metadata():
    # too slow
    from torch.distributed._shard.sharded_tensor import api
    from torch.distributed._shard.sharding_spec import api as api2
    from torch.distributed.checkpoint import default_planner

    def validate_non_overlapping_shards_metadata(*args, **kwargs):
        pass

    api.validate_non_overlapping_shards_metadata = validate_non_overlapping_shards_metadata
    api2.validate_non_overlapping_shards_metadata = validate_non_overlapping_shards_metadata

    def _validate_global_plan(*args, **kwargs):
        return True

    default_planner._validate_global_plan = _validate_global_plan


def _patch_unified_memory():
    if is_torch_npu_available():
        return

    mcore_015 = version.parse(importlib.metadata.version('megatron-core')) >= version.parse('0.15.0rc0')
    if not mcore_015:
        return
    from torch.utils import cpp_extension
    load_inline = cpp_extension.load_inline

    def _new_load_inline(*args, **kwargs):
        name = kwargs.get('name')
        if name == 'managed_alloc_runtime':
            raise RuntimeError
        return load_inline(*args, **kwargs)

    # not create unified memory mempool
    cpp_extension.load_inline = _new_load_inline
    try:
        from megatron.core.inference import unified_memory
    except Exception:
        pass
    finally:
        cpp_extension.load_inline = load_inline


def _patch_mcore_bridge():
    require_version('mcore-bridge>=1.2.0', 'please install mcore-bridge via `pip install mcore-bridge -U`')
    import mcore_bridge
    from mcore_bridge import GPTBridge
    logger.info(f'mcore_bridge.__version__: {mcore_bridge.__version__}')
    origin_save_weights = GPTBridge.save_weights
    origin_convert = GPTBridge._convert
    origin_get_weight = GPTBridge._get_weight
    origin_broadcast_ep_pp = GPTBridge._broadcast_ep_pp

    bridge_dist = origin_convert.__globals__.get('dist', dist)
    import torch.distributed.distributed_c10d as distributed_c10d
    origin_all_reduce = getattr(distributed_c10d.all_reduce, '_swift_origin', distributed_c10d.all_reduce)
    if not getattr(distributed_c10d.all_reduce, '_swift_npu_bridge_control_patched', False):

        def all_reduce(input_, op=dist.ReduceOp.SUM, group=None, async_op=False):
            is_scalar_control = (
                is_torch_npu_available() and isinstance(input_, torch.Tensor) and input_.numel() == 1
                and input_.dtype in {torch.bool, torch.int32, torch.int64} and group is not None and not async_op)
            if is_scalar_control:
                gloo_group = _get_cached_bridge_gloo_group(group)
                if gloo_group is not None:
                    flag = torch.tensor([int(input_.detach().cpu().item())], dtype=torch.int32)
                    origin_all_reduce(flag, op=dist.ReduceOp.SUM, group=gloo_group)
                    value = int(flag.item())
                    if input_.dtype == torch.bool:
                        input_.copy_(torch.ones_like(input_) if value else torch.zeros_like(input_))
                    else:
                        input_.fill_(value)
                    torch.npu.synchronize()
                    if input_.dtype == torch.bool and int(input_.detach().cpu().item()) != int(bool(value)):
                        input_.data = torch.tensor([bool(value)], dtype=input_.dtype, device=input_.device)
                        torch.npu.synchronize()
                    return None
            return origin_all_reduce(input_, op=op, group=group, async_op=async_op)

        all_reduce._swift_npu_bridge_control_patched = True
        all_reduce._swift_origin = origin_all_reduce
        bridge_dist.all_reduce = all_reduce
        dist.all_reduce = all_reduce
        distributed_c10d.all_reduce = all_reduce

    def _convert(self, mg_models, hf_state_dict, hf_prefix: str, to_mcore: bool, tqdm_desc: str = 'Converting: '):
        yield from origin_convert(self, mg_models, hf_state_dict, hf_prefix, to_mcore, tqdm_desc=tqdm_desc)

    if not getattr(origin_convert, '_swift_npu_convert_patched', False):
        _convert._swift_npu_convert_patched = True
        _convert._swift_origin = origin_convert
        GPTBridge._convert = _convert

    def _get_weight(self, mg_weight, mg_key, offset=0, is_expert=False):
        if not is_torch_npu_available():
            return origin_get_weight(self, mg_weight, mg_key, offset, is_expert)

        pp_size = self.ep_pp_size if is_expert else self.pp_size
        is_scalar_placeholder = isinstance(mg_weight, torch.Tensor) and mg_weight.ndim == 0
        if mg_key is not None and pp_size > 1 and is_scalar_placeholder:
            mg_weight = None
        return origin_get_weight(self, mg_weight, mg_key, offset, is_expert)

    if not getattr(origin_get_weight, '_swift_npu_get_weight_patched', False):
        _get_weight._swift_npu_get_weight_patched = True
        _get_weight._swift_origin = origin_get_weight
        GPTBridge._get_weight = _get_weight

    def _broadcast_ep_pp(self, tensor, is_expert):
        if not is_torch_npu_available():
            return origin_broadcast_ep_pp(self, tensor, is_expert)

        pp_group = self.ep_pp_group if is_expert else self.pp_group
        pp_size = self.ep_pp_size if is_expert else self.pp_size
        pp_rank = self.ep_pp_rank if is_expert else self.pp_rank
        if pp_size <= 1:
            return tensor

        device = torch.device('npu', torch.npu.current_device())
        gloo_group = _get_or_create_bridge_gloo_group(self, is_expert)
        bridge_dist.all_reduce = all_reduce
        dist.all_reduce = all_reduce
        distributed_c10d.all_reduce = all_reduce
        # Some non-owner pipeline ranks carry a 0-d placeholder tensor instead
        # of ``None``.  That placeholder is not a valid weight source; treating
        # it as one makes embedding weights become scalar tensors during export.
        has_tensor = tensor is not None and not (isinstance(tensor, torch.Tensor) and tensor.ndim == 0)
        try:
            group_ranks = dist.get_process_group_ranks(pp_group)
        except Exception:
            group_ranks = None

        dtype_mapping = [torch.float64, torch.float32, torch.float16, torch.bfloat16, torch.uint8, torch.int32]
        dtype_mapping_r = {v: k for k, v in enumerate(dtype_mapping)}
        comm_tensor = None if not has_tensor else tensor.to(device, non_blocking=True)

        # Keep metadata on a Gloo control-plane group and use the original HCCL
        # PP/EP-PP group only for the weight payload.  On this colocated NPU
        # stack, tiny HCCL metadata tensors can be observed as stale values even
        # after synchronization, while the large payload broadcast itself is the
        # operation we want to preserve.
        for src_group_rank in range(pp_size):
            src_rank = dist.get_global_rank(pp_group, src_group_rank)
            meta_data = torch.full((10, ), -1, dtype=torch.int64)
            if pp_rank == src_group_rank and has_tensor:
                dtype_idx = dtype_mapping_r.get(comm_tensor.dtype)
                if dtype_idx is None:
                    raise RuntimeError(
                        f'Unsupported dtype in mcore_bridge PP/EP broadcast: '
                        f'dtype={comm_tensor.dtype}, is_expert={is_expert}.')
                if comm_tensor.ndim + 2 > meta_data.numel():
                    raise RuntimeError(
                        f'Tensor shape has too many dims for mcore_bridge PP/EP broadcast metadata: '
                        f'shape={list(comm_tensor.shape)}, is_expert={is_expert}.')
                meta_data[0] = comm_tensor.ndim
                meta_data[1:1 + comm_tensor.ndim] = torch.tensor(list(comm_tensor.shape), dtype=torch.int64)
                meta_data[-1] = dtype_idx

            dist.broadcast(meta_data, src=src_rank, group=gloo_group)
            ndim = int(meta_data[0].item())
            if ndim < 0:
                continue
            dtype_idx = int(meta_data[-1].item())
            if dtype_idx < 0 or dtype_idx >= len(dtype_mapping):
                raise RuntimeError(
                    f'Invalid dtype metadata in mcore_bridge PP/EP broadcast: rank={dist.get_rank()}, '
                    f'src_rank={src_rank}, is_expert={is_expert}, meta_data={meta_data.tolist()}.')
            shape = [int(dim) for dim in meta_data[1:1 + ndim].tolist()]
            dtype = dtype_mapping[dtype_idx]
            if comm_tensor is None or list(comm_tensor.shape) != shape or comm_tensor.dtype != dtype:
                comm_tensor = torch.empty(shape, device=device, dtype=dtype)
            dist.broadcast(comm_tensor, src=src_rank, group=pp_group)
            return comm_tensor

        raise RuntimeError(
            f'No source tensor found in mcore_bridge PP/EP broadcast: '
            f'is_expert={is_expert}, group_size={pp_size}, group_ranks={group_ranks}.')

    if not getattr(origin_broadcast_ep_pp, '_swift_npu_broadcast_ep_pp_patched', False):
        _broadcast_ep_pp._swift_npu_broadcast_ep_pp_patched = True
        _broadcast_ep_pp._swift_origin = origin_broadcast_ep_pp
        GPTBridge._broadcast_ep_pp = _broadcast_ep_pp

    def save_weights(
        self,
        mg_models,
        output_dir: str,
        peft_format: bool = False,
        max_shard_size: str = '5GB',
        args=None,
        processor=None,
    ) -> None:
        origin_save_weights(self, mg_models, output_dir, peft_format=peft_format, max_shard_size=max_shard_size)
        if processor is None or args is None:
            return
        hf_config = self.config.hf_config
        hf_config = copy(hf_config)
        if is_master() and not hasattr(self, 'hf_model'):
            if hasattr(self, 'get_hf_meta_model'):
                self.hf_model = self.get_hf_meta_model()
                self.hf_model.model_meta = processor.model_meta
                self.hf_model.model_info = processor.model_info
            else:
                with torch.device('meta'), disable_safe_ddp_context_use_barrier():
                    self.hf_model = get_model_processor(
                        args.model_dir, model_type=args.model_type, return_dummy_model=True)[0]

        if is_master():
            if peft_format:
                peft_config = copy(mg_models[0].peft_config[self._adapter_name])
                if self.config.task_type == 'seq_cls':
                    peft_config.task_type = 'SEQ_CLS'
                if self.is_multimodal and 'all-linear' in args.target_modules:
                    peft_config.target_modules = get_multimodal_target_regex(
                        self.hf_model,
                        freeze_llm=args.freeze_llm,
                        freeze_vit=args.freeze_vit,
                        freeze_aligner=args.freeze_aligner,
                        include_embedding='all-embedding' in args.target_modules,
                        exclude_router='all-router' not in args.target_modules)
                else:
                    assert not isinstance(peft_config.target_modules, str), (
                        'target_regex is not currently supported for LoRA conversion. Please set `--merge_lora true`.')
                    peft_config.target_modules = self._peft_target_modules
                peft_config.modules_to_save = self._peft_modules_to_save
                peft_config.save_pretrained(output_dir)
            else:
                config = self.config
                llm_config = HfConfigFactory.get_text_config(hf_config)
                if config.mtp_num_layers:
                    for key in ['num_nextn_predict_layers', 'mtp_num_hidden_layers']:
                        if hasattr(llm_config, key):
                            setattr(llm_config, key, config.mtp_num_layers)
                            break
                    else:
                        llm_config.num_nextn_predict_layers = config.mtp_num_layers
                if config.fp8 is not None and config.fp8_recipe == 'blockwise' and config.fp8_param:
                    if getattr(hf_config, 'quantization_config', None) is None:
                        from transformers.utils.quantization_config import FineGrainedFP8Config
                        modules_to_not_convert = get_modules_to_not_convert(self.hf_model)
                        if hasattr(self, '_fp8_skip_modules'):
                            modules_to_not_convert = (modules_to_not_convert or []) + list(self._fp8_skip_modules)
                        hf_config.quantization_config = FineGrainedFP8Config(
                            modules_to_not_convert=modules_to_not_convert)
                elif hasattr(hf_config, 'quantization_config'):
                    del hf_config.quantization_config
                hf_config.save_pretrained(output_dir)
                if getattr(self.hf_model, '_auto_class') is not None:
                    try:
                        custom_object_save(self.hf_model, output_dir, config=hf_config)
                    except FileNotFoundError as e:
                        logger.error(f'custom_object_save Error: {e}')
                save_checkpoint(
                    None,
                    processor,
                    output_dir,
                    model_dirs=[args.model_dir],
                    additional_saved_files=self.hf_model.model_meta.additional_saved_files)
            logger.info(f'Successfully saved `safetensors` model weights in `{output_dir}`.')
        dist.barrier()  # Ensure all weights are saved completely

    GPTBridge.save_weights = save_weights


def init_megatron_env():
    os.environ.pop('VLLM_USE_MODELSCOPE', None)
    logging_level = logging.root.level
    _patch_unified_memory()
    _patch_mcore_bridge()
    _patch__batched_p2p_ops()
    logging.root.setLevel(logging_level)  # revert logger level
    try:
        _patch_torch_FileSystemReader()
    except Exception:
        logger.warning('Failed to patch FileSystemReader.')
    try:
        _patch_validate_non_overlapping_shards_metadata()
    except Exception:
        logger.warning('Patch validate_non_overlapping_shards_metadata failed.')
        pass
    import megatron.core
    logger.info(f'megatron.core.__version__: {megatron.core.__version__}')
