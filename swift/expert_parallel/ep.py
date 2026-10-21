# Copyright (c) ModelScope Contributors. All rights reserved.
"""Expert Parallel — universal, model-agnostic expert sharding for MoE models.

All modern HuggingFace MoE models share the same expert structure:
  - experts.gate_up_proj  nn.Parameter [num_experts, 2*intermediate, hidden]
  - experts.down_proj     nn.Parameter [num_experts, hidden, intermediate]
  - gate (router)         returns (logits, weights, indices)
  - shared_experts        optional dense MLP

This module auto-discovers MoE modules by scanning for the universal
gate_up_proj / down_proj pattern, so no per-model registration is needed.
"""
import gc
import torch
from typing import Optional
from torch import nn
from torch.distributed.tensor import DTensor, Shard
from torch.distributed import init_device_mesh, DeviceMesh
import torch.distributed as dist
from torch.distributed.nn.functional import all_to_all_single as differentiable_all_to_all_single
import torch.nn.functional as F

from swift import get_logger
from swift.model import get_llm_model
from swift.utils import HfConfigFactory, get_dist_setting, get_device, is_master

logger = get_logger()


class ExpertParallel:
    """Universal Expert Parallel for FSDP2-based MoE training.

    Expert weights are sharded across the EP mesh dimension as DTensors
    with Shard(0) placement. Non-expert weights are broadcast-replicated.
    The MoE forward is replaced with an A2A dispatch / local-compute /
    A2A combine pipeline.  Shared experts (if present) are overlapped
    with the A2A dispatch on a CUDA side-stream.
    """

    def __init__(self):
        self._prepared: bool = False
        self.ep_size: Optional[int] = None
        self.ep_world_size: Optional[int] = None
        self.dp_world_size: Optional[int] = None
        self.model_dtype: Optional[torch.dtype] = None
        self.device_mesh: Optional[DeviceMesh] = None

        self.n_routed_experts: Optional[int] = None
        self.n_shared_experts: Optional[int] = None
        self.num_experts_per_tok: Optional[int] = None
        self.num_experts_per_rank: Optional[int] = None

        self.ignored_modules: list = []
        self.moe_modules: list = []

        self.has_shared_experts: bool = False
        self.side_stream: Optional[torch.cuda.Stream] = None
        self.overlap_shared_expert: bool = True

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def ep_group(self):
        if self.device_mesh is None:
            return None
        return self.device_mesh['ep'].get_group()

    @property
    def ep_rank(self):
        if self.device_mesh is None or not dist.is_initialized():
            return 0
        return dist.get_rank(self.device_mesh['ep'].get_group())

    @property
    def dp_group(self):
        if self.device_mesh is None or self.device_mesh.ndim < 2:
            return None
        return self.device_mesh['dp'].get_group()

    # ------------------------------------------------------------------
    # Config normalization — different models name these differently
    # ------------------------------------------------------------------

    @staticmethod
    def _get_n_routed_experts(config):
        for attr in ['n_routed_experts', 'num_experts', 'num_local_experts']:
            val = HfConfigFactory.get_config_attr(config, attr)
            if val is not None and val > 0:
                return val
        return None

    @staticmethod
    def _get_num_experts_per_tok(config):
        for attr in ['num_experts_per_tok', 'num_local_experts_per_tok']:
            val = HfConfigFactory.get_config_attr(config, attr)
            if val is not None and val > 0:
                return val
        return None

    @staticmethod
    def _get_n_shared_experts(config):
        for attr in ['n_shared_experts', 'num_shared_experts']:
            val = HfConfigFactory.get_config_attr(config, attr)
            if val is not None and val > 0:
                return val
        return 0

    # ------------------------------------------------------------------
    # Auto-discovery — find MoE blocks by the universal pattern
    # ------------------------------------------------------------------

    def _find_moe_modules(self, model: nn.Module) -> list:
        """Find MoE block modules by scanning for the universal expert pattern.

        A MoE block is a module that has:
          - an 'experts' child with gate_up_proj (nn.Parameter, 3D) + down_proj (nn.Parameter, 3D)
          - a 'gate' child (the router)

        This works for all modern HF MoE models: Mixtral, Qwen2/3-MoE,
        DeepSeek-V3/V4, OLMoE, GLM4-MoE, Cohere2-MoE, GraniteMoE, etc.
        """
        llm_model = get_llm_model(model)
        moe_modules = []
        for name, module in llm_model.named_modules():
            experts = getattr(module, 'experts', None)
            if experts is None:
                continue
            gate_up = getattr(experts, 'gate_up_proj', None)
            down = getattr(experts, 'down_proj', None)
            if (gate_up is not None and isinstance(gate_up, nn.Parameter) and gate_up.dim() == 3
                    and down is not None and isinstance(down, nn.Parameter) and down.dim() == 3):
                # Verify it has a 'gate' (router) — distinguishes MoE blocks
                if hasattr(module, 'gate'):
                    moe_modules.append(module)
        return moe_modules

    # ------------------------------------------------------------------
    # Non-expert broadcast — replicate all params except experts
    # ------------------------------------------------------------------

    def _collect_expert_params(self, moe_modules) -> set:
        ids = set()
        for m in moe_modules:
            experts = m.experts
            for attr in ('gate_up_proj', 'down_proj'):
                param = getattr(experts, attr, None)
                if param is not None:
                    ids.add(id(param))
        return ids

    @torch.no_grad()
    def _broadcast_non_expert(self, model: nn.Module, moe_modules: list):
        if not (dist.is_available() and dist.is_initialized()):
            logger.info('Skip _broadcast_non_expert: dist not available/initialized')
            return
        rank0 = is_master()
        device = get_device()
        expert_param_ids = self._collect_expert_params(moe_modules)

        def _broadcast_and_replace(name, tensor, is_param):
            if tensor is None or id(tensor) in expert_param_ids:
                return
            if rank0:
                src_tensor = tensor.data.to(device)
            else:
                src_tensor = torch.empty(tensor.shape, dtype=tensor.dtype, device=device)
            dist.broadcast(src_tensor, src=0, group=self.ep_group)
            parts = name.rsplit('.', 1)
            parent = model.get_submodule(parts[0]) if len(parts) == 2 else model
            attr_name = parts[-1]
            if is_param:
                parent._parameters[attr_name] = nn.Parameter(src_tensor, requires_grad=tensor.requires_grad)
            else:
                parent._buffers[attr_name] = src_tensor

        for name, param in model.named_parameters():
            _broadcast_and_replace(name, param, is_param=True)
        for name, buf in model.named_buffers():
            _broadcast_and_replace(name, buf, is_param=False)

        torch.cuda.empty_cache()
        gc.collect()
        logger.info(f'Expert parallel: broadcast non-expert params done on rank={self.ep_rank} device={device}')

    # ------------------------------------------------------------------
    # Expert scatter — shard expert params across EP ranks
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _scatter_expert(self, full_param: nn.Parameter, rank0: bool, device: str) -> DTensor:
        local_shape = (self.num_experts_per_rank,) + tuple(full_param.shape[1:])
        local_slice = torch.empty(local_shape, dtype=self.model_dtype, device=device)

        if dist.is_available() and dist.is_initialized() and self.ep_world_size > 1:
            scatter_list = None
            if rank0:
                scatter_list = []
                for rank in range(self.ep_world_size):
                    start = rank * self.num_experts_per_rank
                    end = start + self.num_experts_per_rank
                    scatter_list.append(full_param.data[start:end].to(device=device, dtype=self.model_dtype))
            dist.scatter(local_slice, scatter_list=scatter_list, src=0, group=self.ep_group)
        else:
            start = self.ep_rank * self.num_experts_per_rank
            end = start + self.num_experts_per_rank
            local_slice = full_param.data[start:end].to(device=device, dtype=self.model_dtype)

        full_param.data = torch.empty(0, dtype=self.model_dtype, device=full_param.device)
        return DTensor.from_local(local_slice, device_mesh=self.device_mesh, placements=[Shard(0)], run_check=False)

    def _shard_experts(self, moe_module: nn.Module):
        experts = moe_module.experts
        rank0 = is_master()
        device = get_device()

        experts.gate_up_proj = nn.Parameter(self._scatter_expert(experts.gate_up_proj, rank0, device))
        experts.down_proj = nn.Parameter(self._scatter_expert(experts.down_proj, rank0, device))

        torch.cuda.empty_cache()
        gc.collect()

        experts.num_experts = self.num_experts_per_rank
        experts.num_local_experts = self.num_experts_per_rank
        experts.ep_group = self.ep_group
        experts.ep_world_size = self.ep_world_size
        logger.info(f'Expert parallel: scattered experts rank={self.ep_rank} device={device} '
                    f'gate_up_proj {experts.gate_up_proj.shape}, down_proj {experts.down_proj.shape}')
        self.ignored_modules.append(experts)

    # ------------------------------------------------------------------
    # Activation — universal gate+up with optional clamp
    # ------------------------------------------------------------------

    @staticmethod
    def _apply_expert_activation(experts: nn.Module, gate_up: torch.Tensor) -> torch.Tensor:
        """Apply the SwiGLU-style activation to the fused gate_up output.

        Standard pattern (Mixtral, Qwen-MoE, OLMoE, etc.):
            gate, up = gate_up.chunk(2); act_fn(gate) * up

        DeepSeek-V4 variant — clamp before act_fn:
            gate.clamp(max=limit); up.clamp(±limit); act_fn(gate) * up
        """
        gate, up = gate_up.chunk(2, dim=-1)
        # DeepSeek-V4 has a 'limit' attribute for the SwiGLU clamp
        if hasattr(experts, 'limit') and experts.limit is not None:
            gate = gate.clamp(max=experts.limit)
            up = up.clamp(min=-experts.limit, max=experts.limit)
        return experts.act_fn(gate) * up

    # ------------------------------------------------------------------
    # Local expert compute — per-expert F.linear loop
    # ------------------------------------------------------------------

    def _compute_local_experts(self, experts: nn.Module, hidden_state: torch.Tensor,
                               expert_indices: torch.Tensor) -> torch.Tensor:
        if hidden_state.shape[0] == 0:
            return hidden_state

        gate_up_local = experts.gate_up_proj.to_local()
        down_local = experts.down_proj.to_local()

        output = torch.zeros_like(hidden_state)
        for local_idx in range(experts.num_experts):
            token_idx = (expert_indices == local_idx).nonzero(as_tuple=True)[0]
            if token_idx.shape[0] == 0:
                continue
            tokens = hidden_state[token_idx]
            gate_up_out = F.linear(tokens, gate_up_local[local_idx])
            current = self._apply_expert_activation(experts, gate_up_out)
            current = F.linear(current, down_local[local_idx])
            output.index_add_(0, token_idx, current)
        return output

    # ------------------------------------------------------------------
    # Shared expert helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_shared_experts(moe_module: nn.Module, residual: torch.Tensor) -> torch.Tensor:
        """Compute shared expert output, handling gated vs ungated patterns.

        - Always-on (DeepSeek-V3/V4, GLM4-MoE, Cohere2-MoE):
            moe_module.shared_experts(residual)
        - Gated (Qwen2-MoE, Qwen3.5-MoE, Qwen3-Next):
            sigmoid(shared_expert_gate(x)) * shared_experts(x)
        """
        shared_experts = moe_module.shared_experts
        shared_expert_gate = getattr(moe_module, 'shared_expert_gate', None)
        shared_out = shared_experts(residual)
        if shared_expert_gate is not None:
            shared_out = torch.sigmoid(shared_expert_gate(residual)) * shared_out
        return shared_out

    # ------------------------------------------------------------------
    # EP MoE forward — the core A2A dispatch / compute / combine pipeline
    # ------------------------------------------------------------------

    def _ep_moe_forward(self, moe_module: nn.Module, hidden_state: torch.Tensor,
                        input_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        """EP MoE forward: router → A2A dispatch → local compute → A2A combine.

        Steps:
        1. Router computes global routing (all ranks identical on replicated non-expert weights)
        2. Expand token-expert pairs
        3. Determine target rank and local expert id
        4. Sort by target rank for contiguous packing
        5. Compute send/recv counts via A2A
        6. A2A dispatch hidden states (differentiable)
        7. Local expert computation
        8. A2A combine results back (differentiable)
        9. Unsort + apply weights + reduce across top_k
        10. Add shared expert output (overlapped on side stream)
        """
        batch_size, seq_len, hidden_dim = hidden_state.shape
        residual = hidden_state
        flat = hidden_state.view(-1, hidden_dim)
        num_tokens = flat.shape[0]

        # ------- Overlap: shared expert on side stream -------
        main_stream = torch.cuda.current_stream()
        if self.has_shared_experts and self.overlap_shared_expert and self.side_stream is not None:
            self.side_stream.wait_stream(main_stream)
            with torch.cuda.stream(self.side_stream):
                shared_out = self._compute_shared_experts(moe_module, residual)

        # ------ Main stream: A2A dispatch / compute / combine ------

        # 1. Router — detect hash router (DeepSeek-V4) vs standard
        if getattr(moe_module, 'is_hash', False) and input_ids is not None:
            router_logits, weights, indices = moe_module.gate(hidden_state, input_ids)
        else:
            router_logits, weights, indices = moe_module.gate(hidden_state)
        top_k = indices.shape[1]

        # 2. Expand token-expert pairs
        flat_indices = indices.view(-1)       # [num_tokens * top_k]
        flat_weights = weights.view(-1)       # [num_tokens * top_k]

        # 3. Determine target rank and local expert id
        target_ranks = flat_indices // self.num_experts_per_rank
        local_expert_indexes = flat_indices % self.num_experts_per_rank

        # 4. Sort by target rank (for contiguous A2A packing)
        sort_order = torch.argsort(target_ranks, stable=True)
        sorted_hidden = flat[sort_order // top_k]
        sorted_expert_indexes = local_expert_indexes[sort_order]

        # 5. Compute send/recv counts
        send_counts = torch.bincount(target_ranks, minlength=self.ep_world_size).to(torch.long)
        recv_counts = torch.zeros_like(send_counts)
        dist.all_to_all_single(output=recv_counts, input=send_counts, group=self.ep_group)

        send_splits = send_counts.tolist()
        recv_splits = recv_counts.tolist()

        # 6. A2A dispatch hidden states (differentiable)
        output_total = sum(recv_splits) if recv_splits else sorted_hidden.size(0)
        recv_hidden = differentiable_all_to_all_single(
            sorted_hidden.new_empty([output_total] + list(sorted_hidden.shape[1:])),
            sorted_hidden.contiguous(),
            recv_splits,
            send_splits,
            self.ep_group,
        )

        # Dispatch expert indices (no grad needed)
        total_recv = sum(recv_splits)
        recv_expert_ids = torch.empty(total_recv, dtype=sorted_expert_indexes.dtype, device=flat.device)
        if total_recv > 0:
            dist.all_to_all_single(
                recv_expert_ids,
                sorted_expert_indexes.contiguous(),
                output_split_sizes=recv_splits,
                input_split_sizes=send_splits,
                group=self.ep_group,
            )

        # 7. Local expert computation
        recv_output = self._compute_local_experts(moe_module.experts, recv_hidden, recv_expert_ids)

        # 8. A2A combine (send results back, differentiable)
        output_total = sum(send_splits) if send_splits else recv_output.size(0)
        combined = differentiable_all_to_all_single(
            recv_output.new_empty([output_total] + list(recv_output.shape[1:])),
            recv_output,
            send_splits,
            recv_splits,
            self.ep_group,
        )

        # 9. Unsort to restore original token-expert pair order
        unsort_order = torch.argsort(sort_order)
        output = combined[unsort_order]

        # 10. Apply routing weights and reduce over top_k
        output = output * flat_weights.unsqueeze(-1)
        output = output.view(num_tokens, top_k, hidden_dim).sum(dim=1)
        routed = output.view(batch_size, seq_len, hidden_dim)

        # Add shared expert
        if self.has_shared_experts and self.overlap_shared_expert and self.side_stream is not None:
            main_stream.wait_stream(self.side_stream)
            return routed + shared_out
        elif self.has_shared_experts:
            return routed + self._compute_shared_experts(moe_module, residual)
        return routed

    # ------------------------------------------------------------------
    # Forward patching
    # ------------------------------------------------------------------

    def _patch_forward(self, moe_module: nn.Module):
        ep = self

        def ep_moe_forward(hidden_states, input_ids=None):
            return ep._ep_moe_forward(moe_module, hidden_states, input_ids)

        moe_module.forward = ep_moe_forward

    # ------------------------------------------------------------------
    # Device mesh initialization — 1D [ep] or 2D [ep, dp]
    # ------------------------------------------------------------------

    def _init_device_mesh(self):
        rank, local_rank, world_size, local_world_size = get_dist_setting()
        assert world_size % self.ep_size == 0, \
            f'{world_size=} must be divisible by {self.ep_size=}'
        self.dp_world_size = world_size // self.ep_size
        self.ep_world_size = self.ep_size

        device = get_device().split(':')[0]

        if self.dp_world_size == 1:
            # Pure EP (like historical behavior)
            self.device_mesh = init_device_mesh(
                device,
                mesh_shape=(self.ep_world_size,),
                mesh_dim_names=('ep',),
            )
        else:
            # EP + DP
            self.device_mesh = init_device_mesh(
                device,
                mesh_shape=(self.ep_world_size, self.dp_world_size),
                mesh_dim_names=('ep', 'dp'),
            )
            logger.info(f'Expert parallel: 2D mesh [ep={self.ep_world_size}, dp={self.dp_world_size}]')

        self.side_stream = torch.cuda.Stream()
        logger.info(f'Expert parallel: device_mesh initialized: {self.device_mesh}')

    # ------------------------------------------------------------------
    # Process MoE — the main orchestration
    # ------------------------------------------------------------------

    def process_moe(self, model: nn.Module):
        moe_modules = self._find_moe_modules(model)
        if not moe_modules:
            raise RuntimeError(
                'Expert parallel: no MoE modules found in model. '
                'Expected modules with experts.gate_up_proj + experts.down_proj (nn.Parameter, 3D).'
            )
        self.moe_modules = moe_modules
        logger.info(f'Expert parallel: found {len(moe_modules)} MoE modules '
                    f'(type: {type(moe_modules[0]).__name__})')

        # Detect shared expert presence
        self.has_shared_experts = all(
            hasattr(m, 'shared_experts') and m.shared_experts is not None
            for m in moe_modules
        )
        if self.has_shared_experts:
            logger.info('Expert parallel: shared_experts detected — will overlap with A2A dispatch')

        # Broadcast non-expert params (replicate across EP ranks)
        self._broadcast_non_expert(model, moe_modules)

        # Shard experts + patch forward
        for moe_module in moe_modules:
            self._shard_experts(moe_module)
            self._patch_forward(moe_module)

    # ------------------------------------------------------------------
    # FSDP2 integration — inject ignored modules
    # ------------------------------------------------------------------

    def inject_ignored_modules(self, trainer):
        """Inject EP expert modules into the FSDP2 plugin's ignored_modules list.

        Expert modules are already sharded on the EP device mesh and must be
        excluded from FSDP2 wrapping.  Call this after creating the trainer
        but before ``trainer.train()``.
        """
        if not self.ignored_modules:
            return
        fsdp_plugin = getattr(trainer.accelerator.state, 'fsdp_plugin', None)
        if fsdp_plugin is None:
            return
        setattr(fsdp_plugin, 'ignored_modules', self.ignored_modules)
        logger.info(f'FSDP2: set ignored_modules for expert parallel ({len(self.ignored_modules)} modules)')

    # ------------------------------------------------------------------
    # Prepare — entry point
    # ------------------------------------------------------------------

    def prepare(self, ep_size: int, model: nn.Module):
        """Initialize expert parallel for the given model.

        Args:
            ep_size: Number of expert-parallel ranks. Must divide world_size.
            model: The HuggingFace MoE model.
        """
        if self._prepared:
            logger.warning('ExpertParallel.prepare() called twice; skipping.')
            return

        # Read config — try multiple attribute names for different models
        self.n_routed_experts = self._get_n_routed_experts(model.config)
        self.n_shared_experts = self._get_n_shared_experts(model.config)
        self.num_experts_per_tok = self._get_num_experts_per_tok(model.config)

        assert self.n_routed_experts is not None and self.n_routed_experts > 0, \
            'Cannot find n_routed_experts / num_experts in model config'
        assert self.num_experts_per_tok is not None and self.num_experts_per_tok > 0, \
            'Cannot find num_experts_per_tok in model config'
        assert self.n_routed_experts % ep_size == 0, \
            f'{self.n_routed_experts=} is not divisible by {ep_size=}'

        self.ep_size = ep_size
        self.num_experts_per_rank = self.n_routed_experts // self.ep_size
        self.model_dtype = next(model.parameters()).dtype

        # Enable output_router_logits for aux loss support
        HfConfigFactory.set_config_attr(model.config, 'output_router_logits', True)

        self._init_device_mesh()
        self.process_moe(model)
        self._prepared = True


expert_parallel = ExpertParallel()