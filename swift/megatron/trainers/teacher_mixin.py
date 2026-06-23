# Copyright (c) ModelScope Contributors. All rights reserved.
"""Shared Megatron teacher infrastructure for GKD and GRPO (OPD-RL).

Centralizes teacher resolution (separate model / API server / same-model LoRA
self-distillation / dynamic self-distillation), local teacher loading, offload, and
the teacher-API logprob fetch. GKD adds full-sequence top-k assembly on top; GRPO
(OPD-RL) consumes the sampled-token logp in the completion frame.
"""
import torch
from contextlib import contextmanager
from megatron.core import mpu
from transformers import AutoConfig
from typing import List, Optional

from swift.infer_engine.protocol import RequestConfig, RolloutInferRequest
from swift.megatron.model import get_mcore_model
from swift.rlhf_trainers.utils import parse_prompt_logprobs
from swift.rlhf_trainers.vllm_client import VLLMInferClient
from swift.utils import get_logger, is_last_rank
from .utils import load_megatron_model_to_gpu, offload_megatron_model_to_cpu

logger = get_logger()


class MegatronTeacherMixin:
    """Teacher setup + logprob fetching shared by Megatron GKD and GRPO."""

    def _setup_teacher(self) -> None:
        """Resolve teacher mode from args and init the API client when applicable.

        Sets ``teacher_model_server`` / ``use_teacher_api`` / ``_is_self_distillation`` /
        ``_teacher_use_disable_adapter`` / ``offload_teacher_model`` / ``_has_teacher``.
        Must be called before ``_init_rollout_engine`` (the API client lives on last rank).
        """
        args = self.args
        self.teacher_model_server = getattr(args, 'teacher_model_server', None)
        self.use_teacher_api = self.teacher_model_server is not None
        self.offload_teacher_model = args.offload_teacher_model
        self._teacher_use_disable_adapter = getattr(args, '_teacher_use_disable_adapter', False)
        self._is_self_distillation = (args.teacher_model is None and self.teacher_model_server is None)
        self._has_teacher = (
            args.teacher_model is not None or self.teacher_model_server is not None
            or self._teacher_use_disable_adapter)
        self.teacher_models = None
        self.teacher_client = None
        if self.use_teacher_api:
            self.teacher_client = VLLMInferClient(base_urls=[self.teacher_model_server]) if is_last_rank() else None

    def _load_teacher_model(self) -> None:
        """Load the separate local teacher mcore model (called from ``prepare_model``).

        No-op for the API path, dynamic self-distillation, and same-model LoRA
        (disable_adapter) — those reuse the student weights or an external server.
        """
        if self.use_teacher_api or self._is_self_distillation or self._teacher_use_disable_adapter:
            return
        args = self.args
        vp_size = getattr(args, 'virtual_pipeline_model_parallel_size', None)
        assert vp_size is None or vp_size == 1, 'Teacher distillation does not support VPP.'
        self.teacher_hf_config = AutoConfig.from_pretrained(args.teacher_model_dir, trust_remote_code=True)
        self.teacher_models = get_mcore_model(args, self.teacher_hf_config)
        self.teacher_config = self.teacher_models[0].config
        if not args.use_cpu_initialization:
            for teacher_model in self.teacher_models:
                teacher_model.cuda(torch.cuda.current_device())
        for teacher_model in self.teacher_models:
            teacher_model.requires_grad_(False)
            teacher_model.eval()
        self.teacher_config.bridge.load_weights(self.teacher_models, args.teacher_model_dir)
        if self.offload_teacher_model:
            self._offload_teacher_models()
            logger.info('Teacher models offloaded to CPU to save GPU memory')

    def _offload_teacher_models(self) -> None:
        if self.teacher_models and not self.use_teacher_api:
            offload_megatron_model_to_cpu(self.teacher_models)

    def _load_teacher_models_to_gpu(self) -> None:
        if self.teacher_models and not self.use_teacher_api:
            load_megatron_model_to_gpu(self.teacher_models, load_grad=False)

    @contextmanager
    def load_teacher_model_context(self):
        """Load the teacher to GPU for a forward and offload after (when offloading is on)."""
        if not self.offload_teacher_model:
            yield
            return
        self._load_teacher_models_to_gpu()
        try:
            yield
        finally:
            self._offload_teacher_models()

    def _fetch_teacher_parsed_logprobs(self, requests: List[RolloutInferRequest], topk: int):
        """Query the teacher API for prompt logprobs; return this DP rank's parsed slice.

        ``topk == 0`` -> sampled-token logp (OPD-RL); ``topk > 0`` -> top-k (GKD). Gathers
        the rollout-group-rank-0 contributions to the main process, infers once, broadcasts,
        and slices back this DP rank's contiguous segment.
        """
        rollout_group = self._get_rollout_group()
        rollout_rank = torch.distributed.get_rank(group=rollout_group)
        contribution = list(requests) if rollout_rank == 0 else []

        world_size = torch.distributed.get_world_size()
        all_contributions = [None] * world_size
        torch.distributed.all_gather_object(all_contributions, contribution)

        if self.is_main_process:
            flat_global = []
            for c in all_contributions:
                if c:
                    flat_global.extend(c)
            request_config = RequestConfig(prompt_logprobs=topk, max_tokens=1, temperature=0.0)
            responses = self.teacher_client.infer(flat_global, request_config=request_config, use_tqdm=False)
            parsed_global = [parse_prompt_logprobs(r, topk=topk) for r in responses]
        else:
            parsed_global = None

        obj_list = [parsed_global]
        torch.distributed.broadcast_object_list(obj_list, src=world_size - 1)
        parsed_global = obj_list[0]

        n = len(requests)
        dp_rank = mpu.get_data_parallel_rank()
        return parsed_global[dp_rank * n:(dp_rank + 1) * n]
