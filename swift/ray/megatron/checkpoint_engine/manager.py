# Copyright (c) ModelScope Contributors. All rights reserved.
from __future__ import annotations

import ray
from typing import Any, List, Optional

from swift.utils.logger import get_logger
from .nccl import NCCLCheckpointEngine

logger = get_logger()


class CheckpointEngineManager:

    def __init__(
        self,
        train_actors: List[Any],
        rollout_replicas: List[Any],
        *,
        weight_sync_mode: str = 'nccl',
        is_colocated: bool = False,
        sleep_level: int = 1,
        train_group: Any,
    ):
        self.train_actors = train_actors
        self._rollout_replicas = rollout_replicas
        self.rollout_actors = [r.primary for r in rollout_replicas]
        self._weight_sync_mode = weight_sync_mode
        self.is_colocated = is_colocated
        self._train_group = train_group

        if is_colocated and sleep_level >= 2:
            logger.warning(
                'sleep_level=%d capped to 1 in colocate mode '
                '(out-of-process vLLM cannot safely discard all GPU memory).', sleep_level)
            sleep_level = 1
        self.sleep_level = sleep_level

        self.base_sync_done: bool = False
        self._model_keys: Optional[List[str]] = None
        self._sleeping_tags: set = set()

    def sync_weights(self, merge_and_sync: bool = True) -> None:
        """Synchronize weights from training model to rollout replicas."""
        if self.is_colocated:
            self.sleep_rollout()
            self.wake_up_rollout(tags=['weights'])
        if self._weight_sync_mode == 'naive':
            self._sync_weights_naive(merge_and_sync)
        else:
            self._sync_weights_nccl(merge_and_sync)

    def sleep_rollout(self) -> None:
        if self._sleeping_tags:
            return
        for replica in self._rollout_replicas:
            replica.sleep(level=self.sleep_level)
        self._sleeping_tags = {'weights', 'kv_cache'}
        logger.debug('CheckpointEngineManager: rollout replicas sleeping (level=%d)', self.sleep_level)

    def wake_up_rollout(self, tags: Optional[List[str]] = None) -> None:
        if not self._sleeping_tags:
            return
        for replica in self._rollout_replicas:
            replica.wake_up(tags=tags)
        if tags is None:
            self._sleeping_tags.clear()
        else:
            self._sleeping_tags -= set(tags)
        logger.debug('CheckpointEngineManager: rollout wake_up tags=%s, still_sleeping=%s', tags, self._sleeping_tags)

    def _sync_weights_naive(self, merge_and_sync: bool) -> None:
        tg = self._train_group
        adapter_only = self.base_sync_done and not merge_and_sync
        need_merge = not adapter_only and merge_and_sync
        if need_merge:
            tg.merge_lora()
        if self.is_colocated:
            tg.offload_to_cpu()
        try:
            tg.update_weights(adapter_only=adapter_only)
        finally:
            if self.is_colocated:
                tg.reload_to_gpu()
            if need_merge:
                tg.unmerge_lora()
        if not self.base_sync_done:
            self.base_sync_done = True
            logger.debug('CheckpointEngineManager[naive]: initial weight sync done')

    def _sync_weights_nccl(self, merge_and_sync: bool) -> None:
        """NCCL broadcast weight sync path.

        Lifecycle:
        1. prepare_checkpoint_engine on all actors
        2. build_topology
        3. init_process_group on all actors (concurrent — required for TCPStore)
        4. send_weights (train) + receive_weights (rollout) concurrently
        5. finalize_checkpoint_engine on all actors
        """
        n_train = len(self.train_actors)
        n_rollout = len(self.rollout_actors)

        # 1. Prepare — train side: rank 0 is master, others are not
        is_master_flags = [True] + [False] * (n_train - 1)
        prepare_refs = [
            actor.prepare_checkpoint_engine.remote(flag) for actor, flag in zip(self.train_actors, is_master_flags)
        ]
        prepare_results = ray.get(prepare_refs)
        model_metadata = prepare_results[0]

        # 1b. Prepare — rollout side: all non-master
        rollout_prepare_refs = [actor.prepare_checkpoint_engine.remote(False) for actor in self.rollout_actors]
        ray.get(rollout_prepare_refs)

        # 2. Build topology
        model_kwargs, rollout_kwargs = NCCLCheckpointEngine.build_topology(n_train, n_rollout, [model_metadata])

        # 3. Init process groups (MUST be concurrent — TCPStore server
        #    blocks until all clients connect)
        train_init_refs = [
            actor.init_checkpoint_process_group.remote(
                rank=model_kwargs['rank'][i],
                world_size=model_kwargs['world_size'][i],
                master_metadata=model_kwargs['master_metadata'][i],
            ) for i, actor in enumerate(self.train_actors)
        ]
        rollout_init_refs = [
            actor.init_checkpoint_process_group.remote(
                rank=rollout_kwargs['rank'][i],
                world_size=rollout_kwargs['world_size'][i],
                master_metadata=rollout_kwargs['master_metadata'][i],
            ) for i, actor in enumerate(self.rollout_actors)
        ]
        ray.get(train_init_refs + rollout_init_refs)

        # 4. Send/receive weights (concurrent)
        adapter_only = self.base_sync_done and not merge_and_sync
        need_merge = not adapter_only and merge_and_sync
        peft_config = None
        if adapter_only:
            peft_config = ray.get(self.train_actors[0].get_peft_config_dict.remote())

        if need_merge:
            merge_refs = [actor.merge_lora.remote() for actor in self.train_actors]
            ray.get(merge_refs)

        train_send_refs = [
            actor.send_checkpoint_weights.remote(adapter_only=adapter_only) for actor in self.train_actors
        ]
        rollout_recv_refs = [
            actor.receive_checkpoint_weights.remote(
                base_sync_done=self.base_sync_done,
                peft_config=peft_config,
            ) for actor in self.rollout_actors
        ]
        ray.get(train_send_refs + rollout_recv_refs)

        if need_merge:
            unmerge_refs = [actor.unmerge_lora.remote() for actor in self.train_actors]
            ray.get(unmerge_refs)

        # 5. Finalize
        train_fin_refs = [actor.finalize_checkpoint_engine.remote() for actor in self.train_actors]
        rollout_fin_refs = [actor.finalize_checkpoint_engine.remote() for actor in self.rollout_actors]
        ray.get(train_fin_refs + rollout_fin_refs)

        if not self.base_sync_done:
            self.base_sync_done = True
            logger.info('CheckpointEngineManager[nccl]: initial weight sync to %d replica(s) '
                        '(lora_only=%s)', n_rollout, not merge_and_sync)
