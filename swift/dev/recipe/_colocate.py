from __future__ import annotations
import time
from typing import Any, Iterable

from swift.dev.utils.logger import get_logger

logger = get_logger()


class ColocateHandover:
    """Wake/sync/offload/reload around a generate phase, for a trainer and sampler sharing GPUs.

    ``enter`` hands the device to the sampler and pushes the trained policy into it; ``exit`` hands the
    device back to the trainer. A disaggregated (``colocate=False``) sampler lives on its own GPUs, so
    there is no device to hand over: ``enter`` reduces to a plain weight sync and ``exit`` is a no-op.

    ``model`` / ``sampler`` / ``manager`` are the twinkle objects the caller already built (the manager
    wires the CUDA-IPC or NCCL path per its ``mode``); ``merge_and_sync`` is forwarded to
    ``manager.sync_weights`` (merged base weights every step, correct for both full and LoRA training).
    """

    def __init__(self,
                 model: Any,
                 sampler: Any,
                 manager: Any,
                 *,
                 colocate: bool,
                 merge_and_sync: bool = True,
                 sleep_level: int = 0):
        self.model = model
        self.sampler = sampler
        self.manager = manager
        self.colocate = colocate
        self.merge_and_sync = merge_and_sync
        self.sleep_level = sleep_level

    def enter(self) -> None:
        """Give the sampler the device and the trained weights, ready to generate.

        Colocate order: wake the sampler's WEIGHTS so the sync has somewhere to write, sync the trained
        policy into them, step the trainer aside (offload), then wake ONLY the KV cache so the sampler can
        generate. The two wakes are tag-disjoint on purpose -- vLLM's ``wake_up`` aborts the whole call on
        the first tag that is not still sleeping, so after waking ``weights`` the second wake may ask only
        for ``kv_cache``: passing ``wake_up()`` (both tags) would hit the already-awake ``weights``, return
        early, and leave the KV cache discarded, so the next generate builds attention metadata on freed
        memory and dies with a CUDA illegal access. (A freshly built sampler is not sleeping yet, so on the
        first round both wakes no-op -- which is why the bug only bites from round two on.)
        """
        if self.colocate:
            self.sampler.wake_up(tags=['weights'])
        self.manager.sync_weights(merge_and_sync=self.merge_and_sync)
        if self.colocate:
            self.model.offload_to_cpu()
            self.sampler.wake_up(tags=['kv_cache'])  # weights already awake; ready to generate

    def exit(self) -> None:
        """Take the device back for the trainer: sleep the sampler, then reload the trainer's weights.

        ``sleep_level`` selects how much the sampler releases: level 1 offloads only the KV cache (weights
        stay resident, so the next ``enter`` re-syncs in place), level 2 also drops the weights. The
        default ``0`` maps to twinkle's own default depth (level 1), keeping the previous behaviour
        byte-for-byte; 1/2 are passed through unchanged.
        """
        if self.colocate:
            self.sampler.sleep(level=self.sleep_level or 1)
            self.model.reload_to_gpu()


class AdaptiveColocateHandover:
    """以 actor 子集暂停生成、同步权重，并在训练与 shared 引擎之间交接 GPU。"""

    serializes = False
    adaptive_shared = True

    def __init__(self,
                 model: Any,
                 sampler: Any,
                 dedicated_manager: Any,
                 *,
                 shared_manager: Any = None,
                 additional_models: Iterable[Any] = (),
                 sleep_level: int = 1):
        if sleep_level not in (0, 1):
            raise ValueError('adaptive shared sampler supports only sleep_level 0 or 1.')
        self.model = model
        self.models = (model, *tuple(additional_models))
        self.sampler = sampler
        self.dedicated_manager = dedicated_manager
        self.shared_manager = shared_manager
        self.controller = None
        self.sleep_level = sleep_level or 1
        self.in_generation = False
        self.enabled = False
        self.borrow_time = 0.0
        self.reclaim_time = 0.0

    @property
    def total_handover_time(self) -> float:
        return self.borrow_time + self.reclaim_time

    @staticmethod
    def _wait(result):
        if callable(result):
            return result()
        return result

    def set_enabled(self, enabled: bool) -> None:
        self.enabled = bool(enabled) and self.shared_manager is not None

    def initialize_shared_workers(self, group, ranks, *, engine_args):
        """构造 shared 引擎前释放训练显存，休眠引擎后恢复所有训练模型。"""
        worker_ids = ()
        initialization_error = None
        can_reload = True
        self.in_generation = True
        try:
            for model in self.models:
                self._wait(model.offload_to_cpu())
            worker_ids = self.sampler.add_workers(group, ranks, shared=True, engine_args=engine_args)
            self.sampler.pause_workers(worker_ids)
            self.sampler.pause_generation(_worker_ids=worker_ids)
            self._wait(self.sampler.sleep(level=self.sleep_level, _worker_ids=worker_ids))
        except BaseException as exc:
            initialization_error = exc
            try:
                if worker_ids and not self.sampler.remove_workers(worker_ids):
                    raise RuntimeError('Shared sampler initialization still has unfinished worker calls.')
            except Exception:
                logger.warning('初始化失败后未能完整关闭 shared sampler actors。', exc_info=True)
            can_reload = not set(worker_ids) & set(self.sampler.worker_ids(shared=True))
            raise
        finally:
            if can_reload:
                try:
                    self._reload_models()
                    self.in_generation = False
                except BaseException:
                    if initialization_error is None:
                        raise
                    logger.warning('初始化失败后未能完整恢复训练模型。', exc_info=True)
        return worker_ids

    def _borrow_shared(self, *, publish: bool):
        shared_ids = self.sampler.worker_ids(shared=True)
        outstanding = any(self.sampler.worker_inflight(shared_ids))
        if not shared_ids or (not self.enabled and not outstanding) or self.in_generation:
            return
        started = time.perf_counter()
        # 即使停止 shared 新准入，也必须让已经提交的请求继续生成并正常接收新 policy。
        self.in_generation = True
        try:
            self._wait(self.sampler.wake_up(tags=['weights'], _worker_ids=shared_ids))
            if publish:
                trainer_ranks = [ranks[0] for ranks in self.sampler.worker_ranks(shared_ids)]
                self.shared_manager.set_receiver_workers(shared_ids, colocate_trainer_ranks=trainer_ranks)
                self.shared_manager.sync_weights(merge_and_sync=True)
            for model in self.models:
                self._wait(model.offload_to_cpu())
            self._wait(self.sampler.wake_up(tags=['kv_cache'], _worker_ids=shared_ids))
            self.sampler.resume_generation(_worker_ids=shared_ids)
            if self.enabled:
                self.sampler.activate_workers(shared_ids)
        except BaseException:
            try:
                self.reclaim()
            except Exception:
                logger.warning('shared 借卡失败后未能完整恢复训练模型。', exc_info=True)
            raise
        self.borrow_time += time.perf_counter() - started

    def borrow(self, *, publish: bool) -> None:
        if not publish:
            self._borrow_shared(publish=False)
            return
        if self.in_generation:
            self.reclaim()
        suspended = self.sampler.worker_ids()
        self.sampler.suspend_admission(suspended)
        dedicated_ids = self.sampler.worker_ids(shared=False, states=('ACTIVE', 'DRAINING'))
        try:
            self.sampler.pause_generation(_worker_ids=dedicated_ids)
            if self.controller is not None:
                dedicated_ids = self.controller.prepare_publish()
            self.dedicated_manager.set_receiver_workers(dedicated_ids)
            self.dedicated_manager.sync_weights(merge_and_sync=True)
            self._borrow_shared(publish=True)
            if self.controller is not None:
                self.controller.finish_publish()
            self.sampler.resume_generation(_worker_ids=dedicated_ids)
            self.sampler.resume_admission(suspended + dedicated_ids)
        except BaseException as error:
            try:
                if self.controller is not None:
                    self.controller.fail_publish(error)
            except Exception:
                logger.warning('发布失败后未能完整清理新增 sampler actors。', exc_info=True)
            try:
                self.reclaim()
            except Exception:
                logger.warning('发布失败后未能完整恢复训练模型。', exc_info=True)
            raise

    def _reload_models(self):
        error = None
        for model in reversed(self.models):
            try:
                self._wait(model.reload_to_gpu())
            except BaseException as exc:
                error = error or exc
        if error is not None:
            raise error

    def reclaim(self) -> None:
        """停止 shared 新准入，等生成暂停后释放显存，最后恢复所有训练模型。"""
        if not self.in_generation:
            return
        started = time.perf_counter()
        shared_ids = self.sampler.worker_ids(shared=True)
        try:
            self.sampler.pause_workers(shared_ids)
            self.sampler.pause_generation(_worker_ids=shared_ids)
            self._wait(self.sampler.sleep(level=self.sleep_level, _worker_ids=shared_ids))
            # pause/sleep 失败时保留交接状态；终止 actors 后再恢复训练显存。
            self._reload_models()
            self.in_generation = False
        finally:
            self.reclaim_time += time.perf_counter() - started

    def close_channels(self):
        try:
            self.dedicated_manager.close()
        finally:
            if self.shared_manager is not None:
                self.shared_manager.close()

    def enter(self) -> None:
        self.borrow(publish=True)

    def exit(self) -> None:
        self.reclaim()

    def drained(self, inflight: int) -> bool:
        return True
