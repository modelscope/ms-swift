from __future__ import annotations
import copy
import hashlib
import math
import os
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, Tuple

import json

import numpy as np

# The reward-channel layer -- resolving, building, placing and scoring a MODEL reward (``--orm`` / ``--prm``)
# -- lives in the shared ``swift.dev.rewards.scorer`` so online RL (grpo / ppo) drives the SAME code, not a
# second copy: a capability two businesses use belongs in a common file, and neither recipe imports the
# other. run_infer re-exports it under its historical private names purely for backward compatibility, so
# this module's call sites and the infer tests keep importing ``_resolve_reward_channel`` / ``_RewardSlot``
# / ... unchanged; new consumers import the public names from ``swift.dev.rewards.scorer`` directly.
from swift.dev.rewards.scorer import (
    SAMPLER_GROUP as _SAMPLER_GROUP,
    RewardModelSpec as _RewardModelSpec,
    RewardSlot as _RewardSlot,
    build_model_reward as _build_model_reward,
    exclusive_reward_groups as _exclusive_reward_groups,
    judge_query as _judge_query,
    normalize_scores as _normalize,
    parse_judge_score as _parse_judge_score,
    plan_sampling_device_groups,
    render_conversation as _render_conversation,
    resolve_reward_channel as _resolve_reward_channel,
    resolve_reward_model as _resolve_reward_model,
)
# The dataset-storage registry owns the store formats; its write side (``.writers``) owns the
# incremental / checkpointed / batch writer family the run streams rows through.
from swift.dev.dataset.storage import STORAGE_MAPPING
from swift.dev.dataset.storage.writers import _write_jsonl, open_store_writer, store_output_complete
from swift.dev.utils.logger import get_logger
from swift.dev.components.base import RecipeComponent

if TYPE_CHECKING:
    from swift.dev.config import (
        DatasetConfig,
        DistributedConfig,
        GenerationConfig,
        InferConfig,
        ModelConfig,
        PluginConfig,
        QuantizeConfig,
        RLConfig,
        RolloutConfig,
        TemplateConfig,
        TunerConfig,
    )

logger = get_logger()


class InferComponent(RecipeComponent):
    """The infer entry point: a sampling path and a forward-pass path, split inside ``__call__``.

    A causal LM samples a candidate group per prompt, optionally scores it, then emits rows: generation
    goes through one ``RolloutEngine`` (single- and multi-turn alike, over a local sampler), scoring
    through ``_RewardChannels`` (rule-based ORM/PRM funcs and/or model rewards), and emission through
    the ``output_format`` strategy -- so a new storage shape is a new emit function, not a new recipe.

    A pooling task (``seq_cls`` / ``embedding`` / ``reranker``) or a ``generative_reranker`` never
    samples: one plain forward pass over the same rows (``_forward_pass``), one value per row.
    """

    def __init__(
        self,
        model_config: ModelConfig,
        template_config: TemplateConfig,
        dataset_config: DatasetConfig,
        infer_config: InferConfig,
        generation_config: Optional[GenerationConfig],
        *,
        rl_config: Optional[RLConfig],
        rollout_config: Optional[RolloutConfig],
        backend: str,
        engine_args: Optional[Dict[str, Any]],
        distributed_config: Optional[DistributedConfig],
        adapters: Optional[List[str]],
        quantize_config: Optional[QuantizeConfig],
        max_rows: Optional[int],
        split_dataset_ratio: float,
        output_path: Optional[str],
        shutdown: bool,
    ) -> None:
        self.model_config = model_config
        self.template_config = template_config
        self.dataset_config = dataset_config
        self.infer_config = infer_config
        self.generation_config = generation_config
        self.rl_config = rl_config
        self.rollout_config = rollout_config
        self.backend = backend
        self.engine_args = engine_args
        self.distributed_config = distributed_config
        self.adapters = adapters
        self.quantize_config = quantize_config
        self.max_rows = max_rows
        self.split_dataset_ratio = split_dataset_ratio
        self.output_path = output_path
        self.shutdown = shutdown

    def __call__(self) -> List[Dict[str, Any]]:
        model_config = self.model_config
        template_config = self.template_config
        dataset_config = self.dataset_config
        infer_config = self.infer_config
        generation_config = self.generation_config
        rl_config = self.rl_config
        rollout_config = self.rollout_config
        backend = self.backend
        engine_args = self.engine_args
        distributed_config = self.distributed_config
        adapters = self.adapters
        quantize_config = self.quantize_config
        max_rows = self.max_rows
        split_dataset_ratio = self.split_dataset_ratio
        output_path = self.output_path
        shutdown = self.shutdown
        from swift.dev.builders import (build_parallel_mesh, build_sampler, build_template, is_pooling_task,
                                        load_model_processor, load_prompt_only_rows, to_sampling_params)

        # A pooling task (seq_cls / embedding / reranker) or a generative_reranker never samples: one
        # plain forward pass, one value per row (see _forward_pass). Split before the sampling-side
        # validation below, which reads knobs (num_return_sequences, output_format) a forward pass
        # never uses.
        task_type = model_config.task_type or 'causal_lm'
        if is_pooling_task(task_type) or task_type == 'generative_reranker':
            return self._forward_pass(task_type)

        num_return = infer_config.num_return_sequences
        output_format = infer_config.output_format
        if output_format == 'dpo':
            if num_return < 2:
                raise ValueError(f"output_format='dpo' needs num_return_sequences >= 2 to have both a chosen "
                                 f'and a rejected candidate, got {num_return}.')
            if infer_config.n_best_to_keep >= num_return:
                raise ValueError(f'n_best_to_keep={infer_config.n_best_to_keep} must be < '
                                 f'num_return_sequences={num_return}: the lowest-scoring candidate becomes the '
                                 'rejected response, so it cannot also be a positive.')
        elif output_format == 'grpo':
            # A GRPO dump is the group itself -- every candidate's trajectory, trainable mask and rollout logps,
            # plus its reward when a channel scored it -- and the group-relative advantage is recomputed at
            # train time, not here. So it needs a real group (>= 2 candidates) and the rollout-token sidecar
            # that carries the logps: save_rollout_tokens is what forces logprob computation, so without it
            # there would be no old_logps to store. This is an explicit requirement, not a default derived from
            # output_format (a lone param never silently sets another).
            if num_return < 2:
                raise ValueError(f"output_format='grpo' needs num_return_sequences >= 2 to dump a group, got "
                                 f'{num_return}.')
            if not infer_config.save_rollout_tokens:
                raise ValueError("output_format='grpo' stores each candidate's rollout logprobs (old_logps), "
                                 'which are only produced when save_rollout_tokens forces logprob computation. '
                                 'Re-run with --save_rollout_tokens true.')
        multi_turn_enabled = bool(rl_config and rl_config.max_turns is not None)
        if infer_config.score_ground_truth and multi_turn_enabled:
            # The reference answer is a single assistant turn with no tool trajectory, while a multi-turn
            # candidate's reward reads its full messages / rollout_infos. Scoring both on one axis compares
            # different shapes, so this is rejected (STAGE2_GOALS: score_ground_truth is mutually exclusive
            # with a multi-turn / gym rollout, whose reference has no rollout trajectory).
            raise ValueError(
                'score_ground_truth cannot combine with a multi-turn rollout: the reference answer has no '
                "tool trajectory to score against the candidates'. Turn off score_ground_truth, or run "
                'single-turn.')

        # Resolve the reward channels BEFORE twinkle init and the sampler build: each merged --orm/--prm list
        # becomes ordered rule/model slots, and the model slots' specs decide whether a reward model needs its
        # own GPU DeviceGroup (declared at init) and whether a generative judge reusing the sampler needs its
        # reward LoRA resident in the engine's adapter list at construction (vLLM/SGLang size their adapter
        # slots up front). The channel callables are built after the sampler, since a judge scores through it.
        sampler_model = model_config.model
        orm_slots = _resolve_reward_channel(
            rl_config.orm if rl_config is not None else [], infer_config.orm_adapter, sampler_model, backend,
            parallel_spec=rl_config.orm_parallel_spec if rl_config is not None else None, channel='orm')
        prm_slots = _resolve_reward_channel(
            rl_config.prm if rl_config is not None else [], infer_config.prm_adapter, sampler_model, backend,
            parallel_spec=rl_config.prm_parallel_spec if rl_config is not None else None, channel='prm')
        reuse_adapters = [slot.spec.adapter for slots in (orm_slots, prm_slots) for slot in slots
                          if slot.is_model and slot.spec.kind == 'generative_reuse' and slot.spec.adapter]
        reward_groups = _exclusive_reward_groups(orm_slots, prm_slots, distributed_config)
        if reward_groups and (distributed_config is None or distributed_config.mode != 'ray'):
            raise ValueError(
                f'reward channel(s) {[name for name, _, _ in reward_groups]} keep their own model resident on '
                "GPU and need a dedicated DeviceGroup, which only DistributedConfig.mode='ray' can place. Run "
                "with mode='ray' (and nproc_per_node), or use a sampler-reusing generative judge / an http(s) "
                'endpoint, which need no extra device.')
        if distributed_config is not None:
            _initialize_for_sampling(distributed_config, reward_groups)

        rows = load_prompt_only_rows(dataset_config, max_rows, split_dataset_ratio)
        if not rows:
            raise ValueError('run_infer got an empty dataset. Set DatasetConfig.dataset or .val_dataset.')
        logger.info(f'run_infer: {len(rows)} prompts, backend={backend}, num_return_sequences={num_return}, '
                    f'output_format={output_format}')

        # A 'dpo' or resumed run uses the checkpointed writer (tmp live-write, per-batch resume snapshot,
        # atomic finalize); a plain 'all' run uses the incremental writer, which also gathers across DP
        # ranks. An existing complete checkpoint output short-circuits the run unless override_exist_file,
        # so re-running a finished job is a no-op.
        # Whether the store_format has those incremental/resumable line writers is a capability the format
        # itself declares in the storage registry (``DatasetStorage.incremental``; only jsonl sets it so
        # far), not something this component decides: every other writable store_format serialises once at
        # finish, and validate_infer_config has already rejected resume for it. A 'dpo' run that picks one
        # forfeits its crash-safe checkpointing -- say so rather than let the choice pass silently. The
        # resulting choice of writer itself belongs to open_store_writer.
        incremental = STORAGE_MAPPING[infer_config.store_format].incremental
        if not incremental and output_format == 'dpo':
            logger.warning(f"output_format='dpo' with store_format={infer_config.store_format!r} writes one store "
                           'at the end, so it cannot checkpoint/resume a long best-of-n run; use --store_format '
                           'jsonl for that.')
        use_checkpoint = (bool(output_path) and incremental
                          and (output_format == 'dpo' or infer_config.resume))
        if use_checkpoint:
            from twinkle.utils import Platform

            local_world = max(1, int(Platform.get_world_size()))
            if local_world > 1 and (distributed_config is None or distributed_config.mode != 'ray'):
                # The checkpoint writer owns one tmp file and one state file and gathers nothing (unlike
                # the incremental writer): under a multi-rank torchrun launch every rank would open the
                # same tmp at its own offset and corrupt it. It assumes the single driving process of the
                # ray layout, so reject the multi-rank local launch up front.
                raise ValueError(
                    f'a checkpointed run (output_format={output_format!r}, resume={infer_config.resume}) '
                    f'needs a single driving process, but this launch has {local_world} local ranks, each '
                    'of which would write the same checkpoint files. Run it under DistributedConfig.mode='
                    "'ray', or with a single rank.")
            if store_output_complete(output_path) and not infer_config.override_exist_file:
                logger.info(f'run_infer: {output_path} exists and override_exist_file is False; nothing to do.')
                return []

        cache = _CandidateCache(infer_config.cache_files)
        quant_method = getattr(quantize_config, 'quant_method', None)
        if backend == 'no' and quant_method is not None:
            raise ValueError(
                f'quant_method={quant_method!r} cannot affect sampler={backend!r}, which loads no local '
                'model. Configure quantization on the remote server or omit --quant_method.')

        template = None
        adapter_path = adapters[0] if adapters else None
        if backend == 'no':
            sampler = None
        else:
            _, processor = load_model_processor(model_config)
            template = build_template(template_config, processor)
            # A generative judge reusing this sampler scores through a reward LoRA, which must be resident
            # before the first request, so merge it into the sampling adapters the engine is built with.
            sampler_adapters = list(adapters or [])
            for reward_adapter in reuse_adapters:
                if reward_adapter not in sampler_adapters:
                    sampler_adapters.append(reward_adapter)
            sampler = build_sampler(
                model_config, backend=backend, engine_args=engine_args,
                device_mesh=build_parallel_mesh(distributed_config, none_if_single_rank=True), template=template,
                adapters=sampler_adapters or None,
                remote_group=_SAMPLER_GROUP if distributed_config is not None
                and distributed_config.mode == 'ray' else None,
                quantize_config=quantize_config)

        channels = _build_channels(
            infer_config, rl_config, orm_slots=orm_slots, prm_slots=prm_slots, sampler=sampler,
            model_config=model_config, template_config=template_config, backend=backend,
            engine_args=engine_args, distributed_config=distributed_config, quantize_config=quantize_config)

        rollout = None
        if sampler is not None:
            # One generation entry for single- and multi-turn alike: a RolloutEngine wrapping the already-built
            # sampler (injected, so this recipe keeps owning its lifecycle and calls close(), not shutdown()).
            from swift.dev.rollout import RolloutEngine
            rollout = RolloutEngine(sampler=sampler, template=template)
            if multi_turn_enabled:
                # Tools are opt-in: with RolloutConfig.tools set, the engine builds its own sandbox env pool
                # and resolves the tool plugins (each episode leases its own env); otherwise no tools.
                rollout.configure_multi_turn(
                    max_turns=rl_config.max_turns, max_trajectory_tokens=rl_config.max_trajectory_tokens,
                    rollout_config=rollout_config)

        batches = _plan_batches(len(rows), infer_config.batch_size, infer_config.max_batches)
        params = to_sampling_params(generation_config, num_samples=num_return)
        ## 这几个选项是否应该由数据湖来完成支持？
        ## 数据湖来决定flush时机和阅读恢复、续写
        writer = open_store_writer(output_path, infer_config.store_format,
                                   store_fields=infer_config.store_fields, checkpoint=use_checkpoint,
                                   resume=infer_config.resume)
        # Rollout-token sidecar: only built when save_rollout_tokens is on, and only a local backend can feed
        # it (the message-only 'no' backend produces no tokens). The NPZ dir sits beside
        # the jsonl and each row embeds a path relative to output_dir, so the pair stays portable.
        recorder = None
        if infer_config.save_rollout_tokens:
            if not output_path:
                raise ValueError('save_rollout_tokens needs an output_path to place the NPZ sidecar beside.')
            recorder = _RolloutRecorder(
                output_path + '.rollout_tokens', base_dir=os.path.dirname(output_path) or 'output', enabled=True)

        if output_format == 'grpo' and channels.empty:
            # No reward channel is a valid GRPO dump -- the group's trajectories, masks and old_logps are the
            # corpus, and training scores/advantages them later. Note once why ``scores`` comes out null.
            logger.info("output_format='grpo' with no reward channel (--orm / --prm) "
                        "stores each prompt's group trajectories + rollout logps with null scores; reward is "
                        'left to the training step.')

        results: List[Dict[str, Any]] = []
        # Throughput/token summary, the dev counterpart of legacy's InferStats: prompts and generated tokens
        # are counted as batches complete, then reported once at the end. Token counts come only from a local
        # backend (a message-only 'no' run has none), so tokens/s is logged only when tokens were seen.
        num_prompts = 0
        num_generated_tokens = 0
        start_time = time.perf_counter()
        emit = {'dpo': _emit_dpo, 'grpo': _emit_grpo}.get(output_format, _emit_all)
        # Two dataset routes, chosen by SAMPLER CAPABILITY (not by backend name): a sampler exposing the
        # non-blocking submit_generation (twinkle GenerationSubmissionMixin -- vLLM/SGLang) rides the SAME
        # per-sample StreamingDriver online RL uses, on the null ledger (version pinned at 0, nothing
        # publishes), reusing its rollout + the group PRM/ORM scoring; anything else (transformers, or the
        # cache-only 'no' backend) keeps the batch loop. The streaming/TUI sink is separate (infer_tui.py).
        async_route = sampler is not None and callable(getattr(sampler, 'submit_generation', None))
        try:
            if async_route:
                # A deterministic seed makes the dataloader's shuffle replayable, which is what resume
                # relies on to skip the already-emitted prefix in the same draw order (see _drive_per_sample).
                seed = getattr(dataset_config, 'data_seed', 0) or 0
                async_results, async_prompts, async_tokens = self._drive_per_sample(
                    rows=rows, rollout=rollout, channels=channels, cache=cache, writer=writer,
                    recorder=recorder, emit=emit, num_return=num_return,
                    multi_turn_enabled=multi_turn_enabled, adapter_path=adapter_path,
                    sampling_params=asdict(to_sampling_params(generation_config)), seed=seed)
                results.extend(async_results)
                num_prompts += async_prompts
                num_generated_tokens += async_tokens
            else:
                for index, (start, end) in enumerate(batches):
                    if writer.skip(index):
                        continue
                    batch = rows[start:end]
                    logger.info(f'run_infer: batch {index + 1}/{len(batches)} ({len(batch)} prompts)')
                    trajectories, ground_truths, groups = _sample_candidates(
                        sampler, rollout, batch, params, template_config, infer_config, cache, backend,
                        adapter_path, multi_turn_enabled)
                    batch_rows: List[Dict[str, Any]] = []
                    for row, trajectory, ground_truth, group in zip(batch, trajectories, ground_truths, groups):
                        group = [candidate for candidate in group if candidate.text]
                        if not group:
                            continue
                        num_prompts += 1
                        num_generated_tokens += sum(len(candidate.response_token_ids) for candidate in group)
                        scores = _score_group(group, row, trajectory, ground_truth, channels, infer_config)
                        batch_rows.extend(emit(row, trajectory, ground_truth, group, scores, infer_config, recorder))
                    results.extend(batch_rows)
                    writer.write(batch_rows)
                    writer.checkpoint(index)
        finally:
            channels.shutdown()
            if rollout is not None:
                rollout.close()
            if shutdown and sampler is not None:
                sampler.shutdown()

        writer.finish(results)
        runtime = time.perf_counter() - start_time
        stats: Dict[str, Any] = {
            'num_prompts': num_prompts,
            'runtime': round(runtime, 3),
            'prompts/s': round(num_prompts / runtime, 3) if runtime > 0 else 0.0,
        }
        if num_generated_tokens:
            stats['num_generated_tokens'] = num_generated_tokens
            stats['tokens/s'] = round(num_generated_tokens / runtime, 1) if runtime > 0 else 0.0
        logger.info(f'run_infer stats: {stats}')
        if infer_config.metric:
            if output_format != 'all':
                logger.warning("metric=%r applies only to output_format='all' rows, which carry one response/"
                               'labels to score; %r rows store a whole candidate group, so the metric is '
                               "skipped. Re-run with output_format='all' for a metric.", infer_config.metric,
                               output_format)
            else:
                logger.info(f'run_infer metric: {compute_metric(results, infer_config.metric)}')
        return results

    def _forward_pass(self, task_type: str) -> List[Dict[str, Any]]:
        """The non-sampling path: one forward pass over the rows, one plain value per row.

        Two task families reach here. A pooling task yields one value per row -- a vector for embedding,
        class logits for seq_cls, a score for reranker -- and picks its route by backend: vLLM / SGLang
        have a pooling head, so the model is built as a pooling engine and served through
        ``sampler.encode`` (the throughput path); transformers has no pooling head, so it falls back to
        ``build_model`` + ``forward_only(task=..)``, a plain HF forward (the reach path). A
        ``generative_reranker`` also yields one score per row, but it is a causal LM, not a pooling
        model: its score is the logprob difference between a positive token (``yes``) and a negative one
        (``no``) at the first generated position -- what ``get_generative_reranker_logits`` computes from
        the lm_head for the HF path, and what the generation path reproduces here from logprobs (the two
        agree because the log-softmax normaliser cancels in the difference). So vLLM/SGLang run it
        through *generation* -- neither serves a decoder-only reranker through ``encode``, and sglang
        must not be launched with ``--is-embedding`` -- while transformers runs the same HF forward the
        pooling path uses, with ``task='generative_reranker'``.

        Either way the row assembly is shared (:func:`_finalize_forward_results`), including its label
        rule: the label comes from the row's own ``label`` column rather than from a trailing assistant
        turn, because there is no completion to strip.
        """
        from swift.dev.builders import to_trajectory

        rows = _forward_rows(self.dataset_config, self.max_rows, self.split_dataset_ratio,
                             self.distributed_config)
        trajectories = [to_trajectory(row, list(row['messages']), self.template_config) for row in rows]
        reranker_use_activation = self.infer_config.reranker_use_activation

        if self.backend in ('vllm', 'sglang'):
            if task_type == 'generative_reranker':
                per_row = _generative_reranker_via_sampler(
                    self.model_config, self.template_config, self.generation_config, self.distributed_config,
                    trajectories, self.adapters, self.quantize_config, self.backend, self.engine_args,
                    self.shutdown, reranker_use_activation)
            else:
                per_row = _pooling_via_sampler(
                    self.model_config, self.template_config, self.distributed_config, trajectories,
                    self.adapters, self.quantize_config, self.backend, self.engine_args, self.shutdown,
                    reranker_use_activation)
        else:
            per_row = _forward_via_transformers(
                self.model_config, self.template_config, self.distributed_config, rows, trajectories,
                self.adapters, self.quantize_config,
                max_batch_size=(self.engine_args or {}).get('max_batch_size'),
                reranker_use_activation=reranker_use_activation)

        return _finalize_forward_results(rows, per_row, task_type, self.backend, self.output_path,
                                         self.infer_config.metric)

    def _drive_per_sample(self, *, rows: List[Dict[str, Any]], rollout: Any, channels: '_RewardChannels',
                          cache: '_CandidateCache', writer: Any, recorder: Any, emit: Any, num_return: int,
                          multi_turn_enabled: bool, adapter_path: Optional[str],
                          sampling_params: Dict[str, Any], seed: int) -> Tuple[List[Dict[str, Any]], int, int]:
        """The async dataset route: the SAME per-sample StreamingDriver online RL rides, on the null ledger.

        vLLM/SGLang expose the non-blocking ``submit_generation`` the driver's per-sample data plane needs, so
        each prompt's ``num_return`` candidates are admitted independently and emitted the instant a group
        completes -- no batch lock-step. The rollout seams (submit/poll/collect/cancel) are the ones GRPO's
        ``build_streaming_driver`` wraps; scoring reuses ``_RewardChannels.score`` at group completion (the same
        ORM/PRM channel the batch path scores with) and rows are shaped by the same ``emit`` + writers. The
        control plane is omitted (``ctx_mgr=None``): infer never publishes weights, so the driver runs its null
        ledger (version pinned at 0, no admission windows, no staleness drop). Returns ``(rows, prompts, tokens)``.

        Groups are emitted in DRAW order, not completion order (see ``_assembly_ready``), and the checkpoint
        counts prompts handled -- cache hits plus drawn groups. The emit order is deterministic (cache hits in
        row order, then draws in the dataloader's seeded order), so a resume restores ``resume_from`` and skips
        exactly that prefix: the cached head by row index, the drawn tail via ``skip_consumed_samples``.

        Under a multi-rank torchrun launch every rank drives its own shard (the loader slices each global
        batch across the dp ranks, so the ordinal is per-rank; a trailing batch narrower than the dp width
        is dropped, per the loader's contract) and the incremental writer's per-group gather merges the
        ranks' rows. A ray launch drives from one process instead -- the loader is pinned to dp=1 -- which
        is also the layout the checkpoint writer's resume path requires.
        """
        from swift.dev.builders import build_prompt_dataloader, split_prompt_and_reference
        from twinkle_agentic.async_rl.streaming_driver import StreamingDriver

        infer_config = self.infer_config
        template_config = self.template_config
        results: List[Dict[str, Any]] = []
        # Resume point: the count of prompts a prior run already wrote (its checkpoint), or 0 for a fresh run
        # (no checkpoint writer, or no state file). The emit order below is deterministic, so "the first N
        # prompts" is stable across runs and a resume can skip exactly that prefix.
        resume_from = max(0, getattr(writer, '_resume_from', -1))
        state: Dict[str, Any] = {'prompts': 0, 'tokens': 0, 'processed': 0, 'next_ordinal': 0}
        # Split once: the trajectory is what the sampler generates from, the reference is emit's label anchor.
        trajectories, ground_truths = split_prompt_and_reference(rows, template_config)
        # Cache short-circuit: a prompt already covered by cache_files is emitted verbatim, never driven (the
        # batch path's _sample_candidates does the same lookup), so only the rest pay for generation.
        to_sample: List[int] = []
        processed = 0
        cached_total = 0
        for index, trajectory in enumerate(trajectories):
            hit = cache.lookup(trajectory, num_return, multi_turn=multi_turn_enabled)
            if hit is None:
                to_sample.append(index)
                continue
            cached_total += 1
            already_done = processed < resume_from
            processed += 1
            if already_done:
                continue  # a prior run emitted it; the restored tmp file already holds its rows
            group = [candidate for candidate in hit if candidate.text]
            if group:
                state['prompts'] += 1
                scores = _score_group(group, rows[index], trajectory, ground_truths[index], channels,
                                      infer_config)
                out_rows = emit(rows[index], trajectory, ground_truths[index], group, scores, infer_config,
                                recorder)
                results.extend(out_rows)
                writer.write(out_rows)
            writer.checkpoint(processed)
        # The draws a prior run already emitted are exactly the first (resume_from - cached_total) of the
        # seeded stream, so fast-forward the dataloader there: the driver's ordinal 0 becomes the first draw
        # not yet emitted. When that covers every draw (a run that finished but did not finalize), there is
        # nothing left to generate.
        draw_skip = max(0, resume_from - cached_total)
        state['processed'] = cached_total + draw_skip
        if to_sample and draw_skip < len(to_sample):
            sample_rows = [rows[index] for index in to_sample]
            prompt_dataloader = build_prompt_dataloader(
                [row['messages'] for row in sample_rows],
                [{key: value for key, value in row.items() if key != 'messages'} for row in sample_rows],
                seed=seed, batch_size=_prompt_batch_size(infer_config.batch_size))
            if draw_skip:
                prompt_dataloader.skip_consumed_samples(draw_skip)
            # Per-draw metadata the driver's ordinal keys: the row it drew, its split trajectory and reference.
            meta: Dict[int, Tuple[Dict[str, Any], Dict[str, Any], Optional[str]]] = {}

            def _ordinal_of(record: Any) -> int:
                # group_key is f'{ordinal}#{group_seq}' (StreamingDriver.run); the ordinal is the draw id.
                return int(record.group_key.split('#')[0])

            def _submit(prompt_ref: Any, trajectory_idx: int, policy: Any) -> Any:
                ordinal = prompt_ref.ordinal
                if ordinal not in meta:
                    traj, gt = split_prompt_and_reference([prompt_ref.row], template_config)
                    meta[ordinal] = (prompt_ref.row, traj[0], gt[0])
                extras = {key: value for key, value in prompt_ref.row.items() if key != 'messages'}
                # adapter_name='' selects the sampler's default template, exactly as the batch route's
                # rollout.generate does (it passes adapter_path only, leaving adapter_name at its '' default);
                # the LoRA itself is loaded from adapter_path. A non-empty name with no template registered
                # would raise inside the sampler, and the null-ledger policy's name is a sentinel, not a real
                # adapter -- so '' is the only value that matches the batch route's sampling.
                return rollout.submit_sample(
                    meta[ordinal][1]['messages'], trajectory_idx, prompt_idx=ordinal,
                    sampling_params=sampling_params, prompt_extras=(extras or None),
                    force_logprobs=infer_config.save_rollout_tokens,
                    adapter_name='', adapter_path=adapter_path,
                    allow_partial_rollout=False, policy_version=0)

            def _poll(handles: List[Any]) -> List[Any]:
                return rollout.poll_completions(handles)

            def _collect(handle: Any) -> Any:
                return rollout.collect_sample(handle)

            def _cancel(handle: Any) -> None:
                rollout.cancel_sample(handle)

            def _assembly_ready(buffer: List[Any]) -> Optional[List[Any]]:
                # Pull the contiguous run of COMPLETE groups starting at the next ordinal to emit -- draw
                # order, not completion order. Infer emits one group per prompt, so (unlike GRPO) there is no
                # optimizer-step row quantum to fill; ordering by draw id is what keeps the checkpoint's
                # count equal to the draws consumed, so a resume skips exactly the emitted prefix. A group
                # still short of num_return (a straggler) holds the frontier, but the null ledger never drops
                # a sample, so it always completes and the wait is finite.
                by_ordinal: Dict[int, List[Any]] = {}
                for record in buffer:
                    by_ordinal.setdefault(_ordinal_of(record), []).append(record)
                pulled: List[Any] = []
                ordinal = state['next_ordinal']
                while len(by_ordinal.get(ordinal, ())) >= num_return:
                    pulled.extend(by_ordinal[ordinal])
                    ordinal += 1
                return pulled or None

            def _consume(records: List[Any]) -> None:
                by_ordinal: Dict[int, List[Any]] = {}
                for record in records:
                    by_ordinal.setdefault(_ordinal_of(record), []).append(record)
                for ordinal in sorted(by_ordinal):
                    group_records = by_ordinal[ordinal]
                    row, trajectory, ground_truth = meta[ordinal]
                    candidates = [
                        _candidate_from_sample(record.sample, multi_turn=multi_turn_enabled)
                        for record in group_records
                    ]
                    candidates = [candidate for candidate in candidates if candidate.text]
                    out_rows: List[Dict[str, Any]] = []
                    if candidates:
                        state['prompts'] += 1
                        state['tokens'] += sum(len(candidate.response_token_ids) for candidate in candidates)
                        scores = _score_group(candidates, row, trajectory, ground_truth, channels, infer_config)
                        out_rows = emit(row, trajectory, ground_truth, candidates, scores, infer_config, recorder)
                        results.extend(out_rows)
                    # Written even when the group came back empty: under a torchrun gather every rank must
                    # enter the same number of writer calls (one per drawn group), or the all_gather inside
                    # the writer mismatches across ranks and the run hangs.
                    writer.write(out_rows)
                    # Advance past this draw and record the cumulative count: every drawn group (even one whose
                    # candidates all came back empty) is a consumed prompt, so the checkpoint stays equal to
                    # the number of draws a resume must skip.
                    state['next_ordinal'] = ordinal + 1
                    state['processed'] += 1
                    writer.checkpoint(state['processed'])
                return None

            StreamingDriver(
                prompt_stream=prompt_dataloader, num_train_epochs=1.0, max_staleness=0,
                num_generations=num_return, groups_per_partition=1, buffer_depth=max(num_return * 4, 8),
                submit=_submit, poll=_poll, collect=_collect, assembly_ready=_assembly_ready, cancel=_cancel,
                consume=_consume, ctx_mgr=None, ctx=None).run()
        return results, state['prompts'], state['tokens']


def _sample_candidates(
    sampler: Any,
    rollout: Any,
    batch: List[Dict[str, Any]],
    params: Any,
    template_config: TemplateConfig,
    infer_config: InferConfig,
    cache: '_CandidateCache',
    backend: str,
    adapter_path: Optional[str],
    multi_turn_enabled: bool,
) -> Tuple[List[Dict[str, Any]], List[Optional[str]], List[List['_Candidate']]]:
    """One batch of prompts -> ``(trajectories, ground_truths, candidate groups)``, sampling only.

    Scoring and row shaping happen in the caller so both output formats share one sampling call.
    Prompts already covered by ``cache_files`` skip generation entirely -- they are the expensive part,
    and their candidates are reused verbatim.
    """
    from swift.dev.builders import split_prompt_and_reference

    trajectories, ground_truths = split_prompt_and_reference(batch, template_config)
    wanted = infer_config.num_return_sequences
    cached = [cache.lookup(trajectory, wanted, multi_turn=multi_turn_enabled) for trajectory in trajectories]
    # to_sample: the prompts the cache did NOT cover -- only these pay for generation, the rest reuse
    # their cached candidates verbatim. Fresh candidates are written back to their original index below.
    to_sample = [index for index, hit in enumerate(cached) if hit is None]
    candidates: List[List[_Candidate]] = [hit or [] for hit in cached]
    if to_sample:
        if sampler is None:
            raise ValueError("sampler='no' requires cache_files to cover every input prompt.")
        if rollout is None:
            raise RuntimeError('sampling was requested but no rollout engine was built.')
        selected_trajectories = [trajectories[i] for i in to_sample]
        prompt_extras = []
        # Per-prompt side inputs for the rollout: the dataset row's non-message columns merged with the
        # trajectory's, so a reward func / tool sees the same extras whether or not the prompt was cached.
        for index, trajectory in zip(to_sample, selected_trajectories):
            extra = {key: copy.deepcopy(value) for key, value in batch[index].items() if key != 'messages'}
            extra.update({key: copy.deepcopy(value) for key, value in trajectory.items() if key != 'messages'})
            prompt_extras.append(extra)
        # One generation call for single- and multi-turn alike: RolloutEngine.generate dispatches on whether
        # a multi-turn engine was configured, and both paths yield the same RolloutSample. logprobs are
        # forced only when the recorder keeps tokens (a plain run skips the extra compute); ``strict`` is a
        # transformers single-turn kwarg. ``adapter_path`` selects the LoRA the engine reserved slots for
        # and is honoured by both paths -- the multi-turn engine threads it into every per-turn sample.
        sample_kwargs: Dict[str, Any] = {}
        if backend == 'transformers':
            sample_kwargs['strict'] = infer_config.generation_strict
        if adapter_path is not None:
            sample_kwargs['adapter_path'] = adapter_path
        samples = rollout.generate(
            [trajectory['messages'] for trajectory in selected_trajectories],
            num_samples=wanted,
            sampling_params=asdict(params),
            prompt_extras=prompt_extras,
            force_logprobs=infer_config.save_rollout_tokens,
            **sample_kwargs)
        expected = len(to_sample) * wanted
        if len(samples) != expected:
            raise RuntimeError(f'rollout returned {len(samples)} samples, expected {expected}.')
        fresh = []
        for offset in range(0, expected, wanted):
            fresh.append([
                _candidate_from_sample(sample, multi_turn=multi_turn_enabled)
                for sample in samples[offset:offset + wanted]
            ])
        for index, group in zip(to_sample, fresh):
            candidates[index] = group
    if len(to_sample) < len(batch):
        logger.info(f'run_infer: reused cached candidates for {len(batch) - len(to_sample)}/{len(batch)} prompts')
    return trajectories, ground_truths, candidates


def _score_group(candidates: List['_Candidate'], row: Dict[str, Any], trajectory: Dict[str, Any],
                 ground_truth: Optional[str], channels: '_RewardChannels',
                 infer_config: InferConfig) -> Optional[List[float]]:
    """Score one prompt's candidates -> one float each, or None when no reward func is configured.

    Shared by both output formats so scoring happens once per group and is orthogonal to storage: a
    single-completion run is scored too. The ground truth, when scored, is appended so per-group
    normalisation sees it as the anchor, then trimmed back off -- it is never emitted as a candidate
    because the model did not produce it.
    """
    if channels.empty:
        return None
    scored = list(candidates)
    if infer_config.score_ground_truth and ground_truth:
        # trajectory['messages'] is the prompt with the reference turn already stripped (see
        # split_prompt_and_reference), so rebuilding prompt + ground_truth-as-assistant is the correct
        # single-turn conversation. InferComponent.__call__ rejects this in a multi-turn rollout, where
        # reference would have no tool trajectory to compare against the candidates.
        scored.append(_Candidate(ground_truth, messages=_candidate_messages(_Candidate(ground_truth), trajectory)))
    return channels.score(scored, row, trajectory, infer_config)[:len(candidates)]


def _emit_all(row: Dict[str, Any], trajectory: Dict[str, Any], ground_truth: Optional[str],
              candidates: List['_Candidate'], scores: Optional[List[float]], infer_config: InferConfig,
              recorder: Any = None) -> List[Dict[str, Any]]:
    """'all' format: one row per prompt carrying every candidate (the plain-inference / eval shape).

    ``response`` is the first candidate and ``responses`` all of them, the shape the acc/rouge metric and
    downstream eval read. When a reward func scored the group, ``scores`` rides along per candidate; with
    save_rollout_tokens, ``rollout_tokens`` is the per-candidate NPZ path list.
    """
    prompt_id = _prompt_key(trajectory['messages'])
    out: Dict[str, Any] = {key: value for key, value in row.items() if key != 'messages'}
    out['response'] = candidates[0].text
    out['responses'] = [candidate.text for candidate in candidates]
    out['messages'] = _candidate_messages(candidates[0], trajectory)
    if candidates[0].multi_turn:
        # Each multi-turn candidate is a distinct trajectory (its own tool calls); ``messages`` keeps only
        # the first, so store every candidate's full trajectory rather than losing all but one.
        out['all_messages'] = [_candidate_messages(candidate, trajectory) for candidate in candidates]
    if candidates[0].rollout_logprobs:
        # Populated only when logprobs were requested (generation_config.logprobs / save_rollout_tokens);
        # mirrors legacy's per-row ``logprobs`` for the response this row keeps.
        out['logprobs'] = list(candidates[0].rollout_logprobs)
    if ground_truth is not None:
        out['labels'] = ground_truth
    if scores is not None:
        # A judge that returned nothing numeric scores nan; null it so the row stays json-serialisable.
        out['scores'] = [None if score != score else score for score in scores]
    if recorder is not None:
        out['rollout_tokens'] = [
            recorder.record(prompt_id, index, candidate) for index, candidate in enumerate(candidates)
        ]
    return [out]


def _emit_dpo(row: Dict[str, Any], trajectory: Dict[str, Any], ground_truth: Optional[str],
              candidates: List['_Candidate'], scores: Optional[List[float]], infer_config: InferConfig,
              recorder: Any = None) -> List[Dict[str, Any]]:
    """'dpo' format: best-of-n chosen/rejected pairs for DPO training.

    Without any reward func (``scores is None``) every candidate is emitted as a positive with no rejected
    turn -- there is no ranking, so naming one candidate worse would be a fabrication. With scores,
    candidates at or below ``reward_threshold`` are dropped, an easy prompt (see ``easy_query_threshold``)
    is skipped whole, then the top ``n_best_to_keep`` become positives paired against the lowest scorer.
    """
    prompt_id = _prompt_key(trajectory['messages'])
    keep_tokens = recorder is not None
    record = recorder.record if keep_tokens else None
    if scores is None:
        rows = []
        for index, positive in enumerate(candidates):
            token_path = record(prompt_id, index, positive) if keep_tokens else None
            rows.append(_dpo_row(row, trajectory, positive, None, ground_truth, rollout_tokens=token_path))
        return rows

    threshold = infer_config.reward_threshold
    # An unscored candidate (nan, from a judge/API that returned nothing numeric) is dropped from the
    # ranking whole: it can be neither a positive nor the rejected response, and writing nan into a
    # reward_score would emit invalid JSON. ``scorable`` is the nan-free pool the ranking works on.
    scorable = [i for i in range(len(candidates)) if scores[i] == scores[i]]
    if not scorable:
        return []
    keep = [i for i in scorable if threshold is None or scores[i] > threshold]
    if not keep:
        return []
    if _is_too_easy(len(keep), len(candidates), infer_config.easy_query_threshold):
        return []
    ranked = sorted(scorable, key=lambda i: scores[i], reverse=True)
    negative_index = ranked[-1]
    positives = [i for i in ranked[:infer_config.n_best_to_keep] if i in keep and i != negative_index]
    if not positives:
        return []
    negative = candidates[negative_index]
    logger.debug(f'scores={[round(s, 4) for s in scores]} positives={positives} negative={negative_index}')
    rejected_tokens = record(prompt_id, negative_index, negative) if keep_tokens else None
    rows = []
    for i in positives:
        rows.append(
            _dpo_row(
                row,
                trajectory,
                candidates[i],
                negative,
                ground_truth,
                rollout_tokens=record(prompt_id, i, candidates[i]) if keep_tokens else None,
                reward_score=scores[i] if keep_tokens else None,
                rejected_rollout_tokens=rejected_tokens,
                rejected_reward_score=scores[negative_index] if keep_tokens else None))
    return rows


def _dpo_row(row: Dict[str, Any], trajectory: Dict[str, Any], positive: '_Candidate',
             negative: Optional['_Candidate'], ground_truth: Optional[str], *,
             rollout_tokens: Optional[str] = None, reward_score: Optional[float] = None,
             rejected_rollout_tokens: Optional[str] = None,
             rejected_reward_score: Optional[float] = None) -> Dict[str, Any]:
    """Build one DPO output row while preserving complete multi-turn chosen/rejected trajectories.

    The token/score fields are the save_rollout_tokens schema additions: each is written only when the
    recorder produced it (a local backend, tokens kept), so with the feature off the row is unchanged.
    """
    generated_keys = {
        'messages', 'rejected_messages', 'rejected_response', 'multi_turn', 'truncated', 'rollout_infos',
        'rejected_truncated', 'rejected_rollout_infos', 'id'
    }
    out: Dict[str, Any] = {key: value for key, value in row.items() if key not in generated_keys}
    out['messages'] = _candidate_messages(positive, trajectory)
    if negative is not None:
        if positive.multi_turn or negative.multi_turn:
            out['rejected_messages'] = _candidate_messages(negative, trajectory)
        else:
            out['rejected_response'] = negative.text
    if positive.multi_turn:
        out['multi_turn'] = True
        out['truncated'] = positive.truncated
        if positive.rollout_infos:
            out['rollout_infos'] = positive.rollout_infos
        if negative is not None:
            out['rejected_truncated'] = negative.truncated
            if negative.rollout_infos:
                out['rejected_rollout_infos'] = negative.rollout_infos
    if ground_truth is not None:
        out['ground_truth'] = ground_truth
    # Group id: every row from one prompt shares it, so positives can be traced back to their group.
    prompt_repr = json.dumps(trajectory['messages'], sort_keys=True, ensure_ascii=False, default=str)
    out['id'] = hashlib.md5(prompt_repr.encode('utf-8')).hexdigest()
    if rollout_tokens is not None:
        out['rollout_tokens'] = rollout_tokens
    if reward_score is not None:
        out['reward_score'] = reward_score
    if rejected_rollout_tokens is not None:
        out['rejected_rollout_tokens'] = rejected_rollout_tokens
    if rejected_reward_score is not None:
        out['rejected_reward_score'] = rejected_reward_score
    return out


def _emit_grpo(row: Dict[str, Any], trajectory: Dict[str, Any], ground_truth: Optional[str],
               candidates: List['_Candidate'], scores: Optional[List[float]], infer_config: InferConfig,
               recorder: Any = None) -> List[Dict[str, Any]]:
    """'grpo' format: one row per prompt holding the whole sampled group as an offline GRPO corpus.

    The row stores what GRPO training reads and nothing it recomputes. Every candidate keeps its full
    ``all_messages`` trajectory (multi-turn candidates diverge at their own tool calls, so each is stored
    whole rather than collapsed to one response) and its rollout tokens -- ``rollout_tokens``, the NPZ
    sidecar with input_ids / labels / completion_mask / logprobs / loss_mask (the old-policy logps) -- plus
    a stable group ``id`` and ``num_generations``. The group-relative advantage is deliberately NOT
    computed here: it is cheap and belongs to the training step, which recomputes it from the stored
    rewards and logps. When a reward channel (``--orm`` / ``--prm``) scored
    the group, each candidate's ``scores`` entry rides along; with no reward channel ``scores`` is null and
    only the trajectories + logps are stored, which is a valid GRPO corpus the trainer scores later.
    """
    prompt_id = _prompt_key(trajectory['messages'])
    out: Dict[str, Any] = {key: value for key, value in row.items() if key != 'messages'}
    out['id'] = prompt_id
    out['messages'] = copy.deepcopy(trajectory['messages'])
    out['completions'] = [candidate.text for candidate in candidates]
    out['all_messages'] = [_candidate_messages(candidate, trajectory) for candidate in candidates]
    out['num_generations'] = len(candidates)
    if ground_truth is not None:
        out['labels'] = ground_truth
    if recorder is not None:
        out['rollout_tokens'] = [
            recorder.record(prompt_id, index, candidate) for index, candidate in enumerate(candidates)
        ]
    if scores is not None:
        # A judge that returned nothing numeric scores nan; null it so the row stays json-serialisable.
        out['scores'] = [None if score != score else score for score in scores]
    else:
        out['scores'] = None
    return [out]


def _prompt_batch_size(batch_size: Optional[int]) -> int:
    """The prompt-source batch size for the per-sample driver, widened to the launch's dp width.

    The driver flattens the loader's batches straight back to rows (``StreamingDriver._iter_rows``), so
    the size is only the reader's fetch granularity -- but the twinkle loader splits every global batch
    across the dp ranks and asserts ``batch_size >= dp_width`` with divisibility, while
    ``infer_config.batch_size`` is the generate-batch size (default 1) and can be smaller than a
    multi-rank torchrun launch's world size. Round up to a multiple of the width; at width 1 the value
    passes through unchanged.
    """
    from twinkle.utils import Platform

    size = max(1, int(batch_size or 1))
    world = max(1, int(Platform.get_world_size()))
    if world > 1:
        size = -(-size // world) * world
    return size


def compute_metric(results: List[Dict[str, Any]], metric: Literal['acc', 'rouge']) -> Dict[str, float]:
    """Score the first completion of each row against its reference answer.

    ``acc`` is exact string equality, which is what legacy's ``--metric acc`` measured -- NOT the
    token-level ``twinkle.metric.Accuracy``, which scores logits against label ids and would report a
    different (and much higher) number for the same run.

    Only the first completion is scored: with ``num_samples > 1`` a best-of-n score is a different
    measurement (pass@n) and reporting it as accuracy would overstate the model. Rows without a
    reference are skipped rather than counted as wrong.
    """
    from twinkle.metric import ExactMatch, RougeBleu

    pairs = [(r['response'], r['labels']) for r in results if r.get('labels') is not None and r.get('response')]
    if not pairs:
        logger.warning('metric requested but no row has both a response and a reference answer.')
        return {}
    predictions, references = zip(*pairs)
    scorer = ExactMatch() if metric == 'acc' else RougeBleu()
    scorer.accumulate(predictions=list(predictions), references=list(references))
    return scorer.calculate()


def _initialize_for_sampling(distributed_config: DistributedConfig,
                            reward_groups: Optional[List[Tuple[str, int, int]]] = None) -> None:
    """Initialize twinkle, giving the sampler -- and each GPU-resident reward model -- its own DeviceGroup.

    A scalar seq_cls/reranker RM (built through ``build_frozen_reward_model`` -> ``build_model``) and an
    independent generative judge keep their weights on GPU, so under mode='ray' each is placed in its own
    disjoint DeviceGroup (see :func:`plan_sampling_device_groups`) rather than sharing the sampler's
    cards; ``build_model`` / ``build_sampler`` target the group by the same name via ``remote_group``.
    A sampler-reusing judge and an http(s) judge need no device and declare no group. Local mode has no
    groups to declare (and the caller forbids a GPU-resident reward there), so it initializes plainly.

    Each reward group is built with its OWN ``gpus_per_worker`` (from its ``parallel_spec``), so an
    independent judge told ``tensor_parallel_size=N`` sees N GPUs per worker instead of one -- the mesh's
    tp dim is real here, not dead info. The sampler group keeps ``gpus_per_worker=1``.
    """
    import twinkle
    from twinkle import DeviceGroup

    if distributed_config.mode != 'ray':
        # Local mode reaches here only with no reward groups: InferComponent.__call__ already rejects a
        # GPU-resident reward channel unless mode='ray' (there is no disjoint DeviceGroup to place it
        # in), so there is nothing left to guard here.
        twinkle.initialize(mode='local')
        return
    # DistributedConfig.nproc_per_node is the SAMPLER's GPU width (one role); twinkle's nproc_per_node
    # below is the TOTAL rank count across every group. Same spelling, two meanings -- the planner maps
    # the former to the latter.
    sampler_ranks = distributed_config.nproc_per_node
    if sampler_ranks is None:
        raise ValueError("DistributedConfig.nproc_per_node is required in mode='ray' (it sizes the sampler's "
                         'Ray DeviceGroup). Pass it explicitly -- there is no default.')
    planned, total_ranks = plan_sampling_device_groups(sampler_ranks, reward_groups or [])
    twinkle.initialize(
        mode='ray',
        nproc_per_node=total_ranks,
        groups=[
            DeviceGroup(name=name, ranks=ranks, device_type='GPU', gpus_per_worker=gpus_per_worker)
            for name, ranks, gpus_per_worker in planned
        ])


class _RolloutRecorder:
    """Writes one NPZ per kept candidate under ``token_dir``; a no-op when ``enabled`` is False.

    ``base_dir`` is the directory the jsonl lives in, so :meth:`record` returns a path relative to it --
    that relative path is what gets embedded in the row, keeping the jsonl portable alongside its sidecar.
    NPZ filenames are positionally deterministic (``{prompt_id}_c{candidate_index}.npz``), so replaying a
    batch on resume overwrites the same files instead of appending duplicates.
    """

    def __init__(self, token_dir: str, base_dir: str = '', enabled: bool = False):
        self.token_dir = token_dir
        self.base_dir = base_dir
        self.enabled = enabled
        if enabled:
            os.makedirs(token_dir, exist_ok=True)

    def record(self, prompt_id: str, candidate_index: int, candidate: Any) -> Optional[str]:
        """Persist one candidate's rollout tokens; return the NPZ path relative to ``base_dir``.

        The sidecar holds the token-level training payload every dump format shares (SFT/DPO/GRPO alike,
        not a GRPO special case): ``input_ids`` / ``labels`` / ``completion_mask`` (the trainable mask over
        the full sequence) / ``response_token_ids`` / ``response_loss_mask`` / ``rollout_logprobs`` (the
        sampling-policy old_logps). old_logps are the one thing training cannot recompute -- the policy has
        drifted -- so they must be stored; the rest pins the response boundary the logps align to.

        Multimodal tensors (pixel_values / image_grid_thw / ...) are deliberately NOT written: they are
        large and, unlike old_logps, fully reproducible. Each output row keeps the dataset's image/video
        columns and the candidate messages (with their placeholders), so training re-encodes the vision
        inputs from those -- the same re-encode an online rollout does -- and the stored token arrays line
        up because the template is deterministic for identical messages + images.

        Returns None when disabled, or when the candidate carries no token feature (a message-only
        backend), so a path is embedded only for candidates that actually have tokens. ``candidate`` is
        read by attribute (``encoded`` / ``response_token_ids`` / ``rollout_logprobs`` /
        ``response_loss_mask``), so both a ``_Candidate`` and a ``RolloutSample`` work.
        """
        if not self.enabled:
            return None
        encoded = getattr(candidate, 'encoded', None) or {}
        input_ids = encoded.get('input_ids')
        labels = encoded.get('labels')
        if not input_ids or not labels:
            return None
        arrays = {
            'input_ids': np.asarray(input_ids, dtype=np.int64),
            'labels': np.asarray(labels, dtype=np.int64),
            'response_token_ids': np.asarray(getattr(candidate, 'response_token_ids', None) or [], dtype=np.int64),
            'rollout_logprobs': np.asarray(getattr(candidate, 'rollout_logprobs', None) or [], dtype=np.float32),
            'response_loss_mask': np.asarray(getattr(candidate, 'response_loss_mask', None) or [], dtype=np.int8),
        }
        completion_mask = encoded.get('completion_mask')
        if completion_mask is not None:
            arrays['completion_mask'] = np.asarray(completion_mask, dtype=np.int8)
        name = f'{prompt_id}_c{candidate_index}.npz'
        abs_path = os.path.join(self.token_dir, name)
        np.savez_compressed(abs_path, **arrays)
        return os.path.relpath(abs_path, self.base_dir) if self.base_dir else name


def _build_channels(infer_config: InferConfig,
                    rl_config: Optional[RLConfig] = None,
                    *,
                    orm_slots: Optional[List['_RewardSlot']] = None,
                    prm_slots: Optional[List['_RewardSlot']] = None,
                    sampler: Any = None,
                    model_config: Optional[ModelConfig] = None,
                    template_config: Optional[TemplateConfig] = None,
                    backend: str = 'vllm',
                    engine_args: Optional[Dict[str, Any]] = None,
                    distributed_config: Optional[DistributedConfig] = None,
                    quantize_config: Optional[QuantizeConfig] = None) -> '_RewardChannels':
    """Build the ORM and PRM scoring channels from their resolved rule/model slots.

    Each channel's slots were resolved before the sampler (see :func:`_resolve_reward_channel`); here a
    rule slot becomes its reward callable and a model slot becomes its scorer -- a scalar seq_cls RM
    scored by forward, a generative judge (reusing the sampler or owning its own engine), or an API judge
    -- through :func:`_build_model_reward`. Slots are built in order so the channel weights, which align
    to the merged ``--orm`` / ``--prm`` list the user typed, line up with the funcs. A generative judge
    that reuses the sampler is why this runs AFTER the sampler is built.
    """
    ray = distributed_config is not None and distributed_config.mode == 'ray'
    orm_funcs, orm_names = _build_slot_funcs(
        orm_slots or [], 'orm', rl_config, infer_config, sampler, model_config, template_config, backend,
        engine_args, distributed_config, quantize_config, ray)
    prm_funcs, prm_names = _build_slot_funcs(
        prm_slots or [], 'prm', rl_config, infer_config, sampler, model_config, template_config, backend,
        engine_args, distributed_config, quantize_config, ray)
    # The channel selectors and their weights live on RLConfig (the shared surface GRPO also uses); the
    # rest of the synthesis knobs are on InferConfig. RLConfig doubles as the reward hyperparameter
    # carrier, so a rule plugin reads its own knobs (cosine_* / repetition_*) off it.
    orm_weights = rl_config.orm_weights if rl_config is not None else None
    prm_weights = rl_config.prm_weights if rl_config is not None else None
    channels = _RewardChannels(
        orm_funcs, orm_names, prm_funcs, prm_names, orm_weights=orm_weights, prm_weights=prm_weights)
    if channels.empty:
        logger.info('run_infer: no reward funcs -- every candidate is emitted as a positive.')
    else:
        logger.info(f'run_infer: orm={orm_names} (x{infer_config.orm_channel_weight}) prm={prm_names}, '
                    f'normalize={infer_config.normalize_rewards}')
    return channels


def _build_slot_funcs(slots: List['_RewardSlot'], group: str, rl_config: Optional[RLConfig],
                      infer_config: InferConfig, sampler: Any, model_config: Optional[ModelConfig],
                      template_config: Optional[TemplateConfig], backend: str,
                      engine_args: Optional[Dict[str, Any]], distributed_config: Optional[DistributedConfig],
                      quantize_config: Optional[QuantizeConfig], ray: bool) -> Tuple[List[Any], List[str]]:
    """Turn one channel's ordered slots into ``(funcs, names)``, preserving slot order.

    A rule slot resolves through the ``reward`` extension point (``cls(args=rl_config)``, so a plugin
    reads its own hyperparameters off the config); a model slot builds its scorer, targeting the channel's
    DeviceGroup only under ray and only for a GPU-resident kind (local mode has no groups; a sampler-reusing
    or API judge stays group-less).
    """
    from swift.dev.utils.plugin import PluginRegistry
    from swift.dev.rewards import REWARD

    funcs: List[Any] = []
    names: List[str] = []
    for slot in slots:
        if slot.is_model:
            spec = slot.spec
            remote_group = group if ray and spec.needs_device_group else None
            funcs.append(
                _build_model_reward(
                    spec,
                    judge_template=infer_config.judge_template,
                    sampler=sampler,
                    model_config=model_config,
                    template_config=template_config,
                    backend=backend,
                    engine_args=engine_args,
                    distributed_config=distributed_config,
                    quantize_config=quantize_config,
                    remote_group=remote_group))
            names.append(spec.display_name)
        else:
            func = PluginRegistry.resolve(REWARD, slot.rule, config=rl_config)
            funcs.append(func)
            names.append(PluginRegistry.display_name(func))
    return funcs, names



@dataclass
class _Candidate:
    """One generated candidate, optionally carrying its full trajectory and rollout tokens."""

    text: str
    messages: Optional[List[Dict[str, Any]]] = None
    rollout_infos: Optional[Dict[str, Any]] = None
    truncated: bool = False
    multi_turn: bool = False
    # Rollout token payload, empty for a message-only backend or when tokens were not kept. The recorder
    # (save_rollout_tokens) writes these to an NPZ sidecar; the jsonl row then carries the relative path.
    encoded: Optional[Dict[str, Any]] = None
    response_token_ids: List[int] = field(default_factory=list)
    rollout_logprobs: List[float] = field(default_factory=list)
    response_loss_mask: List[int] = field(default_factory=list)


def _candidate_from_sample(sample: Any, *, multi_turn: bool) -> _Candidate:
    """Adopt a RolloutSample's payload into a _Candidate (single- and multi-turn alike).

    Both generation paths now yield the same RolloutSample, so one adapter feeds _Candidate: the token
    half (encoded / response_token_ids / rollout_logprobs / response_loss_mask) is empty for a
    message-only backend and populated for a local one, which is what the recorder persists.
    """
    return _Candidate(
        sample.decoded,
        messages=copy.deepcopy(sample.messages) if sample.messages is not None else None,
        rollout_infos=copy.deepcopy(sample.rollout_infos) if sample.rollout_infos else None,
        truncated=sample.truncated,
        multi_turn=multi_turn,
        encoded=dict(sample.encoded) if sample.encoded else None,
        response_token_ids=list(sample.response_token_ids or []),
        rollout_logprobs=list(sample.rollout_logprobs or []),
        response_loss_mask=list(sample.response_loss_mask or []))


def _candidate_messages(candidate: _Candidate, trajectory: Dict[str, Any]) -> List[Dict[str, Any]]:
    if candidate.messages is not None:
        return copy.deepcopy(candidate.messages)
    return copy.deepcopy(trajectory['messages']) + [{'role': 'assistant', 'content': candidate.text}]


class _RewardChannels:
    """The ORM and PRM scoring channels, and how their two scores become one.

    Legacy kept them apart and combined them as ``prm + orm * 10``. That 10 was an unnamed priority --
    it made any ORM difference dominate every PRM difference. Here it is ``orm_channel_weight``, so the
    legacy ranking is reproducible by setting it to 10 and the default (1.0) is an honest sum.
    """

    def __init__(self,
                 orm_funcs: List[Any],
                 orm_names: List[str],
                 prm_funcs: List[Any],
                 prm_names: List[str],
                 orm_weights: Optional[List[float]] = None,
                 prm_weights: Optional[List[float]] = None):
        self.orm_funcs = orm_funcs
        self.orm_names = orm_names
        self.prm_funcs = prm_funcs
        self.prm_names = prm_names
        # Channel weights come from RLConfig.orm_weights / prm_weights (captured in _build_channels);
        # keeping them here avoids threading rl_config through every score call.
        self.orm_weights = orm_weights
        self.prm_weights = prm_weights

    @property
    def empty(self) -> bool:
        return not self.orm_funcs and not self.prm_funcs

    def score(self, candidates: List[_Candidate], row: Dict[str, Any], trajectory: Dict[str, Any],
              infer_config: InferConfig) -> List[float]:
        """Score one prompt's candidates -> one float each.

        Dataset columns are broadcast across the candidates. ``messages`` is different: each reward
        receives the complete trajectory that produced that candidate, including intermediate turns.
        """
        total = [0.0] * len(candidates)
        if self.orm_funcs:
            orm = self._channel(
                candidates, row, trajectory, self.orm_funcs, self.orm_weights, infer_config)
            total = [t + infer_config.orm_channel_weight * value for t, value in zip(total, orm)]
        if self.prm_funcs:
            prm = self._channel(
                candidates, row, trajectory, self.prm_funcs, self.prm_weights, infer_config)
            total = [t + value for t, value in zip(total, prm)]
        return total

    @staticmethod
    def _channel(candidates: List[_Candidate], row: Dict[str, Any], trajectory: Dict[str, Any], funcs: List[Any],
                 weights: Optional[List[float]], infer_config: InferConfig) -> List[float]:
        from swift.dev.rewards.reward import compute_rewards_per_func, weight_rewards

        columns = {key: [value] * len(candidates) for key, value in row.items() if key != 'messages'}
        columns['messages'] = [_candidate_messages(candidate, trajectory) for candidate in candidates]
        columns['rollout_infos'] = [copy.deepcopy(candidate.rollout_infos or {}) for candidate in candidates]
        columns['truncated'] = [candidate.truncated for candidate in candidates]
        rewards_per_func = compute_rewards_per_func([candidate.text for candidate in candidates], funcs, columns)
        scores = weight_rewards(rewards_per_func, weights).tolist()
        # Normalise per channel, before the channels are added: doing it after would let the channel
        # with the larger raw range decide the ranking regardless of the weights.
        return _normalize(scores) if infer_config.normalize_rewards else scores

    def shutdown(self) -> None:
        for func in self.orm_funcs + self.prm_funcs:
            shutdown = getattr(func, 'shutdown', None)
            if shutdown is not None:
                shutdown()


class _CandidateCache:
    """Candidates from earlier runs, keyed by prompt, i.e. legacy's ``cache_files``.

    Keyed on the prompt's messages rather than on row order, because the cache files come from other
    runs whose dataset order and slicing need not match this one's. A prompt is only served from cache
    when it has at least as many candidates as this run asks for -- fewer would silently shrink the
    group that the reward ranking then works on.
    """

    def __init__(self, cache_files: List[str]):
        self.by_prompt: Dict[str, List[_Candidate]] = {}
        for path in cache_files or []:
            if not os.path.isfile(path):
                logger.warning(f'cache_files entry {path} does not exist; ignoring it.')
                continue
            self._load(path)
        if self.by_prompt:
            logger.info(f'run_infer: cache covers {len(self.by_prompt)} prompts from {len(cache_files)} file(s)')

    ## 这里有问题吧？为什么不使用数据湖？
    def _load(self, path: str) -> None:
        with open(path, encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    # A crashed producer leaves a truncated final line; skipping it beats refusing the
                    # whole cache file.
                    logger.warning(f'skipping an unparseable line in {path}')
                    continue
                messages = entry.get('messages')
                if not messages or messages[-1].get('role') != 'assistant':
                    continue
                multi_turn = bool(entry.get('multi_turn'))
                if multi_turn:
                    key = entry.get('id')
                    if not key:
                        continue
                else:
                    key = _prompt_key(messages[:-1])
                self.by_prompt.setdefault(key, []).append(
                    _Candidate(
                        messages[-1].get('content') or '',
                        messages=copy.deepcopy(messages) if multi_turn else None,
                        rollout_infos=copy.deepcopy(entry.get('rollout_infos')),
                        truncated=bool(entry.get('truncated')),
                        multi_turn=multi_turn))

    def lookup(self, trajectory: Dict[str, Any], wanted: int, *, multi_turn: bool = False) -> Optional[List[_Candidate]]:
        if not self.by_prompt:
            return None
        cached = self.by_prompt.get(_prompt_key(trajectory['messages']))
        if cached is None:
            return None
        if multi_turn:
            cached = [candidate for candidate in cached if candidate.multi_turn and candidate.messages is not None]
        else:
            cached = [_Candidate(candidate.text) for candidate in cached]
        return cached[:wanted] if len(cached) >= wanted else None


def _prompt_key(messages: List[Dict[str, Any]]) -> str:
    return hashlib.md5(
        json.dumps(messages, sort_keys=True, ensure_ascii=False, default=str).encode('utf-8')).hexdigest()


def _is_too_easy(n_passing: int, n_candidates: int, easy_query_threshold: Optional[float]) -> bool:
    """Whether a prompt is too easy to be worth keeping.

    A prompt almost every candidate already answers correctly carries no learning signal, and keeping
    it biases the dataset toward what the model can do. None disables the filter.
    """
    if easy_query_threshold is None:
        return False
    return n_passing / n_candidates >= easy_query_threshold


def _plan_batches(n_rows: int, batch_size: int, max_batches: Optional[int]) -> List[Tuple[int, int]]:
    """Fixed ``(start, end)`` slices, computed up front so the resume index means the same thing
    across runs. A trailing partial batch is kept, unlike legacy's ``n // batch_size`` truncation."""
    if batch_size < 1:
        raise ValueError(f'InferConfig.batch_size must be >= 1, got {batch_size}.')
    batches = [(start, min(start + batch_size, n_rows)) for start in range(0, n_rows, batch_size)]
    return batches[:max_batches] if max_batches else batches


#: How many top logprobs to request when scoring a generative reranker off generation. The positive and
#: negative tokens must both land in this window or the score cannot be read; 20 is within the default
#: ``max_logprobs`` of both vLLM and sglang and comfortably covers the two tokens a trained reranker
#: puts its mass on.
_GENERATIVE_RERANKER_TOP_LOGPROBS = 20


def _initialize_for_forward(distributed_config: DistributedConfig) -> None:
    """Initialize twinkle for a forward-pass run: the 'model' DeviceGroup under ray, local otherwise.

    Both forward paths reach the model through twinkle -- ``build_sampler`` for the vLLM/SGLang pooling
    head and ``build_model`` for the transformers forward -- and ray placement targets the DeviceGroup
    named 'model' (``build_model``'s ``remote_group`` default), so the group has to exist before either
    is built; a local (torchrun) run has no group to declare. Infer-side twin of the training
    assembly's bootstrap, minus the auxiliary groups (those are frozen TRAINING models).
    """
    import twinkle
    from twinkle import DeviceGroup

    extra_torch_kwargs = {'seed': 42, 'full_determinism': False, 'tf32': None}
    if distributed_config.mode != 'ray':
        twinkle.initialize(mode='local', extra_torch_kwargs=extra_torch_kwargs)
        return
    nproc = distributed_config.nproc_per_node
    if nproc is None:
        raise ValueError("DistributedConfig.nproc_per_node is required in mode='ray' (it sizes the Ray "
                         "'model' DeviceGroup). Pass it explicitly -- there is no default.")
    twinkle.initialize(
        mode='ray',
        nproc_per_node=nproc,
        extra_torch_kwargs=extra_torch_kwargs,
        name=distributed_config.ray_exp_name,
        groups=[DeviceGroup(name='model', ranks=list(range(nproc)), device_type='GPU', gpus_per_worker=1)])


def _forward_rows(dataset_config: DatasetConfig, max_rows: Optional[int], split_dataset_ratio: float,
                  distributed_config: DistributedConfig) -> List[Dict[str, Any]]:
    """Initialize twinkle for a forward pass, then load the prompt rows it runs over.

    Shared by the pooling and generative-reranker forwards: they differ in the forward they run, not in
    the bootstrap or the input; the rows load AFTER the init so a ray-placed model resolves its DeviceGroup.
    """
    from swift.dev.builders import load_prompt_only_rows

    _initialize_for_forward(distributed_config)
    rows = load_prompt_only_rows(dataset_config, max_rows, split_dataset_ratio)
    if not rows:
        raise ValueError('run_infer got an empty dataset. Set DatasetConfig.dataset or .val_dataset.')
    return rows


def _finalize_forward_results(
    rows: List[Dict[str, Any]],
    per_row: List[Any],
    task_type: Optional[str],
    backend: str,
    output_path: Optional[str],
    metric: Optional[str],
) -> List[Dict[str, Any]]:
    """Assemble, write and score the one-value-per-row output of a forward-pass path.

    Shared by the two forward task families :meth:`InferComponent._forward_pass` dispatches -- a pooling
    task and a ``generative_reranker``: both produce exactly one plain value per row (a vector, class
    logits, or a relevance score), so the row shape is the same. The label comes from the row's own
    ``label`` column rather than a trailing assistant turn, because there is no completion to strip.
    """
    logger.info(f'run_infer: {len(rows)} rows, task_type={task_type}, backend={backend}')
    results = []
    for row, output in zip(rows, per_row):
        passthrough = {key: value for key, value in row.items() if key != 'messages'}
        results.append({
            'response': output,
            'responses': [output],
            'labels': row.get('label'),
            'messages': list(row['messages']),
            **passthrough
        })
    if output_path:
        _write_jsonl(output_path, results)
        logger.info(f'run_infer: wrote {len(results)} rows to {output_path}')
    if metric:
        logger.info(f'run_infer metric: {compute_metric(results, metric)}')
    return results


def _sigmoid(score: float) -> float:
    """Numerically stable sigmoid, mapping a reranker's raw score into a [0, 1] relevance."""
    if score >= 0:
        return 1.0 / (1.0 + math.exp(-score))
    exp = math.exp(score)
    return exp / (1.0 + exp)


def _forward_sampler(
    model_config: ModelConfig,
    template_config: TemplateConfig,
    distributed_config: Optional[DistributedConfig],
    adapters: Optional[List[str]],
    quantize_config: Optional[QuantizeConfig],
    backend: str,
    engine_args: Optional[Dict[str, Any]],
    shutdown: bool,
):
    """Build a vLLM/SGLang sampler for a forward-pass path, and shut it down afterwards.

    Shared by the pooling (``encode``) and generative-reranker (``sample``) sampler paths: they differ
    only in the call they make, while the model/template/mesh build, the single-adapter selection and the
    teardown are identical. Yields ``(sampler, processor, adapter_path)``.
    """
    from swift.dev.builders import build_parallel_mesh, build_sampler, build_template, load_model_processor

    _, processor = load_model_processor(model_config)
    template = build_template(template_config, processor)
    sampler = build_sampler(
        model_config,
        backend=backend,
        engine_args=engine_args,
        device_mesh=build_parallel_mesh(distributed_config, none_if_single_rank=True),
        template=template,
        adapters=adapters,
        quantize_config=quantize_config)
    try:
        yield sampler, processor, (adapters[0] if adapters else None)
    finally:
        if shutdown:
            sampler.shutdown()


def _pooling_via_sampler(
    model_config: ModelConfig,
    template_config: TemplateConfig,
    distributed_config: Optional[DistributedConfig],
    trajectories: List[Dict[str, Any]],
    adapters: Optional[List[str]],
    quantize_config: Optional[QuantizeConfig],
    backend: str,
    engine_args: Optional[Dict[str, Any]],
    shutdown: bool,
    reranker_use_activation: bool = True,
) -> List[Any]:
    """Run the pooling forward on a vLLM/SGLang sampler's ``encode``, one plain value per row."""
    from swift.dev.builders import pooled_data, to_pooling_params

    with _forward_sampler(model_config, template_config, distributed_config, adapters, quantize_config,
                          backend, engine_args, shutdown) as (sampler, _processor, adapter_path):
        # A cross-encoder reranker's raw score is a logit; ``use_activation`` has twinkle sigmoid it into a
        # [0, 1] relevance comparable across queries (legacy reranker_use_activation). It rides in as a
        # PoolingParams override, the channel to_pooling_params documents for these post-processing knobs.
        overrides = {'use_activation': reranker_use_activation} if model_config.task_type == 'reranker' else {}
        pooling_params = to_pooling_params(model_config.task_type, **overrides)
        kwargs: Dict[str, Any] = {}
        if adapter_path is not None:
            kwargs['adapter_path'] = adapter_path
        return pooled_data(sampler.encode(trajectories, pooling_params, **kwargs))


def _forward_via_transformers(
    model_config: ModelConfig,
    template_config: TemplateConfig,
    distributed_config: Optional[DistributedConfig],
    rows: List[Dict[str, Any]],
    trajectories: List[Dict[str, Any]],
    adapters: Optional[List[str]],
    quantize_config: Optional[QuantizeConfig],
    max_batch_size: Optional[int] = None,
    reranker_use_activation: bool = True,
) -> List[Any]:
    """Run one HF ``forward_only`` pass (transformers backend), one plain value per row.

    Serves both the pooling tasks (seq_cls / embedding / reranker) and the transformers
    ``generative_reranker`` path, which reads its yes/no logit difference out of the raw logits
    returned here. Requests are chunked by ``max_batch_size`` so a large dataset does not materialise
    every encoded input and every output tensor at once.
    """
    from swift.dev.builders import build_model, build_template, load_model_processor
    from swift.dev.config import DistributedConfig

    task_type = model_config.task_type
    _, processor = load_model_processor(model_config)
    template = build_template(template_config, processor)
    # build_template leaves the template in 'train' mode (its default caller is the dataset/training
    # path). This is the INFERENCE forward, and a train-mode ``_embedding_encode`` / ``_reranker_encode``
    # reads ``positive_messages`` / ``negative_messages`` off the row -- pairs that only exist for
    # contrastive training. run_infer's ``to_trajectory`` carries just the anchor ``messages``, so train
    # mode hit ``positive[0]`` on an empty list and raised IndexError. Switching to an inference mode
    # makes both encoders take their anchor-only branch, which is what a forward pass wants. causal_lm /
    # seq_cls encode the anchor identically in either mode, so this does not change them.
    template.set_mode('transformers')
    model = build_model(
        model_config, distributed_config or DistributedConfig(), quantize_config=quantize_config)
    if adapters:
        # add_adapter_to_model takes a checkpoint directory as well as a PeftConfig, so a trained
        # adapter is loaded here rather than built -- apply_tuner would create a fresh, untrained one.
        for index, adapter in enumerate(adapters):
            model.add_adapter_to_model(f'adapter_{index}' if index else 'default', adapter)

    # forward_only resolves a generative_reranker's yes/no token ids off the model's template, so it must
    # be attached even though the inputs are already encoded here (the pooling tasks ignore it). Without
    # this the transformers generative-reranker path trips twinkle's "call set_template before forwarding".
    model.set_template(template)

    batch_size = max_batch_size or len(rows) or 1
    per_row: List[Any] = []
    for start in range(0, len(rows), batch_size):
        chunk = trajectories[start:start + batch_size]
        encoded = [template.encode(trajectory) for trajectory in chunk]
        outputs = model.forward_only(inputs=encoded, task=task_type, return_logits=True)
        per_row.extend(_per_row_outputs(outputs, len(chunk)))
    if reranker_use_activation and task_type in ('reranker', 'generative_reranker'):
        # forward_only returns the raw relevance logit (a cross-encoder score, or a generative reranker's
        # yes/no logit difference); sigmoid maps it to [0, 1], matching legacy's activation and the
        # vLLM/SGLang paths. seq_cls / embedding are left untouched.
        per_row = [
            [_sigmoid(value) for value in row] if isinstance(row, (list, tuple)) else _sigmoid(row)
            for row in per_row
        ]
    return per_row


def _per_row_outputs(outputs: Any, num_rows: int) -> List[Any]:
    """Unpack ``forward_only``'s output into one plain-Python value per row.

    The shape depends on the head: seq_cls gives logits per row, embedding gives a vector. Both are
    returned as lists so the result is jsonl-serialisable, which is the whole point of this path.
    """
    tensor = outputs
    if isinstance(outputs, dict):
        # forward_only(task='embedding') returns the pooled vectors under 'embeddings' (plural); the
        # other heads use 'logits' / 'embedding' / 'last_hidden_state'. Missing the plural key left the
        # raw output dict (tensors and all) as the row value, which is neither a vector nor serialisable.
        for key in ('logits', 'embeddings', 'embedding', 'last_hidden_state'):
            if key in outputs:
                tensor = outputs[key]
                break
    if hasattr(tensor, 'tolist'):
        listed = tensor.tolist()
        if isinstance(listed, list) and len(listed) == num_rows:
            return listed
        return [listed]
    return list(tensor) if isinstance(tensor, (list, tuple)) else [tensor] * num_rows


def _generative_reranker_via_sampler(
    model_config: ModelConfig,
    template_config: TemplateConfig,
    generation_config: Optional[GenerationConfig],
    distributed_config: Optional[DistributedConfig],
    trajectories: List[Dict[str, Any]],
    adapters: Optional[List[str]],
    quantize_config: Optional[QuantizeConfig],
    backend: str,
    engine_args: Optional[Dict[str, Any]],
    shutdown: bool,
    reranker_use_activation: bool = True,
) -> List[float]:
    """Score a generative reranker on a vLLM/SGLang generation engine, one score per row."""
    from swift.dev.builders import to_sampling_params

    # Built as a generation engine, not a pooling runner: build_sampler treats ``generative_reranker`` as
    # a generative task_type, so no ``runner='pooling'``/``is_embedding`` is injected.
    with _forward_sampler(model_config, template_config, distributed_config, adapters, quantize_config,
                          backend, engine_args, shutdown) as (sampler, processor, adapter_path):
        tokenizer = _tokenizer_of(processor)
        positive_id, negative_id = _generative_reranker_token_ids(tokenizer)
        # temperature is forced to 1.0 so the reported logprobs are the log-softmax of the raw logits:
        # only then does logprob(yes) - logprob(no) equal the logit difference the HF path returns (any
        # other temperature scales the logits before the softmax and changes the gap). max_tokens=1 --
        # not 0 -- because twinkle's max_tokens==0 logprobs-only path drops logprobs, and we need the
        # first generated token's.
        params = to_sampling_params(
            generation_config, max_tokens=1, temperature=1.0, logprobs=_GENERATIVE_RERANKER_TOP_LOGPROBS)
        kwargs: Dict[str, Any] = {}
        if adapter_path is not None:
            kwargs['adapter_path'] = adapter_path
        responses = sampler.sample(trajectories, params, **kwargs)
        return [
            _score_from_logprobs(response, positive_id, negative_id, reranker_use_activation)
            for response in responses
        ]


def _score_from_logprobs(response: Any, positive_id: int, negative_id: int,
                         use_activation: bool = True) -> float:
    """``logprob(yes) - logprob(no)`` at the first generated position of one SampleResponse.

    With ``use_activation`` the log-ratio is squashed by a sigmoid into ``P(yes) / (P(yes) + P(no))``, a
    [0, 1] relevance comparable across queries -- the same activation legacy applies to a generative
    reranker, and the counterpart of the pooling path's ``PoolingParams.use_activation``.
    """
    sequence = response.sequences[0]
    if not sequence.logprobs:
        raise RuntimeError('generative reranker scoring requested logprobs but the sampler returned none; '
                           'the backend may not honour logprobs on this path.')
    topk = dict(sequence.logprobs[0])
    if positive_id not in topk or negative_id not in topk:
        raise RuntimeError(
            f'generative reranker: the positive/negative tokens ({positive_id}/{negative_id}) are not both '
            f'in the top-{len(topk)} logprobs of the first generated token. Raise '
            '_GENERATIVE_RERANKER_TOP_LOGPROBS (or check GENERATIVE_RERANKER_POSITIVE_TOKEN/'
            'GENERATIVE_RERANKER_NEGATIVE_TOKEN) so both are returned.')
    score = topk[positive_id] - topk[negative_id]
    return _sigmoid(score) if use_activation else score


def _generative_reranker_token_ids(tokenizer: Any) -> Tuple[int, int]:
    """The (positive, negative) token ids, from the same env vars the HF scoring path reads."""
    positive_token = os.environ.get('GENERATIVE_RERANKER_POSITIVE_TOKEN', 'yes')
    negative_token = os.environ.get('GENERATIVE_RERANKER_NEGATIVE_TOKEN', 'no')
    return tokenizer.convert_tokens_to_ids(positive_token), tokenizer.convert_tokens_to_ids(negative_token)


def _tokenizer_of(processor: Any) -> Any:
    """A tokenizer from whatever ``load_model_processor`` returned (a tokenizer, or a processor wrapping one)."""
    return getattr(processor, 'tokenizer', processor)
