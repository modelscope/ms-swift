# Copyright (c) ModelScope Contributors. All rights reserved.
# The infer entry point: a sampling path and a forward-pass path, split inside __call__.
from __future__ import annotations
import copy
import hashlib
import json
import math
import os
import time
from dataclasses import asdict
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, Tuple

import numpy as np

from swift.dev.utils.logger import get_logger
from swift.dev.utils.scoring import collect
from swift.dev.components.base import RecipeComponent

if TYPE_CHECKING:
    from swift.dev.config import (
        DatasetConfig,
        DistributedConfig,
        GenerationConfig,
        InferConfig,
        ModelConfig,
        QuantizeConfig,
        RewardConfig,
        RolloutConfig,
        TemplateConfig,
    )

logger = get_logger()

#: The DeviceGroup name the sampling engine occupies under ray (matches the training assembly).
SAMPLER_GROUP = 'sampler'

#: The prompt-row column carrying each driven prompt's row index (it rides the trajectory back).
_ROW_INDEX = '_infer_row_index'

#: How many top logprobs to request when scoring a generative reranker off generation. The positive
#: and negative tokens must both land in this window or the score cannot be read; 20 is within the
#: default ``max_logprobs`` of both vLLM and sglang and comfortably covers the two tokens a trained
#: reranker puts its mass on.
_GENERATIVE_RERANKER_TOP_LOGPROBS = 20


class InferComponent(RecipeComponent):
    """The infer entry point: a sampling path and a forward-pass path, split inside ``__call__``.

    A causal LM samples a candidate group per prompt, optionally scores it, then emits rows through
    the ``output_format`` strategy -- so a new storage shape is a new emit function, not a new
    recipe. A pooling task (``seq_cls`` / ``embedding`` / ``reranker``) or a ``generative_reranker``
    never samples: one plain forward pass over the same rows, one value per row.
    """

    def __init__(
        self,
        model_config: ModelConfig,
        template_config: TemplateConfig,
        dataset_config: DatasetConfig,
        infer_config: InferConfig,
        generation_config: Optional[GenerationConfig],
        *,
        reward_config: Optional[RewardConfig],
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
        self.reward_config = reward_config
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
        from swift.dev.builders import (
            apply_generation_baseline,
            build_parallel_mesh,
            build_sampler,
            build_template,
            is_pooling_task,
            load_model_processor,
            load_prompt_only_rows,
            to_sampling_params,
        )
        from swift.dev.dataset.storage import store_output_complete, store_writer_kind

        # transformers>=5.2 pulls kernel repos from the Hub during a kernelized load; route them
        # through ModelScope (USE_HF=1 keeps the original HF source). Same wiring as legacy's
        # TransformersEngine.__init__.
        from swift.dev.patch.transformers_hub_kernels import TransformersHubKernelPatch
        from twinkle.patch import apply_patch

        apply_patch(None, TransformersHubKernelPatch())

        model_config = self.model_config
        infer_config = self.infer_config
        reward_config = self.reward_config
        distributed_config = self.distributed_config

        # A pooling task or a generative_reranker never samples: one plain forward pass, one value
        # per row (see _forward_pass). Split before the sampling-side validation below, which reads
        # knobs (num_return_sequences, output_format) a forward pass never uses.
        task_type = model_config.task_type or 'causal_lm'
        if is_pooling_task(task_type) or task_type in ("generative_reranker", "decision"):
            return self._forward_pass(task_type)

        num_return = infer_config.num_return_sequences
        output_format = infer_config.output_format
        multi_turn_enabled = bool(self.rollout_config and self.rollout_config.max_turns is not None)
        self._validate_sampling(num_return, output_format, multi_turn_enabled)
        if self.backend == 'twinkle' and (distributed_config is None or distributed_config.mode != 'remote'):
            raise ValueError(
                'sampler="twinkle" (the client-side sampler over a twinkle-server) requires '
                "--mode remote and --server_base_url: without it there is no transport to the "
                'served sampler. The local backends are vllm/sglang/transformers.')
        # The model's own generation_config.json is the sampling baseline: fill knobs the caller
        # left unset from it, so a model's declared temperature/top_k survives (legacy parity).
        apply_generation_baseline(self.generation_config, model_config.model)

        # Resolve the reward channels BEFORE twinkle init and the sampler build: a channel's model
        # spec decides whether an RM needs its own GPU DeviceGroup (declared at init) and whether a
        # generative judge reusing the sampler needs its reward LoRA in the engine's adapter list.
        orm_plan = _resolve_channel(reward_config, 'orm', model_config.model, self.backend)
        prm_plan = _resolve_channel(reward_config, 'prm', model_config.model, self.backend)
        reuse_adapters = [spec.adapter for plan in (orm_plan, prm_plan) for spec in plan.specs
                          if spec.kind == 'generative_reuse' and spec.adapter]
        reward_groups = _reward_device_groups(orm_plan, prm_plan, distributed_config)
        if reward_groups and (distributed_config is None or distributed_config.mode != 'ray'):
            raise ValueError(
                f'reward channel(s) {[name for name, _, _ in reward_groups]} keep their own model '
                "resident on GPU and need a dedicated DeviceGroup, which only "
                "DistributedConfig.mode='ray' can place. Run with mode='ray' (and nproc_per_node), "
                'or use a sampler-reusing generative judge / an http(s) endpoint, which need no '
                'extra device.')
        _initialize_for_sampling(distributed_config, reward_groups)

        rows = load_prompt_only_rows(self.dataset_config, self.max_rows, self.split_dataset_ratio)
        if not rows:
            raise ValueError('run_infer got an empty dataset. Set DatasetConfig.dataset or .val_dataset.')
        logger.info(f'run_infer: {len(rows)} prompts, backend={self.backend}, '
                    f'num_return_sequences={num_return}, output_format={output_format}')

        # The storage layer owns the writer policy: a 'dpo' or resumed run gets the checkpointed
        # writer when the format honours it, a plain 'all' run the incremental writer (which also
        # gathers across DP ranks), a non-incremental format one write at the end.
        writer_kind = store_writer_kind(
            self.output_path, infer_config.store_format,
            checkpoint=(output_format == 'dpo' or infer_config.resume))
        if writer_kind == 'batch' and output_format == 'dpo':
            logger.warning(f"output_format='dpo' with store_format={infer_config.store_format!r} writes one "
                           'store at the end, so it cannot checkpoint/resume a long best-of-n run; use '
                           '--store_format jsonl for that.')
        if writer_kind == 'checkpoint':
            from twinkle.utils import Platform

            local_world = max(1, int(Platform.get_world_size()))
            if local_world > 1 and (distributed_config is None or distributed_config.mode != 'ray'):
                # The checkpoint writer owns one tmp file and one state file and gathers nothing:
                # under a multi-rank torchrun launch every rank would open the same tmp at its own
                # offset and corrupt it, so reject the multi-rank local launch up front.
                raise ValueError(
                    f'a checkpointed run (output_format={output_format!r}, resume={infer_config.resume}) '
                    f'needs a single driving process, but this launch has {local_world} local ranks, each '
                    "of which would write the same checkpoint files. Run it under DistributedConfig.mode="
                    "'ray', or with a single rank.")
            if store_output_complete(self.output_path) and not infer_config.override_exist_file:
                logger.info(f'run_infer: {self.output_path} exists and override_exist_file is False; '
                            'nothing to do.')
                return []

        cache = _CandidateCache(infer_config.cache_files)
        quant_method = getattr(self.quantize_config, 'quant_method', None)
        if self.backend == 'no' and quant_method is not None:
            raise ValueError(
                f'quant_method={quant_method!r} cannot affect sampler={self.backend!r}, which loads no '
                'local model. Configure quantization on the remote server or omit --quant_method.')

        template = None
        adapter_path = self.adapters[0] if self.adapters else None
        if self.backend == 'no':
            sampler = None
        else:
            _, processor = load_model_processor(model_config)
            template = build_template(self.template_config, processor)
            # A generative judge reusing this sampler scores through a reward LoRA, which must be
            # resident before the first request, so merge it into the sampling adapters the engine
            # is built with.
            sampler_adapters = list(self.adapters or [])
            for reward_adapter in reuse_adapters:
                if reward_adapter not in sampler_adapters:
                    sampler_adapters.append(reward_adapter)
            sampler = build_sampler(
                model_config, backend=self.backend, engine_args=self.engine_args,
                device_mesh=build_parallel_mesh(distributed_config, none_if_single_rank=True),
                template=template, adapters=sampler_adapters or None,
                remote_group=(SAMPLER_GROUP if distributed_config is not None
                              and distributed_config.mode == 'ray' else None),
                quantize_config=self.quantize_config)

        orm_funcs, prm_funcs = _build_scorers(
            infer_config, reward_config, orm_plan, prm_plan, sampler=sampler, model_config=model_config,
            template_config=self.template_config, backend=self.backend, engine_args=self.engine_args,
            distributed_config=distributed_config, quantize_config=self.quantize_config)
        # Every scorer yields one column of the matrix both channels land in; the channel arithmetic
        # stays on this side (see _combined).
        from swift.dev.utils.scoring import build_reward_weights

        self._scorers = [*orm_funcs, *prm_funcs]
        self._orm_count = len(orm_funcs)
        self._orm_weights = build_reward_weights(
            getattr(reward_config, 'orm_weights', None), len(orm_funcs)).tolist()
        self._prm_weights = build_reward_weights(
            getattr(reward_config, 'prm_weights', None), len(prm_funcs)).tolist()

        # One rollout for every generation entry: single-turn is the one-turn degenerate. Tools are
        # opt-in through RolloutConfig.tools; infer drives the rollout in this process.
        rollout = None
        if sampler is not None:
            from swift.dev.builders.streaming import build_rollout
            from swift.dev.config import RolloutConfig

            rollout = build_rollout(
                sampler, template, self.rollout_config or RolloutConfig(),
                sampling_params=asdict(to_sampling_params(self.generation_config)), adapter_path=adapter_path)

        rows = _plan_rows(rows, infer_config)
        from swift.dev.dataset.storage import open_store_writer

        writer = open_store_writer(
            self.output_path, infer_config.store_format, store_fields=infer_config.store_fields,
            checkpoint=(output_format == 'dpo' or infer_config.resume), resume=infer_config.resume)
        # Rollout-token sidecar: only built when save_rollout_tokens is on. The NPZ dir sits beside
        # the jsonl and each row embeds a path relative to output_dir, so the pair stays portable.
        recorder = None
        if infer_config.save_rollout_tokens:
            if not self.output_path:
                raise ValueError('save_rollout_tokens needs an output_path to place the NPZ sidecar beside.')
            recorder = _RolloutRecorder(
                self.output_path + '.rollout_tokens', base_dir=os.path.dirname(self.output_path) or 'output',
                enabled=True)

        if output_format == 'grpo' and not self._scorers:
            # No reward channel is a valid GRPO dump -- the group's trajectories, masks and
            # old_logps are the corpus, and training scores/advantages them later.
            logger.info("output_format='grpo' with no reward channel (--orm / --prm) stores each prompt's "
                        'group trajectories + rollout logps with null scores; reward is left to the '
                        'training step.')

        # Throughput/token summary, the dev counterpart of legacy's InferStats: prompts and
        # generated tokens are counted as groups complete, then reported once at the end.
        start_time = time.perf_counter()
        emit = {'dpo': _emit_dpo, 'grpo': _emit_grpo}.get(output_format, _emit_all)
        try:
            results, num_prompts, num_generated_tokens = self._drive(
                rows=rows, rollout=rollout, cache=cache, writer=writer, recorder=recorder, emit=emit,
                num_return=num_return, multi_turn_enabled=multi_turn_enabled)
        finally:
            for scorer in self._scorers:
                shutdown = getattr(scorer, 'shutdown', None)
                if shutdown is not None:
                    shutdown()
            if rollout is not None:
                rollout.close()
            if self.shutdown and sampler is not None:
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
                logger.warning("metric=%r applies only to output_format='all' rows, which carry one "
                               'response/labels to score; %r rows store a whole candidate group, so the '
                               "metric is skipped. Re-run with output_format='all' for a metric.",
                               infer_config.metric, output_format)
            else:
                logger.info(f'run_infer metric: {compute_metric(results, infer_config.metric)}')
        return results

    def _validate_sampling(self, num_return: int, output_format: str, multi_turn_enabled: bool) -> None:
        """The sampling-path knob combinations that cannot be honoured, rejected up front."""
        infer_config = self.infer_config
        if output_format == 'dpo':
            if num_return < 2:
                raise ValueError(f"output_format='dpo' needs num_return_sequences >= 2 to have both a "
                                 f'chosen and a rejected candidate, got {num_return}.')
            if infer_config.n_best_to_keep >= num_return:
                raise ValueError(f'n_best_to_keep={infer_config.n_best_to_keep} must be < '
                                 f'num_return_sequences={num_return}: the lowest-scoring candidate becomes '
                                 'the rejected response, so it cannot also be a positive.')
        elif output_format == 'grpo':
            # A GRPO dump is the group itself (trajectories, masks, rollout logps); the advantage
            # is recomputed at train time. save_rollout_tokens forces the logprob computation --
            # without it there are no old_logps to store.
            if num_return < 2:
                raise ValueError(f"output_format='grpo' needs num_return_sequences >= 2 to dump a group, "
                                 f'got {num_return}.')
            if not infer_config.save_rollout_tokens:
                raise ValueError("output_format='grpo' stores each candidate's rollout logprobs "
                                 '(old_logps), which are only produced when save_rollout_tokens forces '
                                 'logprob computation. Re-run with --save_rollout_tokens true.')
        if infer_config.score_ground_truth and multi_turn_enabled:
            raise ValueError(
                'score_ground_truth cannot combine with a multi-turn rollout: the reference answer has '
                "no tool trajectory to score against the candidates'. Turn off score_ground_truth, or "
                'run single-turn.')
        if infer_config.generation_strict is False:
            # Honoured nowhere: TransformersSampler.sample has a per-row strict tolerance, but the
            # rollout does not thread it through, so the knob is unreachable from config.
            logger.warning('generation_strict=False has no effect on this path: the rollout engine '
                           'does not thread the sampler-side per-row strictness through generate, '
                           'so it cannot be reached from config. The flag is accepted only for CLI '
                           'compatibility.')

    def _drive(self, *, rows: List[Dict[str, Any]], rollout: Any, cache: '_CandidateCache', writer: Any,
               recorder: Any, emit: Any, num_return: int,
               multi_turn_enabled: bool) -> Tuple[List[Dict[str, Any]], int, int]:
        """Emit every prompt's candidate group: cached ones first, the rest streamed through a driver.

        The driver (no checkpoint manager: the weights never change) runs each prompt's
        ``num_return`` episodes and hands the group out as soon as it completes; groups are emitted
        in DRAW order, so the checkpoint's count equals the prompts consumed and a resume skips
        exactly the emitted prefix. torchrun drives one shard per rank (gather merges), ray one
        process. Returns ``(rows, prompts emitted, generated tokens)``.
        """
        from swift.dev.builders import build_prompt_dataloader, split_prompt_and_reference
        from swift.dev.builders.streaming import build_streaming_driver
        from swift.dev.config import RolloutConfig

        infer_config = self.infer_config
        results: List[Dict[str, Any]] = []
        # Resume point: the count of prompts a prior run already wrote (its checkpoint), or 0 for a
        # fresh run (no checkpoint writer, or no state file).
        resume_from = max(0, getattr(writer, '_resume_from', -1))
        state = {'prompts': 0, 'tokens': 0, 'processed': 0}
        # Split once: the trajectory is what the sampler generates from, the reference is emit's
        # label anchor.
        trajectories, ground_truths = split_prompt_and_reference(rows, self.template_config)

        def emit_group(index: Optional[int], candidates: List[Dict[str, Any]]) -> None:
            out_rows: List[Dict[str, Any]] = []
            candidates = [candidate for candidate in candidates if _decoded(candidate)]
            if index is not None and candidates:
                state['prompts'] += 1
                state['tokens'] += sum(len(candidate.get('response_token_ids') or []) for candidate in candidates)
                scores = self._score_group(candidates, rows[index], trajectories[index], ground_truths[index])
                out_rows = emit(rows[index], trajectories[index], ground_truths[index], candidates, scores,
                                infer_config, recorder, multi_turn=multi_turn_enabled)
                results.extend(out_rows)
            # Written even when the group came back empty: under a torchrun gather every rank must
            # enter the same number of writer calls (one per drawn group), or the all_gather inside
            # the writer mismatches across ranks and the run hangs.
            writer.write(out_rows)
            state['processed'] += 1
            writer.checkpoint(state['processed'])

        # Cache short-circuit: a prompt already covered by cache_files is emitted verbatim, never
        # driven, so only the rest pay for generation.
        to_sample: List[int] = []
        for index, trajectory in enumerate(trajectories):
            hit = cache.lookup(trajectory, num_return, multi_turn=multi_turn_enabled)
            if hit is None:
                to_sample.append(index)
            elif state['processed'] < resume_from:
                state['processed'] += 1  # a prior run emitted it; the restored tmp file holds its rows
            else:
                emit_group(index, hit)
        # The draws a prior run already emitted are the first ones of the in-order stream.
        draw_skip = max(0, resume_from - state['processed'])
        state['processed'] += min(draw_skip, len(to_sample))
        pending = to_sample[draw_skip:]
        if not pending:
            return results, state['prompts'], state['tokens']
        if rollout is None:
            raise ValueError("sampler='no' requires cache_files to cover every input prompt.")
        # Per-prompt side inputs: the dataset row's non-message columns merged with the
        # trajectory's, so a reward func / tool sees the same extras whether or not the prompt was
        # cached; _ROW_INDEX finds the row again when the group comes back.
        extras = []
        for index in pending:
            extra = {key: copy.deepcopy(value) for key, value in rows[index].items() if key != 'messages'}
            extra.update({key: copy.deepcopy(value) for key, value in trajectories[index].items()
                          if key != 'messages'})
            extra[_ROW_INDEX] = index
            extras.append(extra)
        dataloader = build_prompt_dataloader(
            [trajectories[index]['messages'] for index in pending], extras, seed=0, shuffle=False,
            batch_size=_prompt_batch_size(infer_config.batch_size))
        # One group per batch, about _GROUPS_AHEAD rows of episodes running ahead of the emitter.
        driver = build_streaming_driver(
            dataloader=dataloader, rollout=rollout, rollout_config=(self.rollout_config or RolloutConfig()),
            batch_size=num_return, num_generations=num_return, num_train_epochs=1,
            max_staleness=max(0, math.ceil(max(num_return * 4, 8) / num_return) - 1))
        done: Dict[int, List[Dict[str, Any]]] = {}
        next_ordinal = 0
        try:
            for batch, _facts in driver:
                # group_key is the driver's draw ordinal of the group's prompt (stamped on every row)
                done[int(batch[0]['group_key'])] = batch
                while next_ordinal in done:
                    group = done.pop(next_ordinal)
                    emit_group(group[0][_ROW_INDEX], group)
                    next_ordinal += 1
        finally:
            driver.close()
        # A group the driver dropped (an episode failed) never arrived; it counts as consumed, so
        # the groups behind it are emitted now, in order.
        for ordinal in range(next_ordinal, max(done, default=next_ordinal - 1) + 1):
            group = done.get(ordinal)
            emit_group(group[0][_ROW_INDEX] if group else None, group or [])
        return results, state['prompts'], state['tokens']

    def _forward_pass(self, task_type: str) -> List[Dict[str, Any]]:
        """The non-sampling path: one forward pass over the rows, one plain value per row.

        A pooling task routes by backend: vLLM / SGLang have a pooling head, so the model is served
        through ``sampler.encode``; transformers falls back to ``build_model`` + ``forward_only``.
        A ``generative_reranker`` is a causal LM -- generation-scored on vLLM/SGLang, the same HF
        forward on transformers. A ``decision`` (System-1 scoring) model routes by family AND
        backend: the kev / clef families have native SGLang poolers, so ``sampler=sglang`` scores
        them on the SGLang offline engine; every other decision family (and every other engine
        backend) takes the transformers forward, honoured as a plain fallback with a log line,
        not an error. Row assembly is shared, label from the row's ``label`` column.
        """
        from swift.dev.builders import to_trajectory
        from swift.dev.components.infer.sglang_decision import (
            decision_via_sglang,
            is_sglang_decision_template,
        )

        rows = _forward_rows(self.dataset_config, self.max_rows, self.split_dataset_ratio,
                             self.distributed_config)
        reranker_use_activation = self.infer_config.reranker_use_activation

        if task_type == "decision":
            if self.backend == "sglang" and is_sglang_decision_template(self.template_config.template):
                per_row = decision_via_sglang(
                    self.model_config,
                    self.template_config,
                    rows,
                    adapters=self.adapters,
                    engine_args=self.engine_args,
                    shutdown=self.shutdown,
                )
                return _finalize_forward_results(
                    rows, per_row, task_type, self.backend, self.output_path, self.infer_config.metric
                )
            elif self.backend in ("vllm", "sglang"):
                logger.info(
                    f"run_infer: sampler={self.backend!r} has no pooling head for the decision "
                    "scoring_head; running the transformers forward instead."
                )
            per_row = _decision_via_transformers(
                self.model_config,
                self.template_config,
                self.distributed_config,
                rows,
                self.adapters,
                self.quantize_config,
                max_batch_size=(self.engine_args or {}).get("max_batch_size"),
            )
            return _finalize_forward_results(
                rows, per_row, task_type, self.backend, self.output_path, self.infer_config.metric
            )

        trajectories = [to_trajectory(row, list(row["messages"]), self.template_config) for row in rows]
        if self.backend in ('vllm', 'sglang'):
            if task_type == 'generative_reranker':
                per_row = _generative_reranker_via_sampler(
                    self.model_config, self.template_config, self.generation_config,
                    self.distributed_config, trajectories, self.adapters, self.quantize_config,
                    self.backend, self.engine_args, self.shutdown, reranker_use_activation)
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

    def _score_group(self, candidates: List[Any], row: Dict[str, Any],
                     trajectory: Dict[str, Any], ground_truth: Optional[str]) -> Optional[List[float]]:
        """Score one prompt's candidates -> one float each, or None when no channel is configured.

        Shared by the cached and the driven groups so scoring happens once per group and is orthogonal to
        storage. The ground truth, when scored, is appended so per-group normalisation sees it as
        the anchor, then trimmed back off -- it is never emitted as a candidate because the model
        did not produce it.
        """
        if not self._scorers:
            return None
        columns = {key: value for key, value in row.items() if key != 'messages'}
        scored = [{**columns, **candidate} for candidate in candidates]
        if self.infer_config.score_ground_truth and ground_truth:
            # trajectory['messages'] is the prompt with the reference turn already stripped (see
            # split_prompt_and_reference), so rebuilding prompt + ground_truth-as-assistant is the
            # correct single-turn conversation; __call__ rejects this in a multi-turn rollout.
            scored.append({**columns, 'messages': copy.deepcopy(trajectory['messages'])
                           + [{'role': 'assistant', 'content': ground_truth}]})
        from swift.dev.utils.scoring import collect

        matrix = [list(values) for values in zip(*(collect(scorer(scored)) for scorer in self._scorers))]
        return self._combined(matrix)[:len(candidates)]

    def _combined(self, matrix: List[List[Optional[float]]]) -> List[float]:
        """The matrix's two channel blocks -> one score per candidate (weight, normalise, sum).

        The ORM block is scaled by ``orm_channel_weight`` (legacy's ``prm + orm * 10`` is
        reproducible by setting it to 10, and the default 1.0 is an honest sum), the PRM block at
        1.0; ``normalize_rewards`` min-maxes each channel BEFORE the sum so the channel with the
        larger raw range cannot dominate regardless of the weights.
        """
        scores = [0.0] * len(matrix)
        total = len(self._scorers)
        blocks = ((0, self._orm_count, self._orm_weights, self.infer_config.orm_channel_weight),
                  (self._orm_count, total, self._prm_weights, 1.0))
        for start, end, weights, scale in blocks:
            if start == end:
                continue
            channel = [[row[index] for index in range(start, end)] for row in matrix]
            # nan-aware weighted sum: an unparseable verdict (None) scores 0 for that scorer
            # rather than poisoning the whole candidate.
            weighted = [
                sum((float(value) * weight for value, weight in zip(row, weights)
                     if value is not None and value == value), 0.0)
                for row in channel]
            if self.infer_config.normalize_rewards:
                weighted = _min_max(weighted)
            scores = [total + scale * value for total, value in zip(scores, weighted)]
        return scores


def _resolve_channel(reward_config: Optional['RewardConfig'], channel: str, sampler_model: Optional[str],
                     backend: str) -> Any:
    """One merged ``--orm`` / ``--prm`` list resolved into an ordered ChannelPlan.

    A None RewardConfig means "no reward channel at all": every field reads as its empty default.
    """
    from swift.dev.rewards.scorer import ChannelCompanions
    from swift.dev.rewards.scorer import resolve_channel

    return resolve_channel(
        list(getattr(reward_config, channel, None) or []),
        ChannelCompanions(
            adapters=getattr(reward_config, f'{channel}_adapters', None),
            model_types=getattr(reward_config, f'{channel}_model_type', None),
            task_types=getattr(reward_config, f'{channel}_task_type', None),
            model_revisions=getattr(reward_config, f'{channel}_model_revision', None),
            template_names=getattr(reward_config, f'{channel}_template', None),
            parallel_spec=getattr(reward_config, f'{channel}_parallel_spec', None)),
        channel=channel, sampler_model=sampler_model, backend=backend)


def _reward_device_groups(orm_plan: Any, prm_plan: Any,
                          distributed_config: Optional['DistributedConfig']) -> List[Tuple[str, int, int]]:
    """(name, world_size, gpus_per_worker) for each GPU-resident reward channel, 'orm' then 'prm'.

    A channel needs a group only when its reward model keeps weights resident (a scalar RM or an
    independent generative judge); a sampler-reusing judge and an API judge need none. The group's
    width and per-worker GPU count come from the model's OWN parallel_spec (the mesh's tp dim is
    real here, not dead info); with no spec it is the sampler's width, one GPU per rank.
    """
    from twinkle import DeviceMesh

    groups: List[Tuple[str, int, int]] = []
    for plan, name in zip((orm_plan, prm_plan), ('orm', 'prm')):
        spec = next((spec for spec in plan.specs if spec.kind in ('scalar', 'generative_independent')),
                    None)
        if spec is None:
            continue
        if spec.parallel_spec:
            mesh = DeviceMesh.from_spec(spec.parallel_spec)
            groups.append((name, mesh.world_size, mesh.tp_world_size or 1))
        else:
            width = (distributed_config.nproc_per_node
                     if distributed_config is not None and distributed_config.nproc_per_node else 1)
            groups.append((name, width, 1))
    return groups


def _initialize_for_sampling(distributed_config: 'DistributedConfig',
                             reward_groups: List[Tuple[str, int, int]]) -> None:
    """Initialize twinkle, giving the sampler -- and each GPU-resident reward model -- its own group.

    A scalar RM and an independent generative judge keep their weights on GPU, so under mode='ray'
    each is placed in its own disjoint DeviceGroup rather than sharing the sampler's cards;
    ``build_model`` / ``build_sampler`` target the group by the same name via ``remote_group``.
    Local mode has no groups to declare (the caller forbids a GPU-resident reward there), so it
    initializes plainly.
    """
    import twinkle
    from twinkle import DeviceGroup, Platform

    if distributed_config is not None and distributed_config.mode == 'remote':
        # No local twinkle runtime to initialize: this process is a single-process client and the
        # sampler/model live in the twinkle-server. Only the client transport is bound (once; a
        # re-init for a second component in the same process is a no-op on the session layer).
        from twinkle import init_twinkle_client

        if not distributed_config.server_base_url:
            raise ValueError("--mode remote requires --server_base_url (the twinkle-server gateway "
                             'root).')
        init_twinkle_client(base_url=distributed_config.server_base_url, api_key=distributed_config.server_api_key)
        return
    if distributed_config is None or distributed_config.mode != 'ray':
        # None is the run-wide default (single process, no ray): the sampler builds plainly, and
        # the caller has already refused any GPU-resident reward channel that would need a group.
        twinkle.initialize(mode='local')
        return
    sampler_ranks = distributed_config.nproc_per_node
    if sampler_ranks is None:
        raise ValueError("DistributedConfig.nproc_per_node is required in mode='ray' (it sizes the "
                         "sampler's Ray DeviceGroup). Pass it explicitly -- there is no default.")
    groups: List[Tuple[str, List[int], int]] = [(SAMPLER_GROUP, list(range(sampler_ranks)), 1)]
    start = sampler_ranks
    for name, world_size, gpus_per_worker in reward_groups:
        groups.append((name, list(range(start, start + world_size)), gpus_per_worker))
        start += world_size
    twinkle.initialize(
        mode='ray',
        nproc_per_node=start,
        groups=[
            DeviceGroup(name=name, ranks=ranks,
                        device_type=Platform.get_platform().__name__, gpus_per_worker=gpus_per_worker)
            for name, ranks, gpus_per_worker in groups])


def _build_scorers(infer_config: 'InferConfig', reward_config: Optional['RewardConfig'], orm_plan: Any,
                   prm_plan: Any, *, sampler: Any, model_config: 'ModelConfig',
                   template_config: 'TemplateConfig', backend: str, engine_args: Optional[Dict[str, Any]],
                   distributed_config: Optional['DistributedConfig'],
                   quantize_config: Optional['QuantizeConfig']) -> Tuple[List[Any], List[Any]]:
    """Build both channels' scorers from their resolved plans -> ``(orm_funcs, prm_funcs)``.

    Each channel's rules become reward callables and its (at most one) model spec becomes its
    scorer -- a scalar seq_cls RM scored by forward, a generative judge (reusing the sampler or
    owning its own engine), or an API judge -- through the shared single-spec builder. A generative
    judge that reuses the sampler is why this runs AFTER the sampler is built.
    """
    from twinkle.reward import FunctionWorker

    from swift.dev.builders import build_reward_model
    from swift.dev.config import DistributedConfig, RolloutConfig
    from swift.dev.rewards.reward import get_reward_funcs

    ray = distributed_config is not None and distributed_config.mode == 'ray'
    dist = distributed_config or DistributedConfig(mode='local')
    rollout_config = RolloutConfig()

    def _channel_funcs(plan: Any, name: str) -> Tuple[List[Any], List[str]]:
        funcs, names = get_reward_funcs(plan.rules, reward_config)
        funcs = [FunctionWorker(func) for func in funcs]
        names = list(names)
        for spec in plan.specs:
            remote_group = name if ray and spec.kind in ('scalar', 'generative_independent') else None
            funcs.append(build_reward_model(
                spec, model_config=model_config, template_config=template_config,
                distributed_config=dist, rollout_config=rollout_config, backend=backend,
                sampler=sampler, engine_args=engine_args, remote_group=remote_group,
                judge_template=getattr(reward_config, 'judge_template', None),
                quantize_config=quantize_config))
            names.append(spec.display_name)
        return funcs, names

    orm_funcs, orm_names = _channel_funcs(orm_plan, 'orm')
    prm_funcs, prm_names = _channel_funcs(prm_plan, 'prm')
    if not orm_funcs and not prm_funcs:
        logger.info('run_infer: no reward funcs -- every candidate is emitted as a positive.')
    else:
        logger.info(f'run_infer: orm={orm_names} (x{infer_config.orm_channel_weight}) prm={prm_names}, '
                    f'normalize={infer_config.normalize_rewards}')
    return orm_funcs, prm_funcs


def _min_max(scores: List[float]) -> List[float]:
    """Min-max the group into [0, 1]; a degenerate group collapses to a constant, nan passes through.

    A judge/API that returned nothing numeric scores nan and must not poison the group's min/max:
    normalise over the real scores and pass nan through untouched so the caller still sees it as
    "unscored" (and can drop or null it).
    """
    if not scores:
        return list(scores)
    valid = [value for value in scores if value == value]
    if not valid:
        return list(scores)
    low, high = min(valid), max(valid)
    if low == high:
        constant = min(1.0, low) if low > 0 else 0.0
        return [value if value != value else constant for value in scores]
    return [value if value != value else (value - low) / (high - low + 1e-5) for value in scores]


def _decoded(candidate: Dict[str, Any]) -> str:
    """The candidate's answer: its last assistant turn's text."""
    from twinkle.data_format import last_assistant_text

    return last_assistant_text(candidate)


def _candidate_messages(candidate: Dict[str, Any]) -> List[Dict[str, Any]]:
    """The candidate's full conversation (the emit view); a copy so output rows own their messages."""
    return copy.deepcopy(candidate['messages'])


def _emit_all(row: Dict[str, Any], trajectory: Dict[str, Any], ground_truth: Optional[str],
              candidates: List[Any], scores: Optional[List[float]], infer_config: 'InferConfig',
              recorder: Any = None, *, multi_turn: bool = False) -> List[Dict[str, Any]]:
    """'all' format: one row per prompt carrying every candidate (the plain-inference / eval shape).

    ``response`` is the first candidate and ``responses`` all of them, the shape the acc/rouge metric
    and downstream eval read. When a reward func scored the group, ``scores`` rides along per
    candidate; with save_rollout_tokens, ``rollout_tokens`` is the per-candidate NPZ path list.
    """
    prompt_id = _prompt_key(trajectory['messages'])
    out: Dict[str, Any] = {key: value for key, value in row.items() if key != 'messages'}
    out['response'] = _decoded(candidates[0])
    out['responses'] = [_decoded(candidate) for candidate in candidates]
    out['messages'] = _candidate_messages(candidates[0])
    if multi_turn:
        # Each multi-turn candidate is a distinct trajectory (its own tool calls); ``messages`` keeps
        # only the first, so store every candidate's full trajectory rather than losing all but one.
        out['all_messages'] = [_candidate_messages(candidate) for candidate in candidates]
    if candidates[0].get('rollout_logps'):
        # Present for a generated candidate (a cached one has none); mirrors legacy's per-row
        # ``logprobs`` for the response this row keeps.
        out['logprobs'] = list(candidates[0]['rollout_logps'])
    if ground_truth is not None:
        out['labels'] = ground_truth
    if scores is not None:
        # A judge that returned nothing numeric scores nan; null it so the row stays
        # json-serialisable.
        out['scores'] = [None if score != score else score for score in scores]
    if recorder is not None:
        out['rollout_tokens'] = [
            recorder.record(prompt_id, index, candidate) for index, candidate in enumerate(candidates)
        ]
    return [out]


def _emit_dpo(row: Dict[str, Any], trajectory: Dict[str, Any], ground_truth: Optional[str],
              candidates: List[Any], scores: Optional[List[float]], infer_config: 'InferConfig',
              recorder: Any = None, *, multi_turn: bool = False) -> List[Dict[str, Any]]:
    """'dpo' format: best-of-n chosen/rejected pairs for DPO training.

    Without any reward func (``scores is None``) every candidate is emitted as a positive with no
    rejected turn -- there is no ranking, so naming one candidate worse would be a fabrication.
    With scores, candidates at or below ``reward_threshold`` are dropped, an easy prompt (see
    ``easy_query_threshold``) is skipped whole, then the top ``n_best_to_keep`` become positives
    paired against the lowest scorer.
    """
    prompt_id = _prompt_key(trajectory['messages'])
    keep_tokens = recorder is not None
    record = recorder.record if keep_tokens else None
    if scores is None:
        rows = []
        for index, positive in enumerate(candidates):
            token_path = record(prompt_id, index, positive) if keep_tokens else None
            rows.append(_dpo_row(row, trajectory, positive, None, ground_truth, rollout_tokens=token_path,
                                 multi_turn=multi_turn))
        return rows

    threshold = infer_config.reward_threshold
    # An unscored candidate (nan, from a judge that returned nothing numeric) can be neither a
    # positive nor the rejected response, and nan in reward_score breaks JSON serialisation --
    # ``scorable`` is the nan-free pool the ranking works on.
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
    logger.debug(f'scores={[round(s, 4) for s in scores]} positives={positives} '
                 f'negative={negative_index}')
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
                rejected_reward_score=scores[negative_index] if keep_tokens else None,
                multi_turn=multi_turn))
    return rows


def _dpo_row(row: Dict[str, Any], trajectory: Dict[str, Any], positive: Any,
             negative: Optional[Any], ground_truth: Optional[str], *,
             rollout_tokens: Optional[str] = None, reward_score: Optional[float] = None,
             rejected_rollout_tokens: Optional[str] = None,
             rejected_reward_score: Optional[float] = None,
             multi_turn: bool = False) -> Dict[str, Any]:
    """Build one DPO output row while preserving complete multi-turn chosen/rejected trajectories.

    The token/score fields are the save_rollout_tokens schema additions: each is written only when
    the recorder produced it (a local backend, tokens kept), so with the feature off the row is
    unchanged.
    """
    generated_keys = {
        'messages', 'rejected_messages', 'rejected_response', 'multi_turn', 'truncated', 'rollout_infos',
        'rejected_truncated', 'rejected_rollout_infos', 'id'
    }
    out: Dict[str, Any] = {key: value for key, value in row.items() if key not in generated_keys}
    out['messages'] = _candidate_messages(positive)
    if negative is not None:
        if multi_turn:
            out['rejected_messages'] = _candidate_messages(negative)
        else:
            out['rejected_response'] = _decoded(negative)
    if multi_turn:
        out['multi_turn'] = True
        out['truncated'] = bool(positive.get('truncated'))
        # The ledger column travels with the trajectory: generated_keys above strips a dataset column
        # of the same name, so what the row carries must come from the candidate that was kept.
        out['rollout_infos'] = dict(positive.get('rollout_infos') or {})
        if negative is not None:
            out['rejected_truncated'] = bool(negative.get('truncated'))
            out['rejected_rollout_infos'] = dict(negative.get('rollout_infos') or {})
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
               candidates: List[Any], scores: Optional[List[float]], infer_config: 'InferConfig',
               recorder: Any = None, *, multi_turn: bool = False) -> List[Dict[str, Any]]:
    """'grpo' format: one row per prompt holding the whole sampled group as an offline GRPO corpus.

    Each candidate keeps its full ``all_messages`` trajectory (multi-turn candidates diverge at
    their own tool calls, so none is collapsed to one response) and its rollout-token NPZ sidecar,
    plus a stable group ``id`` and ``num_generations``. The group-relative advantage is NOT
    computed here: the training step recomputes it from the stored rewards and logps.
    """
    prompt_id = _prompt_key(trajectory['messages'])
    out: Dict[str, Any] = {key: value for key, value in row.items() if key != 'messages'}
    out['id'] = prompt_id
    out['messages'] = copy.deepcopy(trajectory['messages'])
    out['completions'] = [_decoded(candidate) for candidate in candidates]
    out['all_messages'] = [_candidate_messages(candidate) for candidate in candidates]
    out['num_generations'] = len(candidates)
    if ground_truth is not None:
        out['labels'] = ground_truth
    if recorder is not None:
        out['rollout_tokens'] = [
            recorder.record(prompt_id, index, candidate) for index, candidate in enumerate(candidates)
        ]
    if scores is not None:
        # A judge that returned nothing numeric scores nan; null it so the row stays
        # json-serialisable.
        out['scores'] = [None if score != score else score for score in scores]
    else:
        out['scores'] = None
    return [out]


def _plan_rows(rows: List[Dict[str, Any]], infer_config: 'InferConfig') -> List[Dict[str, Any]]:
    """The rows this run will process: ``max_batches`` truncates, a batch below one is fatal.

    Both properties are easy to break silently -- a coerced batch size of zero would make the
    generate loop plan no batches at all, and rounding the row count down to whole batches would
    drop the trailing partial batch, i.e. silently skip data the user asked to run.
    """
    if infer_config.batch_size < 1:
        raise ValueError(f'InferConfig.batch_size must be >= 1, got {infer_config.batch_size}.')
    if infer_config.max_batches:
        return rows[:infer_config.max_batches * infer_config.batch_size]
    return rows


def _prompt_batch_size(batch_size: Optional[int]) -> int:
    """The prompt-source batch size for the per-sample driver, widened to the launch's dp width.

    The driver flattens the loader's batches straight back to rows, so the size is only the reader's
    fetch granularity -- but the twinkle loader splits every global batch across the dp ranks and
    asserts ``batch_size >= dp_width`` with divisibility, while ``infer_config.batch_size`` is the
    generate-batch size (default 1) and can be smaller than a multi-rank torchrun launch's world
    size. Round up to a multiple of the width; at width 1 the value passes through unchanged.
    """
    from twinkle.utils import Platform

    size = max(1, int(batch_size or 1))
    world = max(1, int(Platform.get_world_size()))
    if world > 1:
        size = -(-size // world) * world
    return size


def compute_metric(results: List[Dict[str, Any]], metric: Literal['acc', 'rouge']) -> Dict[str, float]:
    """Score the first completion of each row against its reference answer.

    ``acc`` is exact string equality, what legacy's ``--metric acc`` measured -- NOT the token-level
    ``twinkle.metric.Accuracy`` (logits vs label ids), which reports a much higher number. Only the
    first completion is scored: with ``num_samples > 1`` best-of-n is pass@n, a different measure.
    Rows without a reference are skipped, not counted as wrong.
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


class _RolloutRecorder:
    """Writes one NPZ per kept candidate under ``token_dir``; a no-op when ``enabled`` is False.

    ``base_dir`` is the directory the jsonl lives in, so :meth:`record` returns a path relative to
    it -- that relative path is what gets embedded in the row, keeping the jsonl portable alongside
    its sidecar. NPZ filenames are positionally deterministic
    (``{prompt_id}_c{candidate_index}.npz``), so replaying a batch on resume overwrites the same
    files instead of appending duplicates.
    """

    def __init__(self, token_dir: str, base_dir: str = '', enabled: bool = False):
        self.token_dir = token_dir
        self.base_dir = base_dir
        self.enabled = enabled
        if enabled:
            os.makedirs(token_dir, exist_ok=True)

    def record(self, prompt_id: str, candidate_index: int, candidate: Any) -> Optional[str]:
        """Persist one candidate's rollout tokens; return the NPZ path relative to ``base_dir``.

        The sidecar holds the payload every dump format shares (``input_ids`` / ``labels`` /
        ``completion_mask`` / ``response_token_ids`` / ``rollout_logps``)
        -- old_logps are the one thing training cannot recompute (the policy has drifted). Multimodal
        tensors are omitted on purpose: reproducible, so training re-encodes them from the row's kept
        media columns and messages, as an online rollout would. None when disabled or token-less.
        """
        if not self.enabled:
            return None
        input_ids = candidate.get('input_ids')
        labels = candidate.get('labels')
        if not input_ids or not labels:
            return None
        arrays = {
            'input_ids': np.asarray(input_ids, dtype=np.int64),
            'labels': np.asarray(labels, dtype=np.int64),
            'response_token_ids': np.asarray(candidate.get('response_token_ids') or [], dtype=np.int64),
            'rollout_logps': np.asarray(candidate.get('rollout_logps') or [], dtype=np.float32),
        }
        completion_mask = candidate.get('completion_mask')
        if completion_mask is not None:
            arrays['completion_mask'] = np.asarray(completion_mask, dtype=np.int8)
        name = f'{prompt_id}_c{candidate_index}.npz'
        abs_path = os.path.join(self.token_dir, name)
        np.savez_compressed(abs_path, **arrays)
        return os.path.relpath(abs_path, self.base_dir) if self.base_dir else name


class _CandidateCache:
    """Candidates from earlier runs, keyed by prompt, i.e. legacy's ``cache_files``.

    Keyed on the prompt's messages rather than on row order, because the cache files come from
    other runs whose dataset order and slicing need not match this one's. A prompt is only served
    from cache when it has at least as many candidates as this run asks for -- fewer would silently
    shrink the group that the reward ranking then works on.
    """

    def __init__(self, cache_files: List[str]):
        # Values pair each stored sample with ITS OWN row's multi_turn flag: a multi-turn-stored
        # trajectory can serve a single-turn run as a plain completion, never the other way round.
        self.by_prompt: Dict[str, List[Tuple[Any, bool]]] = {}
        for path in cache_files or []:
            if not os.path.exists(path):
                logger.warning(f'cache_files entry {path} does not exist; ignoring it.')
                continue
            self._load(path)
        if self.by_prompt:
            logger.info(f'run_infer: cache covers {len(self.by_prompt)} prompts from '
                        f'{len(cache_files)} file(s)')

    def _load(self, path: str) -> None:
        # Any run output is a store the storage layer reads back -- a jsonl dump (row-wise
        # tolerant of a crashed producer's truncated tail), an arrow store directory, or another
        # registered format -- so one run's output serves as another run's cache verbatim.
        from swift.dev.dataset.storage import STORAGE_MAPPING, detect_storage_format

        storage_format = detect_storage_format(path)
        if storage_format is None:
            logger.warning(f'cache_files entry {path} is not a store the storage layer reads (no known '
                           'format detected; name the output .jsonl/.json or point at a store '
                           'directory); ignoring it.')
            return
        try:
            rows = STORAGE_MAPPING[storage_format]().iter_rows(path)
        except Exception as e:
            # A cache is an optimisation, not a dependency: an unreadable entry (a foreign or
            # broken file) is skipped with the reason stated, like a missing one -- the run just
            # pays for the generation itself.
            logger.warning(f'cache_files entry {path} could not be read as {storage_format!r} ({e}); '
                           'ignoring it.')
            return
        for entry in rows:
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
            self.by_prompt.setdefault(key, []).append((
                {'messages': copy.deepcopy(messages), 'truncated': bool(entry.get('truncated'))}, multi_turn))

    def lookup(self, trajectory: Dict[str, Any], wanted: int,
               *, multi_turn: bool = False) -> Optional[List[Any]]:
        if not self.by_prompt:
            return None
        cached = self.by_prompt.get(_prompt_key(trajectory['messages']))
        if cached is None:
            return None
        if not multi_turn:
            # A multi-turn-stored entry still serves as a plain completion, but the row must carry
            # THIS run's single-turn view -- the current prompt plus that completion, never the
            # stored tool-call trajectory.
            prompt_messages = copy.deepcopy(trajectory['messages'])
            cached = [{'messages': prompt_messages + [{'role': 'assistant', 'content': _decoded(sample)}]}
                      for sample, _flag in cached]
        else:
            cached = [copy.deepcopy(sample) for sample, flag in cached if flag]
        return cached[:wanted] if len(cached) >= wanted else None


def _prompt_key(messages: List[Dict[str, Any]]) -> str:
    return hashlib.md5(
        json.dumps(messages, sort_keys=True, ensure_ascii=False, default=str).encode('utf-8')).hexdigest()


def _is_too_easy(n_passing: int, n_candidates: int, easy_query_threshold: Optional[float]) -> bool:
    """Whether a prompt is too easy to be worth keeping.

    A prompt almost every candidate already answers correctly carries no learning signal, and
    keeping it biases the dataset toward what the model can do. None disables the filter.
    """
    if easy_query_threshold is None:
        return False
    return n_passing / n_candidates >= easy_query_threshold


def _initialize_for_forward(distributed_config: 'DistributedConfig') -> None:
    """Initialize twinkle for a forward-pass run: the 'model' DeviceGroup under ray, local otherwise.

    Both forward paths reach the model through twinkle -- ``build_sampler`` for the vLLM/SGLang
    pooling head and ``build_model`` for the transformers forward -- and ray placement targets the
    DeviceGroup named 'model' (``build_model``'s ``remote_group`` default), so the group has to
    exist before either is built; a local (torchrun) run has no group to declare.
    """
    import twinkle
    from twinkle import DeviceGroup, Platform

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
        groups=[DeviceGroup(name='model', ranks=list(range(nproc)),
                            device_type=Platform.get_platform().__name__, gpus_per_worker=1)])


def _forward_rows(dataset_config: 'DatasetConfig', max_rows: Optional[int], split_dataset_ratio: float,
                  distributed_config: 'DistributedConfig') -> List[Dict[str, Any]]:
    """Initialize twinkle for a forward pass, then load the prompt rows it runs over.

    Shared by the pooling and generative-reranker forwards: they differ in the forward they run,
    not in the bootstrap or the input; the rows load AFTER the init so a ray-placed model resolves
    its DeviceGroup.
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

    Shared by the two forward task families :meth:`InferComponent._forward_pass` dispatches -- a
    pooling task and a ``generative_reranker``: both produce exactly one plain value per row (a
    vector, class logits, or a relevance score), so the row shape is the same. The label comes from
    the row's own ``label`` column rather than a trailing assistant turn, because there is no
    completion to strip.
    """
    from swift.dev.dataset.storage import STORAGE_MAPPING

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
        STORAGE_MAPPING['jsonl']().write(results, output_path)
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
    model_config: 'ModelConfig',
    template_config: 'TemplateConfig',
    distributed_config: Optional['DistributedConfig'],
    adapters: Optional[List[str]],
    quantize_config: Optional['QuantizeConfig'],
    backend: str,
    engine_args: Optional[Dict[str, Any]],
    shutdown: bool,
):
    """Build a vLLM/SGLang sampler for a forward-pass path, and shut it down afterwards.

    Shared by the pooling (``encode``) and generative-reranker (``sample``) sampler paths: they
    differ only in the call they make, while the model/template/mesh build, the single-adapter
    selection and the teardown are identical. Yields ``(sampler, processor, adapter_path)``.
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
    model_config: 'ModelConfig',
    template_config: 'TemplateConfig',
    distributed_config: Optional['DistributedConfig'],
    trajectories: List[Dict[str, Any]],
    adapters: Optional[List[str]],
    quantize_config: Optional['QuantizeConfig'],
    backend: str,
    engine_args: Optional[Dict[str, Any]],
    shutdown: bool,
    reranker_use_activation: bool = True,
) -> List[Any]:
    """Run the pooling forward on a vLLM/SGLang sampler's ``encode``, one plain value per row."""
    from swift.dev.builders import pooled_data, to_pooling_params

    with _forward_sampler(model_config, template_config, distributed_config, adapters, quantize_config,
                          backend, engine_args, shutdown) as (sampler, _processor, adapter_path):
        # A cross-encoder reranker's raw score is a logit; ``use_activation`` has twinkle sigmoid
        # it into [0, 1] relevance (legacy reranker_use_activation), riding in as a PoolingParams
        # override -- the channel to_pooling_params documents for such post-processing knobs.
        overrides = {'use_activation': reranker_use_activation} if model_config.task_type == 'reranker' else {}
        pooling_params = to_pooling_params(model_config.task_type, **overrides)
        kwargs: Dict[str, Any] = {}
        if adapter_path is not None:
            kwargs['adapter_path'] = adapter_path
        return pooled_data(sampler.encode(trajectories, pooling_params, **kwargs))


def _forward_via_transformers(
    model_config: 'ModelConfig',
    template_config: 'TemplateConfig',
    distributed_config: Optional['DistributedConfig'],
    rows: List[Dict[str, Any]],
    trajectories: List[Dict[str, Any]],
    adapters: Optional[List[str]],
    quantize_config: Optional['QuantizeConfig'],
    max_batch_size: Optional[int] = None,
    reranker_use_activation: bool = True,
) -> List[Any]:
    """Run one HF ``forward_only`` pass (transformers backend), one plain value per row.

    Serves both the pooling tasks (seq_cls / embedding / reranker) and the transformers
    ``generative_reranker`` path, which reads its yes/no logit difference out of the raw logits
    returned here. Requests are chunked by ``max_batch_size`` so a large dataset does not
    materialise every encoded input and every output tensor at once.
    """
    from swift.dev.builders import build_model, build_template, load_model_processor
    from swift.dev.config import DistributedConfig

    task_type = model_config.task_type
    _, processor = load_model_processor(model_config)
    template = build_template(template_config, processor)
    # build_template leaves 'train' mode (its default caller is the dataset path), whose pair
    # encoders read contrastive-only ``positive_messages`` / ``negative_messages`` columns; the
    # inference mode below takes the anchor-only branch a forward pass wants.
    template.set_mode('transformers')
    model = build_model(
        model_config, distributed_config or DistributedConfig(), quantize_config=quantize_config)
    if adapters:
        # add_adapter_to_model takes a checkpoint directory as well as a PeftConfig, so a trained
        # adapter is loaded here rather than built -- apply_tuner would create a fresh, untrained one.
        for index, adapter in enumerate(adapters):
            model.add_adapter_to_model(f'adapter_{index}' if index else 'default', adapter)

    # forward_only resolves a generative_reranker's yes/no token ids off the model's template, so
    # it must be attached even with inputs pre-encoded -- else the transformers reranker path trips
    # twinkle's "call set_template before forwarding".
    model.set_template(template)

    batch_size = max_batch_size or len(rows) or 1
    per_row: List[Any] = []
    for start in range(0, len(rows), batch_size):
        chunk = trajectories[start:start + batch_size]
        encoded = [template.encode(trajectory) for trajectory in chunk]
        outputs = model.forward_only(inputs=encoded, task=task_type, return_logits=True)
        per_row.extend(_per_row_outputs(outputs, len(chunk)))
    if reranker_use_activation and task_type in ('reranker', 'generative_reranker'):
        # forward_only returns the raw relevance logit (a cross-encoder score, or a generative
        # reranker's yes/no logit difference); sigmoid maps it to [0, 1], matching legacy's
        # activation and the vLLM/SGLang paths. seq_cls / embedding are left untouched.
        per_row = [
            [_sigmoid(value) for value in row] if isinstance(row, (list, tuple)) else _sigmoid(row)
            for row in per_row
        ]
    return per_row


def _decision_via_transformers(
    model_config: "ModelConfig",
    template_config: "TemplateConfig",
    distributed_config: Optional["DistributedConfig"],
    rows: List[Dict[str, Any]],
    adapters: Optional[List[str]],
    quantize_config: Optional["QuantizeConfig"],
    max_batch_size: Optional[int] = None,
) -> List[Any]:
    """Run the decision (System-1 scoring) forward, one per-record prediction list per row.

    Differs from :func:`_forward_via_transformers` in three deliberate ways:

    * rows encode from their RAW columns (``messages`` + ``kinds`` / ``questions`` / ``options`` /
      ``target_probs``), not ``to_trajectory`` dicts -- the decision templates read the decision
      columns, and ``to_trajectory`` drops them;
    * the template/task pair is built with ``task_type='decision'`` and the DEV InputProcessor is
      installed (``set_processor`` + ``processor_kwargs``): the decision batch flatten
      (``pop_media -> collate_media -> finalize_batch``) lives in the template's media hooks, which
      the default twinkle processor never runs;
    * each chunk's ``ScoringOutput`` decodes through ``template.decode_decision`` -- per record,
      ``[{'index': argmax option, 'probs': masked softmax over real options}]`` -- rather than the
      one-tensor-per-row unpacking of the pooling heads.
    """
    from swift.dev.builders import build_model, build_template, load_model_processor
    from swift.dev.config import DistributedConfig
    from swift.dev.processor import InputProcessor, processor_kwargs

    # 'decision' is a first-class forward task in twinkle's task registry
    # (twinkle.processor.tasks); no runtime patching is needed.

    _, processor = load_model_processor(model_config)
    template = build_template(template_config, processor, task_type="decision")
    template.set_mode("transformers")
    model = build_model(model_config, distributed_config or DistributedConfig(), quantize_config=quantize_config)
    if adapters:
        # add_adapter_to_model takes a checkpoint directory as well as a PeftConfig, so a trained
        # adapter (LoRA + its sidecar scoring_head) is loaded here rather than built.
        for index, adapter in enumerate(adapters):
            model.add_adapter_to_model(f"adapter_{index}" if index else "default", adapter)
    model.set_processor(InputProcessor, **processor_kwargs(template))
    model.set_template(template)

    batch_size = max_batch_size or len(rows) or 1
    per_row: List[Any] = []
    for start in range(0, len(rows), batch_size):
        chunk = rows[start : start + batch_size]
        encoded = [template.encode(row) for row in chunk]
        outputs = collect(model.forward_only(inputs=encoded, task="decision", return_logits=True))
        preds, _logprobs = template.decode_decision(outputs, top_logprobs=1)
        per_row.extend(preds)
    return per_row


def _per_row_outputs(outputs: Any, num_rows: int) -> List[Any]:
    """Unpack ``forward_only``'s output into one plain-Python value per row.

    The shape depends on the head: seq_cls gives logits per row, embedding gives a vector. Both are
    returned as lists so the result is jsonl-serialisable, which is the whole point of this path.
    A lazy remote return is collected here, so the local-mode dict and the ray handle feed the
    same unpacking.
    """
    outputs = collect(outputs)
    tensor = outputs
    if isinstance(outputs, dict):
        # forward_only(task='embedding') returns the pooled vectors under 'embeddings' (plural),
        # the other heads under 'logits' / 'embedding' / 'last_hidden_state'; missing the plural
        # key would leave the raw tensor dict as the row value.
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
    model_config: 'ModelConfig',
    template_config: 'TemplateConfig',
    generation_config: Optional['GenerationConfig'],
    distributed_config: Optional['DistributedConfig'],
    trajectories: List[Dict[str, Any]],
    adapters: Optional[List[str]],
    quantize_config: Optional['QuantizeConfig'],
    backend: str,
    engine_args: Optional[Dict[str, Any]],
    shutdown: bool,
    reranker_use_activation: bool = True,
) -> List[float]:
    """Score a generative reranker on a vLLM/SGLang generation engine, one score per row."""
    from swift.dev.builders import to_sampling_params

    # Built as a generation engine, not a pooling runner: build_sampler treats
    # ``generative_reranker`` as a generative task_type, so no ``runner='pooling'``/``is_embedding``
    # is injected.
    with _forward_sampler(model_config, template_config, distributed_config, adapters, quantize_config,
                          backend, engine_args, shutdown) as (sampler, processor, adapter_path):
        tokenizer = _tokenizer_of(processor)
        positive_id, negative_id = _generative_reranker_token_ids(tokenizer)
        # temperature is forced to 1.0 so logprob(yes) - logprob(no) equals the raw logit
        # difference the HF path returns; max_tokens=1 (not 0) because twinkle's logprobs-only
        # max_tokens==0 path drops the logprobs we need.
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

    With ``use_activation`` the log-ratio is squashed by a sigmoid into ``P(yes) / (P(yes) +
    P(no))``, a [0, 1] relevance comparable across queries -- the same activation legacy applies to
    a generative reranker, and the counterpart of the pooling path's ``PoolingParams.use_activation``.
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
    """A tokenizer from whatever ``load_model_processor`` returned (a tokenizer, or a processor
    wrapping one)."""
    return getattr(processor, 'tokenizer', processor)
