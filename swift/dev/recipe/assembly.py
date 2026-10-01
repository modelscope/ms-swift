# Copyright (c) ModelScope Contributors. All rights reserved.
"""The shared spine of every training recipe -- and the one place plugins are inserted.

Eight recipes ran the same sequence -- validate -> processor/template -> dataloaders -> step budget
-> model -> tuner -> loss/optimizer -> loop.fit -> final checkpoint -- and differed only in which
``configure_*_loss`` they called, which ``task`` they trained, and (for the RL ones) what extra models
they build. That sequence is also *order-locked* by twinkle in several places, so eight copies meant
eight chances to get the order wrong: the notes on why ``apply_tuner`` precedes ``set_processor``, and
why a class is passed there but an instance to ``set_template``, were duplicated four times over.
Worse, the parts already recognised as shared were reached as
``from swift.dev.recipe.run_sft import _initialize_twinkle`` -- seven recipes importing a private name
out of an eighth recipe.

Plugin loading is a stage of :meth:`TrainAssembly.prepare`, which gives an extension point exactly one
insertion site: no recipe can forget it and none can do it differently. It runs before the run's first
name lookup (a reward / loss name means nothing until the user's file has been imported) and before
cross-validation, so a plugin may register the very thing validation then checks.

A recipe whose shape genuinely differs (DPO's reference model, GRPO's sampler) still calls
``prepare()`` and may use individual stages; the stages are separate methods, and each keeps its result
on the assembly, precisely so a recipe can take the ones it needs and hand-roll the rest.
"""
from __future__ import annotations
import math
import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple

from swift.dev.utils import get_logger

if TYPE_CHECKING:
    from swift.dev.config import (
        CheckpointConfig,
        DatasetConfig,
        DistributedConfig,
        LoggingConfig,
        MegatronConfig,
        ModelConfig,
        MoEConfig,
        PluginConfig,
        QuantizeConfig,
        RLHFConfig,
        TemplateConfig,
        TrainConfig,
        TunerConfig,
    )

logger = get_logger()


def configure_frozen_adapter(model: Any, template: Any, adapters: List[str], *, role: str) -> Any:
    """Load one frozen adapter onto an auxiliary model and bind its processor/template group."""
    from swift.dev.processor import InputProcessor

    if len(adapters) > 1:
        raise ValueError(f'{role}_adapters currently supports exactly one adapter per auxiliary model.')
    if adapters:
        adapter_name = f'{role}_adapter'
        model.add_adapter_to_model(adapter_name, adapters[0], is_trainable=False)
        model.set_processor(InputProcessor, adapter_name=adapter_name)
        model.set_template(template, adapter_name=adapter_name)
    else:
        model.set_processor(InputProcessor)
        model.set_template(template)
    return model


def _resolve_step_interval(value: Optional[float], total_steps: int, name: str) -> Optional[int]:
    """Resolve a checkpoint/eval interval after the optimizer-step budget is known.

    Values in (0, 1) are ratios of the total run and round up so a positive ratio can never become
    step zero. Values >= 1 are absolute intervals and must be integral.
    """
    if value is None:
        return None
    if value < 0:
        raise ValueError(f'{name} must be non-negative, got {value}.')
    if 0 < value < 1:
        return max(1, math.ceil(total_steps * value))
    if not float(value).is_integer():
        raise ValueError(f'{name}={value} is not a ratio in (0, 1) or an integer step interval.')
    return int(value)


@dataclass
class TrainAssembly:
    """Assemble a training run from atomic Configs, one stage at a time.

    ``recipe`` is the caller's name and is used in the fail-fast messages, so "the dataloader is too
    small" still says which recipe computed it.
    """

    recipe: str
    model_config: 'ModelConfig'
    template_config: 'TemplateConfig'
    dataset_config: 'DatasetConfig'
    train_config: 'TrainConfig'
    distributed_config: 'DistributedConfig'
    checkpoint_config: Optional['CheckpointConfig'] = None
    tuner_config: Optional['TunerConfig'] = None
    rlhf_config: Optional['RLHFConfig'] = None
    #: The twinkle task the loop runs (``None`` -> the loop's own default, ``'causal_lm'``).
    task: Optional[str] = None
    #: Override for the LOOP's forward task only, leaving ``task`` (and so task_type / template
    #: encoding) untouched. Fused-linear-CE needs the split: rows still encode as causal_lm, but the
    #: training forward must run under 'fused_lm_ce' so TransformersFusedCEPatch skips the lm_head
    #: GEMM for the Liger fused kernel. ``None`` -> fall back to ``task``.
    loop_task: Optional[str] = None
    output_dir: str = 'output'
    logging_config: Optional['LoggingConfig'] = None
    quantize_config: Optional['QuantizeConfig'] = None
    megatron_config: Optional['MegatronConfig'] = None
    moe_config: Optional['MoEConfig'] = None
    plugin_config: Optional['PluginConfig'] = None

    # --- stage results, in the order the stages produce them ---
    sp_mesh: Any = field(default=None, init=False)
    processor: Any = field(default=None, init=False)
    template: Any = field(default=None, init=False)
    dataloader: Any = field(default=None, init=False)
    eval_dataloader: Any = field(default=None, init=False)
    total_opt_steps: int = field(default=0, init=False)
    model: Any = field(default=None, init=False)
    loop: Any = field(default=None, init=False)

    @property
    def ga(self) -> int:
        """Gradient accumulation, read straight off the Config.

        dev does NOT replicate legacy's implicit derivation (legacy swift.trainers.arguments derives
        ``max(1, ceil(16 / per_device_train_batch_size / world_size))`` when it is unset, targeting a
        global batch of ~16), so the same argv yields a different effective batch than legacy on
        multi-GPU. Explicit over implicit; documented as a known behavioral difference.
        """
        return self.train_config.gradient_accumulation_steps

    @property
    def task_type(self) -> Optional[str]:
        """The task_type the template must encode with.

        The recipe's own ``task`` wins (an embedding run encodes embedding rows whether or not the
        Config says so), otherwise the Config's -- which is where the task-agnostic recipes get theirs:
        DPO's 'rm' type rides ``task_type='seq_cls'`` and would silently encode SFT rows without it.
        """
        return self.task or self.model_config.task_type

    @staticmethod
    def initialize_twinkle(distributed_config: 'DistributedConfig',
                           *,
                           seed: int = 42,
                           full_determinism: bool = False) -> None:
        """Initialize twinkle for EVERY backend -- required for a DeviceMesh to reach the model.

        mode='ray' builds the 'model' DeviceGroup the workers hold, on EITHER backend: build_model
        places the model there via remote_group='model' (Megatron in _build_megatron_model, the
        transformers path in _apply_ray_placement), while the driver orchestrates and does not join
        the group. Everything else (both backends under torchrun) initializes in 'local' mode.

        ``seed`` / ``full_determinism`` are forwarded to ``twinkle.initialize`` so the user's
        TrainConfig.seed actually steers the run: twinkle seeds the driver here and re-seeds each ray
        worker from the same values (infra/__init__.py), so without this twinkle would fall back to
        its own default (seed=42) and silently overwrite the configured seed.

        See doc.md 'run_sft twinkle 初始化' for why this is load-bearing on the transformers backend
        and why there is no teardown counterpart.
        """
        import twinkle
        from twinkle import DeviceGroup

        if distributed_config.mode == 'ray':
            # The DeviceGroup named 'model' is what build_model targets with remote_group='model'
            # (Megatron and transformers alike); the driver orchestrates and is not part of the group.
            nproc = distributed_config.nproc_per_node
            if nproc is None:
                raise ValueError("DistributedConfig.nproc_per_node is required in mode='ray' (it sizes the Ray "
                                 "'model' DeviceGroup). Pass it explicitly -- there is no default.")
            twinkle.initialize(
                mode='ray',
                nproc_per_node=nproc,
                seed=seed,
                full_determinism=full_determinism,
                # --ray_exp_name names this Ray run (the cluster/worker-name prefix); a local run has
                # no Ray experiment to name, so it is read only here. None keeps twinkle's default.
                name=distributed_config.ray_exp_name,
                groups=[DeviceGroup(name='model', ranks=list(range(nproc)), device_type='GPU', gpus_per_worker=1)])
        else:
            twinkle.initialize(mode='local', seed=seed, full_determinism=full_determinism)

    def prepare(self) -> 'TrainAssembly':
        """Process and validate Configs before building anything heavy.

        ``process_configs`` loads configured plugins and resolves derived values. ``validate_configs``
        then reads that resolved state and rejects invalid combinations. Both are idempotent, so CLI
        callers may safely have completed the same lifecycle before entering the recipe.
        """
        from swift.dev.config import process_configs, validate_configs

        process_configs(
            self.model_config,
            self.template_config,
            self.dataset_config,
            self.train_config,
            self.distributed_config,
            self.checkpoint_config,
            self.tuner_config,
            self.rlhf_config,
            megatron_config=self.megatron_config,
            quantize_config=self.quantize_config,
            plugin_config=self.plugin_config,
        )
        validate_configs(
            self.model_config,
            self.template_config,
            self.dataset_config,
            self.train_config,
            self.distributed_config,
            self.checkpoint_config,
            self.tuner_config,
            self.rlhf_config,
            self.logging_config,
            quantize_config=self.quantize_config,
            megatron_config=self.megatron_config,
            moe_config=self.moe_config,
        )
        return self

    def plan_sp_mesh(self) -> Any:
        """Build the ONE Ulysses SP mesh all consumers share (None when SP is off).

        The mesh is the only thing twinkle's SP needs: the model activates its SP strategy from
        ``mesh.ulysses_size > 1``, and the dataloader slices by ``mesh.data_world_size`` so SP peers
        receive identical samples. ``build_hf_device_mesh`` returns None for sp<=1 or non-local mode,
        preserving the deliberate no-mesh local path bit-for-bit. Runs AFTER :meth:`prepare`:
        ``_check_hf_sequence_parallel`` has already rejected the combinations where this mesh would
        be meaningless or harmful (megatron backend, rlhf, ray, fsdp, non-divisible world).
        """
        from swift.dev.builders import build_hf_device_mesh

        self.sp_mesh = build_hf_device_mesh(self.distributed_config, self.template_config.sequence_parallel_size)
        return self.sp_mesh

    def require_task_type(self) -> str:
        """Default ``ModelConfig.task_type`` to this recipe's ``task``, and refuse a conflicting one.

        Defaulting rather than demanding: reaching a recipe already states the intent, while a
        ``'causal_lm'`` task_type on (say) the embedding path would silently encode SFT-shaped rows
        that the contrastive loss cannot group. An explicitly different value is a user error.
        """
        task_type = self.model_config.task_type or self.task
        if task_type != self.task:
            raise ValueError(f'{self.recipe} requires ModelConfig.task_type={self.task!r} (or None), got '
                             f'{task_type!r}. Other task types encode a different row layout and reach a '
                             f'different loss.')
        return task_type

    def build_template(self) -> Any:
        """Build the processor and the swift template the dataset will encode with.

        ``task_type`` is passed explicitly because it is normally read off ``model_info.task_type``,
        which a ``load_model=False`` processor never populates -- it would default to 'causal_lm' and
        encode single-sequence rows instead of this task's layout.
        """
        from swift.dev.builders import build_template, load_model_processor

        # TODO: refactor to get only processor
        _, self.processor = load_model_processor(self.model_config)
        kwargs = {'task_type': self.task_type} if self.task_type else {}
        self.template = build_template(self.template_config, self.processor, **kwargs)
        # Match legacy save_args: keep one run-level args.json that every periodic/final checkpoint copies.
        # This makes --load_args work for a checkpoint produced before training reaches checkpoint-final.
        self.write_ckpt_args_json(
            self.output_dir,
            self.processor,
            self.model_config,
            self.template_config,
            self.tuner_config,
            self.quantize_config,
            task_type=self.task_type)
        return self.template

    def build_dataset(self, **kwargs) -> Any:
        """Build train + eval dataloaders (``list[InputFeature]``; twinkle's processor collates).

        One call loads train + val (split-off or a separate val_dataset) with a single load_dataset;
        either loader may be None. ``template_config`` is passed so ``DatasetConfig.cached_dataset``
        (pre-encoded splits written by ``swift export --to_cached_dataset``) gets legacy's max_length /
        truncation_strategy='delete' length filter applied on load.
        """
        from swift.dev.builders import build_dataset

        self.dataloader, self.eval_dataloader = build_dataset(
            self.dataset_config,
            self.template,
            self.train_config,
            distributed_config=self.distributed_config,
            template_config=self.template_config,
            **kwargs)
        return self.dataloader

    def plan_steps(self) -> int:
        """The optimizer-step budget, with the zero-step fail-fast.

        ``N <= ga`` micro-batches yield 0 optimizer steps (the GA gate lags one step, so a dataset that
        never fills a full lagged window would run forward/backward but NEVER update the model).
        Rather than silently run a no-op training, fail with an actionable message. Checked BEFORE
        :meth:`build_model` so we fail before loading heavy weights.
        """
        from swift.dev.recipe.train_loop import num_optimizer_steps

        if self.train_config.max_steps and self.train_config.max_steps > 0:
            self.total_opt_steps = self.train_config.max_steps
        else:
            try:
                micro_per_epoch = len(self.dataloader)  # micro-batches (already /batch_size) per epoch
            except TypeError:
                micro_per_epoch = 0  # IterableDataset: caller must set max_steps
            total_micro = math.ceil(micro_per_epoch * self.train_config.num_train_epochs)
            self.total_opt_steps = num_optimizer_steps(total_micro, self.ga)

        if self.total_opt_steps <= 0:
            raise ValueError(f'{self.recipe}: computed {self.total_opt_steps} optimizer steps -- the dataloader is '
                             f'too small for gradient_accumulation_steps={self.ga}, or it is a streaming/iterable '
                             f'dataset with no max_steps. Set TrainConfig.max_steps explicitly, or provide enough '
                             f'data.')

        self.resolve_step_intervals(self.total_opt_steps)
        return self.total_opt_steps

    def resolve_step_intervals(self, total_steps: int) -> None:
        """Resolve the eval/save intervals into the concrete optimizer-step counts the loop uses."""
        self.train_config.eval_steps = self._resolve_eval_steps(total_steps)
        if self.checkpoint_config is not None:
            self.checkpoint_config.save_steps = _resolve_step_interval(
                self.checkpoint_config.save_steps, total_steps, 'save_steps')

    def _resolve_eval_steps(self, total_steps: int) -> Optional[int]:
        """Turn ``eval_strategy`` into the optimizer-step interval the loop counts evals against.

        ``process_configs`` already chose the strategy (defaulting it to ``save_strategy`` and forcing
        'no' when the run has nothing to evaluate); this resolves it to a number:
          - 'no': no periodic eval.
          - 'epoch': one eval per epoch, i.e. the per-epoch share of the step budget.
          - 'steps' -- or a bare ``--eval_steps`` with no strategy -- the configured interval: a ratio
            in (0, 1) against ``total_steps``, else an absolute count.
        """
        strategy = (self.train_config.eval_strategy or '').lower()
        if strategy == 'no':
            return None
        if strategy == 'epoch':
            epochs = self.train_config.num_train_epochs or 1.0
            return max(1, math.ceil(total_steps / epochs))
        return _resolve_step_interval(self.train_config.eval_steps, total_steps, 'eval_steps')

    def build_model(self) -> Any:
        """Build the model, apply the tuner, and install dev's processor + template on it.

        The order is forced by twinkle and is the reason this lives in one place:

        - Full-param resume loads weights from the ckpt dir instead of the original model id,
          mirroring legacy's ``_init_ckpt_dir``. LoRA and ``resume_only_model`` keep the original id:
          the base weights are unchanged and only the adapter / optimizer state is restored later.
        - ``apply_tuner`` MUST precede the loss/optimizer configuration so those target the adapter's
          optimizer group (twinkle's add_adapter_to_model creates and activates it). On LoRA resume it
          still runs first, to make the model a PeftModel before the saved adapter is loaded into it.
        - ``set_processor`` gets the CLASS, not an instance: twinkle injects device_mesh (+ framework)
          through construct_class, which returns an instance *unchanged* and silently drops those
          kwargs -- an instance would lose the device_mesh that CP/SP splitting needs.
        - ``set_template`` gets an INSTANCE on purpose, so the configured template survives
          construct_class. Installing it is what makes ONE template (swift's, carrying every
          TemplateConfig field) produce the training tokens; without it the model falls back to a bare
          twinkle Template built from model_id alone, and ``--system`` was silently ignored.
        """
        import copy

        from swift.dev.adapter import apply_tuner
        from swift.dev.builders import apply_full_param_freeze, build_model
        from swift.dev.processor import InputProcessor

        resume_dir = self.resume_dir
        # Full-parameter checkpoints always provide the model weights, including resume_only_model;
        # that flag only suppresses optimizer/scheduler/RNG restoration. LoRA keeps the original base
        # and reloads the adapter through resume_from_checkpoint after apply_tuner.
        redirect_to_ckpt = bool(resume_dir) and self.tuner_config is None
        model_config = self.model_config
        if redirect_to_ckpt:
            model_config = copy.copy(model_config)
            model_config.model = resume_dir

        self.model = build_model(
            model_config,
            self.distributed_config,
            self.train_config,
            self.tuner_config,
            device_mesh=self.sp_mesh,
            quantize_config=self.quantize_config,
            megatron_config=self.megatron_config,
            moe_config=self.moe_config)
        if self.tuner_config is not None:
            # task_type is threaded so a seq_cls/reranker LoRA run adds the freshly-built
            # classification head to modules_to_save -- LoRA trains/saves target_modules only, so
            # without it the head is neither trained nor persisted (checkpoint reloads random).
            apply_tuner(self.model, self.tuner_config, gradient_accumulation_steps=self.ga, task_type=self.task_type)
        else:
            # Full-parameter path: consume the freeze_parameters / trainable_parameters knobs here,
            # before configure_optimizer reads requires_grad to build its param groups. An adapter run
            # skips this -- it selects what trains through target_modules / modules_to_save instead.
            apply_full_param_freeze(self.model, self.train_config)
        self.model.set_processor(
            InputProcessor,
            padding_free=self.template_config.padding_free,
            cp_partition_mode=self.distributed_config.cp_partition_mode,
            task_type=self.task_type)
        self.model.set_template(self.template)
        return self.model

    def build_loop(self) -> Any:
        """Build the training loop and, if resuming, restore state into it.

        Resume order is twinkle-locked: the optimizer must already exist, so
        ``model.resume_from_checkpoint`` (which restores optim/sched/RNG/cur_step) runs after
        ``configure_optimizer`` and before ``fit``; ``loop.resume`` then seeds the loop counters and
        the dataloader skip position. LoRA passes ``adapter_name`` because twinkle defaults it to ''
        while ``apply_tuner`` created 'default'.
        """
        from swift.dev.callbacks import build_callbacks
        from swift.dev.optimizer import resolve_max_grad_norm
        from swift.dev.recipe.train_loop import SFTLoop

        loop_task = self.loop_task or self.task
        kwargs = {'task': loop_task} if loop_task else {}
        self.loop = SFTLoop(
            self.model,
            self.dataloader,
            max_steps=self.total_opt_steps,
            num_train_epochs=self.train_config.num_train_epochs,
            gradient_accumulation_steps=self.ga,
            max_grad_norm=resolve_max_grad_norm(self.train_config),
            logging_config=self.logging_config,
            output_dir=self.output_dir,
            eval_dataloader=self.eval_dataloader,
            eval_steps=self.train_config.eval_steps,
            eval_iters=self.train_config.eval_iters,
            save_steps=self.checkpoint_config.save_steps if self.checkpoint_config else None,
            no_save_optim=bool(self.checkpoint_config
                               and (self.checkpoint_config.no_save_optim or self.checkpoint_config.save_only_model)),
            no_save_rng=bool(self.checkpoint_config
                             and (self.checkpoint_config.no_save_rng or self.checkpoint_config.save_only_model)),
            safe_serialization=(self.checkpoint_config.safe_serialization if self.checkpoint_config else True),
            max_shard_size=(self.checkpoint_config.max_shard_size if self.checkpoint_config else '5GB'),
            save_total_limit=(self.checkpoint_config.save_total_limit if self.checkpoint_config else None),
            ignore_data_skip=bool(self.checkpoint_config and self.checkpoint_config.ignore_data_skip),
            manual_gc=bool(self.megatron_config and self.megatron_config.manual_gc),
            manual_gc_eval=bool(self.megatron_config and self.megatron_config.manual_gc_eval),
            manual_gc_steps=self.megatron_config.manual_gc_steps if self.megatron_config else 0,
            callbacks=build_callbacks(self.train_config),
            eval_delay=int(self.train_config.eval_delay or 0),
            eval_on_start=self.train_config.eval_on_start,
            metric_for_best_model=self.train_config.metric_for_best_model,
            greater_is_better=self.train_config.greater_is_better,
            load_best_model_at_end=self.train_config.load_best_model_at_end,
            best_model_adapter_name=(None if self.tuner_config is None else 'default'),
            hub_pusher=self._build_hub_pusher(),
            hub_strategy=(self.checkpoint_config.hub_strategy if self.checkpoint_config else 'every_save'),
            **self._generative_eval_kwargs(),
            **kwargs)

        if self.resume_dir:
            self.loop.resume(self.resume_model())
        return self.loop

    def _generative_eval_kwargs(self) -> Dict[str, Any]:
        """The loop kwargs for in-training generative eval; empty when the run scores val-loss instead.

        Builds the resident EvalScope sampler once (design 1.4) plus the inputs the loop hands to
        ``run_evalscope``: the normalized benchmark names, the report model id, and the TaskConfig
        kwargs. The sampler backend decides whether a colocate memory schedule brackets each run.
        """
        if not self.train_config.predict_with_generate:
            return {}
        from swift.dev.eval import build_task_config, model_name, validate_eval_datasets

        cfg = self.train_config
        eval_sampler, eval_enter, eval_exit = self._build_eval_sampler()
        return {
            'predict_with_generate': True,
            'eval_sampler': eval_sampler,
            'eval_template': self.template,
            'eval_datasets': validate_eval_datasets(cfg.eval_dataset),
            'eval_model_id': model_name(self.model_config),
            # TrainConfig carries no eval_output_dir / eval_num_proc (those belong to the standalone
            # ``swift eval`` EvalConfig), so the work dir is derived from the run's output dir and the
            # request concurrency is left to EvalScope's default.
            'eval_task_config': build_task_config(
                work_dir=os.path.join(self.output_dir, 'eval'),
                limit=cfg.eval_limit,
                dataset_args=cfg.eval_dataset_args,
                generation_config=cfg.eval_generation_config,
                extra_eval_args=cfg.extra_eval_args),
            'eval_enter': eval_enter,
            'eval_exit': eval_exit,
        }

    def _build_eval_sampler(self) -> Tuple[Any, Optional[Callable], Optional[Callable]]:
        """Build the resident generative-eval sampler and its optional colocate memory schedule.

        Returns ``(eval_sampler, eval_enter, eval_exit)``. Two backends, per
        ``TrainConfig.eval_sampler_backend``:

        - 'transformers' (default): a facade over the live training module. ``TransformersSampler(model=..)``
          holds no weights of its own -- it generates on the model's workers, on the weights they already
          have -- so there is no second copy and no weight sync, and it is built on the driver with no
          remote_group. No memory schedule is needed, so eval_enter/eval_exit are None.
        - 'vllm' / 'sglang': a co-resident engine placed in the trainer's 'model' DeviceGroup and
          weight-synced through a CheckpointEngineManager (ray mode only -- validate.py refuses it
          otherwise). eval_enter/eval_exit run the colocate device hand-over around each eval so the
          engine and the trainer take turns on the GPU, mirroring the GRPO rollout schedule.
        """
        backend = self.train_config.eval_sampler_backend
        if backend == 'transformers':
            from twinkle.sampler import TransformersSampler
            sampler = TransformersSampler(model=self.model)
            sampler.set_template(self.template)
            return sampler, None, None

        from twinkle.checkpoint_engine import CheckpointEngineManager

        from swift.dev.builders import build_ray_dp_mesh, build_sampler
        # The colocate hand-over sleeps the engine to free the GPU for the trainer, which each backend
        # only permits when it was built with its own memory-saver flag on (vLLM: enable_sleep_mode,
        # sglang: enable_memory_saver); without it sleep()/wake_up() warn and no-op, leaving the trainer
        # offloaded and the engine without a KV cache.
        sleep_arg = 'enable_sleep_mode' if backend == 'vllm' else 'enable_memory_saver'
        sampler = build_sampler(
            self.model_config,
            backend=backend,
            engine_args={sleep_arg: True},
            template=self.template,
            # Co-resident in the model's 'model' DeviceGroup (colocate), so it carries the SAME pure-DP
            # mesh the training model was placed with: a Ray-remote twinkle object built with
            # device_mesh=None and no global default is rejected, and a colocated engine that sliced its
            # sample inputs across a different rank layout than the weights it syncs would be wrong.
            device_mesh=build_ray_dp_mesh(self.distributed_config),
            remote_group='model')
        manager = CheckpointEngineManager(model=self.model, sampler=sampler, platform='GPU', mode='colocate')

        def eval_enter() -> None:
            # The manager's documented colocate order: wake the engine's weights, sync the trained policy
            # into them, step the trainer aside, then give the engine its KV cache so it can generate.
            sampler.wake_up(tags=['weights'])
            manager.sync_weights(merge_and_sync=True)
            self.model.offload_to_cpu()
            sampler.wake_up()

        def eval_exit() -> None:
            sampler.sleep()
            self.model.reload_to_gpu()

        return sampler, eval_enter, eval_exit

    def _build_hub_pusher(self) -> Optional[Callable[[str], None]]:
        """A callable that uploads one checkpoint dir to the hub, or None when pushing is off.

        Built here -- the assembly owns the hub target and credentials -- and injected into the loop,
        which fires ``on_push_begin`` then calls it after a save (main process only). Goes through
        twinkle's ``HubOperation`` rather than ``TrainableModel.upload_to_hub``: the latter hardcodes
        ``private=True`` and never passes a revision, so it would silently ignore ``hub_private_repo`` /
        ``hub_revision``. ``HubOperation.push_to_hub`` honors both and dispatches ModelScope vs HF off the
        repo_id prefix (``hf://`` / ``ms://``).
        """
        cfg = self.checkpoint_config
        if cfg is None or not cfg.push_to_hub:
            return None
        if not cfg.hub_model_id:
            raise ValueError('CheckpointConfig.push_to_hub is set but hub_model_id is empty, so there is no '
                             'target repo to upload to. Pass --hub_model_id <repo_id>.')
        from twinkle.hub import HubOperation

        token = self.dataset_config.hub_token if self.dataset_config is not None else None
        repo_id = cfg.hub_model_id
        private = cfg.hub_private_repo
        revision = cfg.hub_revision or 'master'
        # hub_always_push blocks until each upload lands, for a run where a dropped push would go
        # unnoticed; off by default so uploads run on a background thread and do not stall training.
        async_upload = not cfg.hub_always_push

        def push(checkpoint_dir: str) -> None:
            push_kwargs = dict(
                repo_id=repo_id,
                folder_path=checkpoint_dir,
                token=token,
                private=private,
                revision=revision,
                commit_message=f'Upload {os.path.basename(checkpoint_dir.rstrip("/"))} from swift.dev training')
            if async_upload:
                HubOperation.async_push_to_hub(**push_kwargs)
            else:
                HubOperation.push_to_hub(**push_kwargs)

        return push

    def resume_model(self, model: Any = None, checkpoint_dir: Optional[str] = None, *,
                     adapter_name: Optional[str] = None) -> dict:
        """Restore one trainable model and return twinkle's training-progress state.

        Custom RLHF loops use the same checkpoint semantics as :meth:`build_loop`. PPO passes its
        critic explicitly with ``checkpoint_dir=<policy checkpoint>/value_model``; the policy uses
        the assembly defaults.
        """
        from swift.dev.builders import is_megatron_backend

        target = model or self.model
        path = checkpoint_dir or self.resume_dir
        if not path:
            return {}
        resume_kwargs = {'resume_only_model': self.resume_only_model}
        if self.checkpoint_config is not None and is_megatron_backend(self.distributed_config):
            resume_kwargs.update(
                no_load_optim=self.checkpoint_config.no_load_optim,
                no_load_rng=self.checkpoint_config.no_load_rng,
                data_parallel_random_init=bool(self.megatron_config and self.megatron_config.data_parallel_random_init),
            )
        if adapter_name is None and target is self.model and self.tuner_config is not None:
            adapter_name = 'default'
        if adapter_name is not None:
            resume_kwargs['adapter_name'] = adapter_name

        trainer_state_path = os.path.join(path, 'trainer_state.json')
        if not os.path.isfile(trainer_state_path):
            if not self.resume_only_model:
                raise FileNotFoundError(
                    f'{trainer_state_path} is missing, so this model-only checkpoint cannot resume optimizer, '
                    'scheduler, RNG, or progress state. Set --resume_only_model true to start a new run from its '
                    'weights, or resume from a checkpoint saved without --no_save_optim/--save_only_model.')
            # Full-parameter models were constructed from ``path`` by build_model(). A tuned policy still
            # needs its adapter weights loaded explicitly; no trainer state means this is a new run at step 0.
            if adapter_name:
                target.load(path, adapter_name=adapter_name)
            return {
                'cur_step': 0,
                'consumed_train_samples': 0,
                'gradient_accumulation_steps': self.ga,
            }
        return target.resume_from_checkpoint(path, **resume_kwargs)

    @property
    def resume_dir(self) -> Optional[str]:
        return self.checkpoint_config.resume_from_checkpoint if self.checkpoint_config else None

    @property
    def resume_only_model(self) -> bool:
        return bool(self.checkpoint_config and self.checkpoint_config.resume_only_model)

    def fit(self, configure_loss: Callable[[Any], None], *, save_final: bool = True) -> List[dict]:
        """Run every stage in the locked order and train. Returns the loss/grad_norm history.

        ``configure_loss`` is the one thing a training recipe genuinely owns -- which objective this
        run optimises -- so it arrives as a callable taking the built model. It is invoked between
        the model and the optimizer because the optimizer must see the loss's parameters (a loss may
        add its own), and after the tuner for the optimizer-group reason above.

        ``save_final`` writes a final checkpoint after training (periodic saves are governed by
        ``save_steps``). Passing False is test-oriented: it yields the loss trajectory without a
        checkpoint, which also sidesteps the Megatron mode='local' distributed-save gap.
        """
        from swift.dev.optimizer import configure_optimizer

        self.prepare()
        self.plan_sp_mesh()
        if self.task:
            self.require_task_type()
        self.build_template()
        self.build_dataset(device_mesh=self.sp_mesh)
        self.plan_steps()
        self.build_model()
        configure_loss(self.model)
        configure_optimizer(
            self.model,
            self.train_config,
            num_training_steps=self.total_opt_steps,
            distributed_config=self.distributed_config)
        self.build_loop()

        history = self.loop.fit()
        if save_final:
            self.save_final()
        return history

    def save_final(self, name: str = 'checkpoint-final') -> str:
        """Persist the final checkpoint plus the args.json ``swift infer`` reads back.

        The path is recomputed rather than taken from ``loop.save``: in Ray (Megatron) mode save()
        returns a deferred handle, not a path.
        """
        self.loop.save(name, is_final=True)
        ckpt_dir = os.path.join(self.output_dir, name)
        TrainAssembly.write_ckpt_args_json(
            ckpt_dir,
            self.processor,
            self.model_config,
            self.template_config,
            self.tuner_config,
            self.quantize_config,
            task_type=self.task_type)
        return ckpt_dir

    @staticmethod
    def write_ckpt_args_json(ckpt_dir: str,
                             processor: Any,
                             model_config: 'ModelConfig',
                             template_config: 'TemplateConfig',
                             tuner_config: Optional['TunerConfig'] = None,
                             quantize_config: Optional['QuantizeConfig'] = None,
                             *,
                             task_type: Optional[str] = None) -> None:
        """Write the self-describing args.json swift infer reads back from the ckpt.

        Legacy save_args (base_args.py:303-310) dumps the FULL argument dict; infer's read side
        load_args_from_ckpt (base_args.py:246-301) only consumes two lists:
          - force_load_keys (always applied): tuner_type, task_type, bnb_4bit_* -- a MISSING key here
            silently leaves infer on its default (task_type='causal_lm', no adapter), degrading
            seq_cls / reranker / LoRA checkpoints.
          - load_keys (applied only when the current value is None/empty): model, model_type,
            model_revision, torch_dtype, attn_impl, template, system, truncation_strategy, ...
        We write that consumed subset (not the full dict): the force-loaded tuner/task/BNB keys plus
        the load-if-empty model, template, and quantization keys dev already knows.

        Master-only: the checkpoint dir exists on the master rank alone (twinkle's save_pretrained
        guards on Platform.is_master()), so an unguarded open() elsewhere raised FileNotFoundError
        under multi-GPU DDP.

        ``task_type`` overrides ``ModelConfig.task_type`` when the latter is unset: the recipe implies
        it (reaching run_embedding IS the declaration), so a run that never spelled it out would
        otherwise omit a force_load key and leave ``swift infer <ckpt>`` on causal_lm.
        """
        import json
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized() and dist.get_rank() != 0:
            return
        os.makedirs(ckpt_dir, exist_ok=True)

        from swift.dev.version import __version__ as swift_version
        model_meta = getattr(processor, 'model_meta', None)
        args = {
            # swift_version gates model_type loading in BaseArguments.load_args_from_ckpt
            # (model_type is only honored when swift_version >= 4.0.0.dev); must be present.
            'swift_version': swift_version,
            # force_load_keys: infer applies these regardless of its current value.
            'task_type': model_config.task_type or task_type,
            'tuner_type': tuner_config.tuner if tuner_config is not None else 'full',
            'quant_method': quantize_config.quant_method if quantize_config is not None else None,
            'quant_bits': quantize_config.quant_bits if quantize_config is not None else None,
            'bnb_4bit_compute_dtype': (
                quantize_config.bnb_4bit_compute_dtype if quantize_config is not None else None),
            'bnb_4bit_quant_type': quantize_config.bnb_4bit_quant_type if quantize_config is not None else None,
            'bnb_4bit_use_double_quant': (
                quantize_config.bnb_4bit_use_double_quant if quantize_config is not None else None),
            'bnb_4bit_quant_storage': quantize_config.bnb_4bit_quant_storage if quantize_config is not None else None,
            # load_keys: infer applies these only when its own value is None/empty.
            # Record the portable id/path the user passed, not the machine-local snapshot dir
            # _resolve_model substituted for loading: legacy saves the id, and `swift infer <ckpt>`
            # re-resolves it on any machine. Falls back to .model when no resolution happened.
            'model': getattr(model_config, '_model_id_or_path', None) or model_config.model,
            'model_type': getattr(model_meta, 'model_type', None),
            'model_revision': model_config.model_revision,
            'torch_dtype': model_config.torch_dtype,
            'attn_impl': model_config.attn_impl,
            'template': template_config.template or getattr(model_meta, 'template', None),
            'system': template_config.system,
            'truncation_strategy': template_config.truncation_strategy,
            'max_length': template_config.max_length,
        }
        args = {k: v for k, v in args.items() if v is not None}
        with open(os.path.join(ckpt_dir, 'args.json'), 'w', encoding='utf-8') as f:
            json.dump(args, f, ensure_ascii=False, indent=2)
