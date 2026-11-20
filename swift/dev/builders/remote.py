# Copyright (c) ModelScope Contributors. All rights reserved.
"""mode='remote' object sources: local builders yield in-process objects, these yield
twinkle_client handles over HTTP.

The remote shape (e2e-verified by examples/server/client.py): the dataset and template stay
in THIS process (build_dataset/build_template are reused; the InputFeature rows they emit
cross the wire as forward inputs), only the model is a MultiLoraTransformersModel handle.
That handle exposes the same method face as the local TrainableModel, with arguments
narrowed to "class name + serializable kwargs" -- the str branch of the local
``Union[instance, type, str]`` signatures. Loss and tuner therefore need remote variants
(the local ones pass peft/Loss instances); configure_optimizer's transformers branch is
already a description ("set_optimizer(name, scalar kwargs)") and is reused as is.

The server-side processor is swift's own dev one, registered into ``twinkle.processor`` at
worker startup via ``TWINKLE_WORKER_SETUP_HOOK``; the client asks for it by class NAME and
ships a serialized ``TemplateSpec`` so the server rebuilds the dev Template itself.
"""
from __future__ import annotations
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, Optional
from uuid import uuid4

from swift.dev.utils.logger import get_logger

if TYPE_CHECKING:
    from swift.dev.config import DistributedConfig, ModelConfig, TemplateConfig, TrainConfig, TunerConfig

logger = get_logger()

#: Micro-batching knobs twinkle's forward-only micro-batcher refuses for a readout task whose model
#: carries an attached head. The remote multi-LoRA host ALWAYS carries per-adapter head slots, so every
#: remote readout forward (seq_cls / value / reranker / decision) must run whole-batch.
_READOUT_MICRO_BATCH_KEYS = ('micro_batch_size', 'max_tokens_per_micro_batch')


def _remote_forward_kwargs(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize forward kwargs onto what the remote readout host supports.

    twinkle sample-splits a micro-batched forward, but a readout task's labels/decision_meta are
    organized per RECORD, so the split desynchronizes them from the logits; the backend raises rather
    than degrade. The local legacy in-model head (carries_task_head=False) merges per sample and keeps
    micro-batching, so the mode-blind recipe code legitimately passes micro_batch_size for seq_cls/value
    eval and the PPO critic. The restriction is a property of the REMOTE attached-head host, so the
    normalization lives here at the remote boundary -- one place that covers eval, the critic, and any
    future readout forward -- instead of a per-call-site guard. Whole-batch is the only supported remote
    readout mode (the recipe's eval chunking still bounds memory), so dropping the knobs is correct, not
    a silent degradation.
    """
    from twinkle.processor.tasks import get_task_spec

    if get_task_spec(kwargs.get('task', 'causal_lm')).readout:
        return {key: value for key, value in kwargs.items() if key not in _READOUT_MICRO_BATCH_KEYS}
    return kwargs


def build_remote_model(model_config: 'ModelConfig', distributed_config: 'DistributedConfig') -> Any:
    """Connect to the twinkle-server and return the client-side trainable model handle.

    ``model`` names the SERVER-side served model id (e.g. 'ms://Qwen/Qwen2.5-7B-Instruct');
    the handle strips the scheme itself. The server must already run it as a multi-LoRA host
    (``swift deploy`` / a server_config.yaml with a model app); this function never starts one.
    """
    from twinkle_client.model import MultiLoraTransformersModel

    from twinkle import init_twinkle_client

    client = init_twinkle_client(base_url=distributed_config.server_base_url, api_key=distributed_config.server_api_key)
    handle = MultiLoraTransformersModel(model_config.model or '')
    logger.info(f"[remote] driving server {distributed_config.server_base_url!r} model {model_config.model!r}")
    return RemoteModel(handle, client)


class RemoteModel:
    """Normalizes the twinkle_client handle's RESPONSES onto the local TrainableModel face.

    The handle mirrors the local method signatures (narrowed to class-name + scalar kwargs), but
    replies with protocol Response objects where the local model returns plain values:
    ``calculate_metric`` / ``resume_from_checkpoint`` wrap their dict payload in a ``result``
    field. The shared recipe code is mode-blind and reads those returns as dicts, so the unwrap
    lives here with the other remote object sources instead of at every consumption point.
    """

    def __init__(self, handle: Any, client: Any = None) -> None:
        self._handle = handle
        self.client = client
        # The shared loop sizes its per-rank micro-batches off ``device_mesh.data_world_size``;
        # remotely ONE client process drives the server, whose own sharding splits the batch it
        # receives. A single-process mesh (dp=1) keeps every division correct.
        from twinkle import DeviceMesh

        self.device_mesh = DeviceMesh.from_sizes(dp_size=1)

    def __getattr__(self, name: str) -> Any:
        # forward_backward / set_* / add_adapter_to_model / save / ... delegate unchanged.
        return getattr(self.__dict__['_handle'], name)

    def calculate_metric(self, is_training: bool = True, **kwargs: Any) -> Any:
        response = self._handle.calculate_metric(is_training=is_training, **kwargs)
        return getattr(response, 'result', response)

    def resume_from_checkpoint(self, path: str, **kwargs: Any) -> Any:
        response = self._handle.resume_from_checkpoint(path, **kwargs)
        return getattr(response, 'result', response)

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        # The local face returns the ModelOutput dict directly; the client wraps it in
        # ForwardResponse.result. Downstream consumers (advantage _lane) call .get(field) on it.
        response = self._handle.forward(*args, **_remote_forward_kwargs(kwargs))
        return getattr(response, 'result', response)

    def forward_only(self, *args: Any, **kwargs: Any) -> Any:
        response = self._handle.forward_only(*args, **_remote_forward_kwargs(kwargs))
        return getattr(response, 'result', response)

    def forward_backward(self, *args: Any, **kwargs: Any) -> Any:
        return self._handle.forward_backward(*args, **_remote_forward_kwargs(kwargs))

    def wait_async_saves(self, *args: Any, **kwargs: Any) -> None:
        # Remote saves are synchronous (the client blocks to the terminal state), so there are no
        # pending async saves to wait for; the local decorated method has no client-side counterpart.
        return None


@dataclass(frozen=True)
class RemoteLossSpec:
    """A loss as (class NAME, construction kwargs) -- what crosses HTTP.

    The webui canvas wires a loss OBJECT on its LOSS channel; over HTTP only the name and the
    (scalar) construction kwargs travel, so the remote canvas loss component produces this spec
    and its train_step consumer calls ``set_loss(name, **kwargs)`` on the client handle. A custom
    user formula cannot take this form and is refused at the component instead. ``require_values``
    carries the value-task flag the local object exposes as an attribute (the PPO critic's
    forward_backward must run the value task).
    """

    name: str
    kwargs: Dict[str, Any] = field(default_factory=dict)
    require_values: bool = False


def apply_remote_tuner(model: Any, tuner_config: Optional['TunerConfig'],
                       gradient_accumulation_steps: int,
                       task_type: Optional[str] = None,
                       model_config: Optional['ModelConfig'] = None,
                       adapter_name: str = 'default') -> None:
    """Attach the tuner on the server: LoRA becomes a named adapter on the resident base model.

    trainable_tokens has no server-side counterpart yet, and 'full' needs no adapter call (the
    server model trains all parameters by default) -- both fail loudly rather than degrade.
    For a task with a readout head (seq_cls / reranker / decision) the adapter call also builds
    the head server-side: the server model is a plain causal-lm host, and
    ``add_adapter_to_model(task=...)`` is the channel that attaches heads to it (the local path
    instead resolves a head-carrying model_cls or a family loader at build time).
    """
    from peft import LoraConfig

    if tuner_config is None:
        return
    if tuner_config.tuner == 'full':
        return
    if tuner_config.tuner != 'lora':
        raise ValueError(
            f"--mode remote does not support tuner={tuner_config.tuner!r}: the server-side adapter "
            "mechanism is LoRA (a named adapter on the resident base). Use --tuner lora, or drop "
            "--tuner for full-parameter remote training.")
    if tuner_config.target_parameters or tuner_config.trainable_token_indices:
        raise ValueError(
            '--mode remote does not support target_parameters / trainable_token_indices: they need '
            'the in-process module graph (parameter objects / embedding tensors) that cannot cross '
            'HTTP. Drop the knobs, or run with mode=local/ray.')
    # The SAME LoraConfig construction the local path uses (all-linear expansion, DoRA/bias flags, init
    # resolution): a hand-rolled narrower config here silently dropped those knobs and left the default
    # ['all-linear'] list unexpanded, which peft only expands for the bare string. BUT pass task_type=None
    # so _lora_common_kwargs skips the seq_cls/reranker head-module injection: the server multi-tenant host
    # rejects modules_to_save (add_adapter_to_model) because a readout head is a HOST-CONSTRUCTION property
    # (the deployment YAML's task_type, bound per-tenant through the task= kwarg below), not a peft
    # modules_to_save injection -- that is the local in-process path. Refuse an explicit modules_to_save
    # loudly rather than let the server reject it opaquely.
    from swift.dev.builders.adapter import _lora_common_kwargs

    if tuner_config.modules_to_save:
        raise ValueError(
            '--mode remote does not support --modules_to_save: the server multi-tenant host binds a '
            'seq_cls/reranker readout head from its deployment task_type (the host YAML), not from peft '
            'modules_to_save. Drop --modules_to_save; the task head rides --task_type + the server host.')
    lora_config = LoraConfig(**_lora_common_kwargs(tuner_config, task_type=None))
    # No save_dir: the server owns the checkpoint store (a client-side dir would have to exist
    # on the server already). GA rides the adapter because the server's grad-sync gate is
    # per-adapter state, not a forward kwarg. The head kwargs are plain names/ints, so they
    # cross HTTP with the rest of the call.
    training_id = model.add_adapter_to_model(adapter_name, lora_config, gradient_accumulation_steps=gradient_accumulation_steps,
                                             **_remote_task_head_kwargs(task_type, model_config))
    # The server-side run id reaches the log on purpose: the webui executor parses it to map
    # THIS task onto the server's training-run APIs (checkpoint list / prune / explicit stop).
    logger.info(f'[remote] server training created: id={training_id} adapter={adapter_name!r}')


def _remote_task_head_kwargs(task_type: Optional[str], model_config: Optional['ModelConfig']) -> Dict[str, Any]:
    """The ``add_adapter_to_model`` kwargs that build the task's readout head server-side.

    No head at all for causal_lm / embedding / generative_reranker (pooling or the vocab head
    already serve the task). seq_cls and reranker build the standard Linear head (reranker is a
    single-score one); decision replays the family loader's build through its (module, class)
    NAME -- the same names the local Megatron path ships across the Ray boundary.
    """
    if task_type in (None, 'causal_lm', 'embedding', 'generative_reranker'):
        return {}
    if task_type == 'seq_cls':
        if model_config is None or model_config.num_labels is None:
            raise ValueError('ModelConfig.num_labels is required for task_type="seq_cls".')
        return {'task': 'seq_cls', 'num_labels': model_config.num_labels}
    if task_type == 'reranker':
        num_labels = getattr(model_config, 'num_labels', None) if model_config is not None else None
        return {'task': 'reranker', 'num_labels': num_labels or 1}
    if task_type == 'decision':
        if model_config is None:
            raise ValueError('task_type="decision" needs the ModelConfig to resolve its family loader.')
        from swift.dev.builders.model import _resolve_model_loader
        from swift.dev.model.loader.decision import ScoringLoader

        model_loader = _resolve_model_loader(model_config)
        if not isinstance(model_loader, ScoringLoader):
            raise ValueError(
                f'task_type="decision" needs a decision family loader; model_type='
                f'{model_config.model_type!r} resolved {model_loader!r}.')
        return {'task': 'decision', 'task_head_loader': model_loader.task_head_loader()}
    raise ValueError(f'--mode remote has no readout-head configuration for task_type={task_type!r}.')


def configure_remote_loss(model: Any, train_config: 'TrainConfig',
                           task_type: Optional[str] = None,
                           model_config: Optional['ModelConfig'] = None) -> None:
    """Set the run's loss on the server model by NAME (the instance the local path passes
    cannot cross HTTP).

    causal_lm keeps the supervised main line (cross entropy, channel-grouped, DFT-weighted,
    liger fused). The task heads route by the same swift loss names the local
    ``configure_*_loss`` family resolves -- embedding (infonce & friends), reranker /
    generative_reranker (pointwise / listwise), seq_cls (problem_type x num_labels), decision
    (scoring / distill_rps / smooth_brier / ce_rps) -- and the resolved CLASS name is what
    crosses the wire; a dev-registered loss class is exported into ``twinkle.loss`` by
    ``register_loss``, so the server resolves it the same way this process does. A server
    without the class (e.g. liger not installed there) fails at set_loss time, loudly.
    """
    from swift.dev.loss.configure import DECISION_LOSS_TYPES, EMBEDDING_LOSS_TYPES, PROBLEM_TYPES, RERANKER_LOSS_TYPES
    from swift.dev.utils.naming import resolve_loss

    if _liger_fused_ce_enabled(train_config) and (train_config.enable_channel_loss
                                                  or train_config.enable_dft_loss):
        # Same refusal the local configure_loss makes: the fused kernel computes CE itself, so
        # forwarding dft would be silently swallowed by the loss's **kwargs and the channel name
        # dropped by the elif -- a remote run must not degrade silently where local fails loudly.
        raise ValueError('fused_linear_cross_entropy replaces the standard CE path and cannot be combined '
                         'with enable_channel_loss or enable_dft_loss.')

    task_type = task_type or 'causal_lm'
    kwargs: Dict[str, Any] = {}
    if task_type == 'causal_lm':
        name = 'cross_entropy'
        if _liger_fused_ce_enabled(train_config):
            name = 'liger_fused_linear_cross_entropy'
        elif train_config.enable_channel_loss:
            name = 'channel'
        kwargs = {'reduction': 'sum', 'dft': bool(train_config.enable_dft_loss),
                  'label_smoothing': train_config.label_smoothing_factor or 0.0}
    elif task_type == 'embedding':
        name = train_config.loss or 'infonce'
        if name not in EMBEDDING_LOSS_TYPES:
            raise NotImplementedError(f'configure_embedding_loss supports {list(EMBEDDING_LOSS_TYPES)}, '
                                      f'got {name!r}.')
        if train_config.mrl_dims is not None:
            kwargs['mrl_dims'] = train_config.mrl_dims
    elif task_type in ('reranker', 'generative_reranker'):
        name = train_config.loss or 'pointwise_reranker'
        if name not in RERANKER_LOSS_TYPES:
            raise NotImplementedError(f'configure_reranker_loss supports {list(RERANKER_LOSS_TYPES)}, '
                                      f'got {name!r}.')
    elif task_type == 'seq_cls':
        problem_type = getattr(model_config, 'problem_type', None) if model_config is not None else None
        num_labels = getattr(model_config, 'num_labels', None) if model_config is not None else None
        if problem_type not in PROBLEM_TYPES:
            raise ValueError(f'problem_type must be one of {list(PROBLEM_TYPES)}, got {problem_type!r}. '
                             'It is required (not inferred) so the training objective is explicit.')
        if num_labels is None:
            raise ValueError('ModelConfig.num_labels is required for task_type="seq_cls".')
        name = 'seq_cls'
        kwargs = {'problem_type': problem_type, 'num_labels': num_labels}
    elif task_type == 'decision':
        name = train_config.loss or 'scoring'
        if name not in DECISION_LOSS_TYPES:
            raise ValueError(f'configure_decision_loss supports {list(DECISION_LOSS_TYPES)}, got {name!r}.')
    else:
        raise ValueError(f'--mode remote has no loss configuration for task_type={task_type!r}.')
    loss_cls = resolve_loss(name)
    model.set_loss(loss_cls.__name__, **kwargs)


class RemoteWeightSync:
    """Publish the remote policy's LoRA weights to the sampler over HTTP.

    Async rollout retains named snapshots while sampling requests use them.
    """

    def __init__(self, model: Any, sampler: Any, *, versioned: bool = False) -> None:
        self.model = model
        self.sampler = sampler
        self._version = 0
        self._versioned = versioned
        self._publish_lock = threading.Lock()
        self._snapshot_lock = threading.Lock()
        self._snapshots: dict[str, int] = {}
        self._unconfirmed: dict[str, set[str | None]] = {}
        self._pending_saves: dict[str, str | None] = {}
        self._retiring: set[str] = set()
        self._closed = False
        if versioned:
            if not hasattr(sampler, 'on_adapter_task_unknown') or getattr(model, 'client', None) is None:
                raise RuntimeError('Remote async GRPO requires a Twinkle HTTP sampler with adapter lease support.')
            sampler.adapter_lease = self._lease_adapter
            sampler.on_adapter_task_unknown = self._note_unknown_sample

    @contextmanager
    def _lease_adapter(self, uri):
        with self._snapshot_lock:
            if self._closed:
                raise RuntimeError('Remote weight sync is closed.')
            if uri is None:
                uri = self.sampler.adapter_path
            if uri in self._retiring:
                raise ValueError(f'Remote sampler snapshot is being reclaimed: {uri}')
            tracked = uri in self._snapshots
            if tracked:
                self._snapshots[uri] += 1
        try:
            yield uri
        finally:
            if tracked:
                with self._snapshot_lock:
                    self._snapshots[uri] -= 1
                    reclaim = self._closed
                if reclaim:
                    with self._publish_lock:
                        self._reclaim_snapshots()

    def _note_unknown_sample(self, uri, request_id) -> None:
        with self._snapshot_lock:
            if uri in self._snapshots:
                self._unconfirmed.setdefault(uri, set()).add(request_id)

    def sync_weights(self, merge_and_sync: bool = True) -> None:
        with self._publish_lock:
            if self._closed:
                raise RuntimeError('Remote weight sync is closed.')
            self._publish()

    def _publish(self) -> None:
        import os
        import time

        perf = os.environ.get('TWINKLE_PERF') == '1'
        start = time.perf_counter() if perf else None
        name = f'remote-sampler-weights-{uuid4().hex}' if self._versioned else 'remote-sampler-weights'
        options = {}
        if self._versioned:
            options['on_outcome_unknown'] = lambda request_id: self._pending_saves.__setitem__(name, request_id)
        response = self.model.save(name, save_optimizer=False, is_sampler=True, **options)
        if perf:
            logger.info(f'[remote][perf] weight-sync save round-trip {time.perf_counter() - start:.3f}s '
                        f'version={self._version + 1}')
        uri = getattr(response, 'twinkle_path', None)
        if not uri:
            raise ValueError('the remote weight sync got no twinkle:// path back from model.save().')
        with self._snapshot_lock:
            if hasattr(self.sampler, 'adapter_path'):
                self.sampler.adapter_path = uri
            else:
                self.sampler.adapter_uri = uri
            self._version += 1
            if self._versioned:
                self._snapshots[uri] = 0
        if self._versioned:
            self._reclaim_snapshots(keep=uri)

    def _reclaim_snapshots(self, *, keep=None) -> None:
        self._reconcile_tasks()
        with self._snapshot_lock:
            candidates = [
                uri for uri, count in self._snapshots.items()
                if uri != keep and count == 0 and uri not in self._unconfirmed
            ]
        for uri in candidates:
            with self._snapshot_lock:
                if self._snapshots[uri] or uri in self._unconfirmed:
                    continue
                del self._snapshots[uri]
                self._retiring.add(uri)
            try:
                run_id, kind, name = uri.removeprefix('twinkle://').split('/')
                if not uri.startswith('twinkle://') or kind != 'sampler_weights':
                    raise ValueError(f'Unexpected sampler checkpoint URI: {uri!r}')
                self.model.client.delete_checkpoint(run_id, f'{kind}/{name}')
            except Exception as exc:
                # Failed deletions are retried at the next publication or close.
                logger.warning(f'Could not reclaim remote sampler snapshot {uri}: {exc}')
                with self._snapshot_lock:
                    self._snapshots[uri] = 0
                # The delete may still finish on the server after a transport error.
                # Keep the URI unavailable to new readers while retrying.
            else:
                with self._snapshot_lock:
                    self._retiring.remove(uri)

    def _retrieve_task(self, request_id):
        from requests import RequestException

        if request_id is None:
            return None
        try:
            return self.model.client.retrieve_task(request_id)
        except (RequestException, ValueError):
            return None

    def _reconcile_tasks(self) -> None:
        from twinkle.protocol.types.lifecycle import TERMINAL_STATUSES

        for name, request_id in tuple(self._pending_saves.items()):
            task = self._retrieve_task(request_id)
            if task is None or task.status not in TERMINAL_STATUSES:
                continue
            if task.status == 'completed':
                uri = task.result.get('twinkle_path') if isinstance(task.result, dict) else None
                if not isinstance(uri, str) or not uri.endswith(f'/sampler_weights/{name}'):
                    logger.warning(f'Could not identify the completed sampler snapshot {name}')
                    continue
                with self._snapshot_lock:
                    self._snapshots[uri] = 0
            del self._pending_saves[name]

        with self._snapshot_lock:
            unknown_samples = [(uri, tuple(ids)) for uri, ids in self._unconfirmed.items()
                               if self._snapshots[uri] == 0]
        for uri, request_ids in unknown_samples:
            for request_id in request_ids:
                task = self._retrieve_task(request_id)
                if task is not None and task.status in TERMINAL_STATUSES:
                    with self._snapshot_lock:
                        self._unconfirmed[uri].discard(request_id)
                        if not self._unconfirmed[uri]:
                            del self._unconfirmed[uri]

    def version(self) -> int:
        return self._version

    def close(self) -> None:
        if self._versioned:
            with self._publish_lock:
                with self._snapshot_lock:
                    self._closed = True
                self._reclaim_snapshots()
                if self._pending_saves or self._unconfirmed:
                    logger.warning('Keeping remote sampler snapshots with unresolved server tasks: saves=%s, samples=%s',
                                   self._pending_saves, self._unconfirmed)


def configure_remote_metric(model: Any, task_type: Optional[str] = None) -> None:
    """Register the run's metric on the server model by NAME (classes cannot cross HTTP).

    embedding gets twinkle's own ``EmbeddingMetric`` (the local ``configure_embedding_metric``
    passes the class). seq_cls / reranker / decision have no remote metric registration yet:
    training and the loss/grad numerics are unaffected, only their eval-side readouts are
    absent -- the decision metric family is a swift.dev class pending the same namespace export.
    """
    if task_type == 'embedding':
        model.add_metric('EmbeddingMetric', is_training=True)


def _liger_fused_ce_enabled(train_config: 'TrainConfig') -> bool:
    """Same predicate the local recipe uses to pair the loss with loop_task='fused_lm_ce'."""
    from swift.dev.loss import liger_fused_ce_enabled

    return liger_fused_ce_enabled(train_config)


#: The server-side processor class the remote path asks for by NAME. The server must have imported
#: ``swift.dev.processor.server_plugin`` in every backend worker (``TWINKLE_WORKER_SETUP_HOOK`` in
#: the model app's runtime env); ``register`` then exports this class into ``twinkle.processor``,
#: so no client-supplied plugin path ever crosses the wire.
REMOTE_PROCESSOR_CLS = 'DevInputProcessor'


def configure_remote_processor(model: Any, template_config: 'TemplateConfig', model_config: 'ModelConfig',
                               distributed_config: 'DistributedConfig') -> None:
    """Set the server-side processor to swift's dev one, enabled at server startup.

    The dev Template itself is rebuilt inside the worker from the serialized ``TemplateSpec``
    below, so the per-row stages (position_ids/mrope, bookkeeping drops) run next to the batch
    stages twinkle already owns, and the client no longer finishes rows before shipping them.
    """
    from dataclasses import asdict

    import json

    from swift.dev.builders.template import build_template_spec

    spec = build_template_spec(template_config, model_config)
    # JsonValue on the wire: tuples become lists; TemplateSpec.from_dict accepts either.
    template_spec = json.loads(json.dumps(asdict(spec)))
    model.set_template(
        'swift.dev.template.Template',
        template_spec=template_spec,
    )
    model.set_processor(
        REMOTE_PROCESSOR_CLS,
        template_spec=template_spec,
        padding_free=template_config.padding_free,
        padding_side=template_config.padding_side,
        cp_partition_mode=distributed_config.cp_partition_mode,
    )
