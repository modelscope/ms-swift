from __future__ import annotations

import logging
import numpy as np
from typing import TYPE_CHECKING, Any, Literal, Optional

if TYPE_CHECKING:
    from swift.dev.config import DatasetConfig, DistributedConfig, TemplateConfig, TrainConfig

logger = logging.getLogger(__name__)


def build_dataset(dataset_config: DatasetConfig,
                  template: Any,
                  train_config: TrainConfig,
                  distributed_config: DistributedConfig,
                  *,
                  encode: bool = True,
                  template_config: Optional[TemplateConfig] = None) -> Any:
    """Load train (+val) once and return ``(train_loader, val_loader)`` (either may be None).

    A single call mirrors legacy ``BaseArguments.load_dataset`` (base_args.py): one ``load_dataset``
    yields both the train split and the split-off val split

    encode: whether to pre-tokenize now (SFT: True) or keep the raw messages and defer tokenization
      to the training/rollout phase (RL such as GRPO/GKD: False). Mirrors legacy
      ``SwiftSft._get_dataset``'s ``pre_process = not (rlhf_type in {grpo, gkd})``.

    ``DatasetConfig.cached_dataset`` / ``cached_val_dataset`` point at directories written by
    ``swift export --to_cached_dataset`` (see swift/pipelines/export/cached_dataset.py). Those rows
    went through the SAME preprocessing chain and already carry a ``lengths`` column, so they are
    concatenated in AFTER encoding and never re-encoded -- mirroring legacy
    ``SwiftSft._prepare_dataset``, which appends get_cached_dataset()'s splits alongside the freshly
    encoded ones.
    """
    from swift.dev.dataset import load_dataset

    load_kwargs = _load_kwargs(dataset_config)

    train_raw, val_raw = (None, None)
    if dataset_config.dataset:
        # load-time shuffle uses dataset_shuffle (whole-dataset shuffle at load, legacy semantics).
        train_raw, val_raw = load_dataset(
            dataset_config.dataset,
            split_dataset_ratio=dataset_config.split_dataset_ratio,
            shuffle=dataset_config.dataset_shuffle,
            **load_kwargs)
    if dataset_config.val_dataset:
        # A separate val_dataset overrides any split-off val, loaded with split_ratio=1.0 and its own
        # val_dataset_shuffle -- legacy base_args.py:364.
        _, val_raw = load_dataset(
            dataset_config.val_dataset,
            split_dataset_ratio=1.0,
            shuffle=dataset_config.val_dataset_shuffle,
            **load_kwargs)

    # Pre-encoded splits from disk. Loaded here (before the per-split builders) so each split can be
    # merged with its freshly-encoded counterpart, and so a run with ONLY cached_dataset still
    # produces a loader.
    cached_train, cached_val = _load_cached_datasets(dataset_config, template_config)

    encode_mode = _encode_mode(dataset_config, template) if encode else None
    train_loader = _build_split_loader(
        train_raw,
        template,
        dataset_config,
        train_config,
        distributed_config,
        encode=encode,
        encode_mode=encode_mode,
        is_val=False,
        cached=cached_train)
    val_loader = _build_split_loader(
        val_raw,
        template,
        dataset_config,
        train_config,
        distributed_config,
        encode=encode,
        encode_mode=encode_mode,
        is_val=True,
        cached=cached_val)
    return train_loader, val_loader


def _load_cached_datasets(dataset_config: DatasetConfig, template_config: Any) -> tuple:
    """Load ``cached_dataset`` / ``cached_val_dataset`` from disk via the dev loader.

    Delegates to ``DatasetLoader.load_cached_datasets`` (dev), which owns the 3.x
    ``length``->``lengths`` rename, the ``truncation_strategy='delete'`` length filter and the
    ``path#sample_count`` subsampling syntax -- so the dev dataset path no longer reaches into
    ``swift.pipelines``. ``max_length`` / ``truncation_strategy`` live on TemplateConfig, which is
    optional here: a caller with no template config still loads caches, and no length filter is then
    applied (``truncation_strategy=None``).

    Returns ``(train_datasets, val_datasets)`` as lists (possibly empty), not concatenated -- the
    per-split builder merges them with the encoded split.
    """
    if not dataset_config.cached_dataset and not dataset_config.cached_val_dataset:
        return [], []
    from swift.dev.dataset import DatasetLoader

    train_datasets, val_datasets = DatasetLoader.load_cached_datasets(
        dataset_config.cached_dataset,
        dataset_config.cached_val_dataset,
        max_length=getattr(template_config, 'max_length', None),
        truncation_strategy=getattr(template_config, 'truncation_strategy', None),
        data_seed=dataset_config.data_seed,
        shuffle=dataset_config.dataset_shuffle,
    )
    logger.info(f'cached_dataset: {len(train_datasets)} train split(s), {len(val_datasets)} val split(s)')
    return train_datasets, val_datasets


def _encode_mode(dataset_config: DatasetConfig, template: Any) -> Literal['stream', 'eager', 'lazy']:
    if dataset_config.streaming:
        return 'stream'
    lazy = dataset_config.lazy_tokenize
    # legacy default: dataset/utils.py:125-130
    if lazy is None:
        lazy = (
            template.model_meta.is_multimodal and not dataset_config.cached_dataset
            and not dataset_config.cached_val_dataset and not dataset_config.packing
            and not dataset_config.group_by_length)
        logger.info(f'Setting lazy_tokenize: {lazy}')
    return 'lazy' if lazy else 'eager'


def _load_kwargs(dataset_config: DatasetConfig) -> dict:
    """Full load_dataset kwargs, mirroring legacy DataArguments.get_dataset_kwargs (data_args.py).

    The previous build_dataset passed only 6 of these; interleave_prob / stopping_strategy /
    shuffle_buffer_size / columns / use_hf / hub_token / download_mode / remove_unused_columns /
    disable_auto_column_mapping / model_name / model_author were silently dropped.
    """
    return dict(
        seed=dataset_config.data_seed,
        num_proc=dataset_config.dataset_num_proc,
        load_from_cache_file=dataset_config.load_from_cache_file,
        streaming=dataset_config.streaming,
        interleave_prob=dataset_config.interleave_prob,
        stopping_strategy=dataset_config.stopping_strategy,
        shuffle_buffer_size=dataset_config.shuffle_buffer_size,
        use_hf=dataset_config.use_hf,
        hub_token=dataset_config.hub_token,
        download_mode=dataset_config.download_mode,
        columns=dataset_config.columns,
        strict=dataset_config.strict,
        model_name=dataset_config.model_name,
        model_author=dataset_config.model_author,
        remove_unused_columns=dataset_config.remove_unused_columns,
        disable_auto_column_mapping=dataset_config.disable_auto_column_mapping,
    )


def _build_split_loader(raw: Any,
                        template: Any,
                        dataset_config: DatasetConfig,
                        train_config: TrainConfig,
                        distributed_config: DistributedConfig,
                        *,
                        encode: bool,
                        encode_mode: Optional[str],
                        is_val: bool,
                        cached: Optional[list] = None) -> Any:
    """Encode -> merge cached -> optional pack -> dataloader for one split (None+no cache -> None)."""
    from swift.dev.legacy_dataloader import build_dataloader, identity_collate

    if raw is None and not cached:
        return None

    # 1. encode. encode=False keeps raw rows; otherwise use the mode resolved by build_dataset.
    if raw is None:
        enc = None
    elif not encode:
        enc = raw
    else:
        enc = _encode(
            raw,
            template,
            mode=encode_mode,
            num_proc=dataset_config.dataset_num_proc,
            strict=dataset_config.strict,
            data_seed=dataset_config.data_seed)

    # 1.5 merge the pre-encoded splits from disk. They are appended AFTER encoding because they were
    #     already encoded when exported -- re-running the preprocessor would either waste the work
    #     (AddLengthPreprocessor) or corrupt rows (EncodePreprocessor on already-tokenized input).
    #     Legacy does the same in SwiftSft._prepare_dataset: cached splits join the freshly encoded
    #     ones in one concat_datasets.
    if cached:
        enc = _concat_with_cached(enc, cached, encode_mode=encode_mode)

    # 2. optional packing. Dependency on encode is asymmetric:
    #    - map-style PackingDataset reads a `lengths` column at construction (packing.py:78), which
    #      only the eager AddLengthPreprocessor writes -- so it REQUIRES encode having run. Legacy
    #      also forbids packing+lazy (base_args.py), so encode=False + non-streaming packing is an
    #      illegal combo -> fail fast rather than KeyError('lengths') deep inside packing.
    #    - IterablePackingDataset (streaming) tokenizes raw rows itself (packing.py:176), so it does
    #      NOT need a prior encode and works on the raw dataset.
    if dataset_config.packing:
        if not encode and not dataset_config.streaming:
            raise ValueError('packing=True with encode=False requires streaming=True: map-style packing needs '
                             'the `lengths` column produced by encoding, which is skipped when encode=False.')
        enc = _pack(
            enc,
            template,
            streaming=dataset_config.streaming,
            packing_length=dataset_config.packing_length,
            packing_strategy=dataset_config.packing_strategy,
            num_proc=dataset_config.packing_num_proc,
            strict=dataset_config.strict)

    # 3. dataloader (list[InputFeature]; processor collates in forward). The dataloader-sampler
    #    shuffle is the per-epoch reshuffle (train_dataloader_shuffle); it is distinct from the
    #    load-time dataset shuffle applied above. Eval never reshuffles.
    batch_size = (train_config.per_device_eval_batch_size if is_val else train_config.per_device_train_batch_size)
    dl_shuffle = (not is_val) and dataset_config.train_dataloader_shuffle
    # Only the TRAIN loader is resumable (deterministic epoch-aware skip). Eval never resumes.
    # Iterable/streaming datasets cannot support deterministic resume -> not resumable.
    # (run_sft rejects streaming+resume up front.)
    resumable = (not is_val) and (not dataset_config.streaming)
    # group_by_length: batch similar-length samples to cut padding. Mirrors legacy swift
    # (trainers/mixin.py:1306-1309): only applied on the map-style TRAIN loader, and it reads a
    # `lengths` column off the (eagerly-encoded) dataset. Legacy also DISABLES it for eval
    # (_disable_group_by_length, mixin.py:1327-1345), so we gate it to the train split. The
    # BatchSamplerShard then feeds `lengths` to transformers get_length_grouped_indices to build
    # length-grouped, still-shuffled batches. We must pass `lengths` explicitly -- BatchSamplerShard
    # raises if group_by_length=True without them. (streaming is rejected up front by
    # validate_configs, so the map-style path is the only one that reaches here with it on.)
    group_by_length = dataset_config.group_by_length and (not is_val)
    lengths = _extract_lengths(enc) if group_by_length else None
    # data_sharding only affects the shuffled TRAIN order (eval uses a sequential sampler), and it
    # is mutually exclusive with group_by_length -- legacy downgrades it with a warning rather than
    # failing, so existing Megatron scripts that set both keep running (batch_sampler.py:86-90).
    data_sharding = dataset_config.data_sharding and (not is_val)
    if data_sharding and group_by_length:
        logger.warning('`group_by_length=True` is incompatible with `data_sharding=True`. '
                       'Setting `data_sharding=False` to enable length grouping.')
        data_sharding = False
    from swift.dev.builders.model import build_device_mesh
    dp_shard_in_loader = _dp_shard_in_loader(distributed_config)
    # Only the loader-side sharding path needs the layout, and build_device_mesh requires
    # nproc_per_node -- which non-Megatron configs are not obliged to set -- so build it on demand.
    device_mesh = build_device_mesh(distributed_config) if dp_shard_in_loader else None
    return build_dataloader(
        enc,
        collate_fn=identity_collate,
        batch_size=batch_size,
        shuffle=dl_shuffle and not dataset_config.streaming,
        drop_last=_drop_last(distributed_config, is_val=is_val),
        data_seed=dataset_config.data_seed,
        group_by_length=group_by_length,
        lengths=lengths,
        data_sharding=data_sharding,
        num_workers=(dataset_config.dataloader_num_workers or 0),
        pin_memory=dataset_config.dataloader_pin_memory,
        resumable=resumable,
        dp_shard_in_loader=dp_shard_in_loader,
        device_mesh=device_mesh,
    )


def _concat_with_cached(enc: Any, cached: list, *, encode_mode: Optional[str]) -> Any:
    """Concatenate the freshly-encoded split with pre-encoded splits loaded from disk.

    Both sides must be map-style HF datasets for ``concatenate_datasets`` to work. That holds for
    the eager path (AddLengthPreprocessor returns a Dataset) but NOT for:
      - lazy mode, where ``enc`` is a ``LazyLLMDataset`` wrapper, and
      - streaming, an IterableDataset,
    neither of which can be concatenated with a Dataset. Mixing `dataset` with `cached_dataset` is
    therefore rejected in those modes instead of failing obscurely inside datasets. Using
    cached_dataset ALONE is fine in any mode -- the cache is already map-style and encoded.

    The two sides can also disagree on Arrow FEATURES even when the columns match: a `messages`
    column round-tripped through save_to_disk keeps the schema it was written with, while a freshly
    loaded one is re-inferred (typed struct vs List(Json)), and concatenate_datasets refuses to align
    those. So the cached splits are cast to the fresh split's features first, and a genuinely
    incompatible cache (different columns / dtypes -- e.g. exported with another template) is
    reported as such instead of surfacing a raw pyarrow error.
    """
    from swift.dev.dataset import DatasetLoader

    if enc is None:
        _, cached = _align_features(None, cached)
        return DatasetLoader.concat_datasets(cached)
    if encode_mode in ('lazy', 'stream'):
        raise ValueError(
            f'cached_dataset cannot be combined with `dataset` when encode_mode={encode_mode!r}: the freshly '
            'loaded split is a lazy/iterable wrapper and cannot be concatenated with the map-style cached '
            'dataset. Either set DatasetConfig.lazy_tokenize=False (streaming=False) to encode eagerly, or '
            'pass only cached_dataset (already encoded on disk).')
    enc, cached = _align_features(enc, cached)
    return DatasetLoader.concat_datasets([enc] + cached)


def _align_features(enc: Any, cached: list) -> tuple:
    """Make ``enc`` and the cached splits share one Arrow schema, returning both sides aligned.

    ``messages`` is the hard case. datasets represents a list-of-dicts column either as a plain Arrow
    struct (``list<struct<role,content>>``) or as the ``arrow.json`` EXTENSION type (the ``Json``
    feature), and which one you get depends on how the split was produced -- a fresh
    ``AddLengthPreprocessor`` map yields the struct, while the same data round-tripped through
    ``save_to_disk`` comes back as arrow.json. ``concatenate_datasets`` refuses to align the two.

    The asymmetry that decides the direction: pyarrow CAN cast struct -> arrow.json (it serializes),
    but NOT arrow.json -> struct (it sees an opaque string). So when the two disagree, both sides are
    cast to the CACHED (Json) schema rather than to the fresh one. Columns that genuinely differ
    (a cache exported under another template) are reported instead of raising a raw pyarrow error.
    """
    cached = list(cached)
    if not cached:
        return enc, cached
    enc_features = getattr(enc, 'features', None)
    cache_features = getattr(cached[0], 'features', None)
    if enc_features is None or cache_features is None or enc_features == cache_features:
        return enc, cached
    if set(enc_features) != set(cache_features):
        raise ValueError(f'cached_dataset columns {sorted(cache_features)} do not match the training split '
                         f'{sorted(enc_features)}. The cache was produced with a different template/config; '
                         're-export it with the same TemplateConfig, or drop the mismatched path.')
    # Cast the fresh split onto the cache's schema (struct -> Json is castable; the reverse is not).
    try:
        enc = enc.cast(cache_features)
    except Exception as e:
        raise ValueError(f'cached_dataset could not be aligned with the training split: {e}. Re-export the '
                         'cache with the current TemplateConfig, or train on the cache alone.') from e
    aligned = []
    for ds in cached:
        features = getattr(ds, 'features', None)
        if features is not None and features != cache_features:
            ds = ds.cast(cache_features)
        aligned.append(ds)
    return enc, aligned


def _extract_lengths(enc: Any) -> list:
    column_names = getattr(enc, 'column_names', None)
    if column_names is not None and 'lengths' in column_names:
        return list(enc['lengths'])
    raise ValueError('group_by_length=True requires a `lengths` column, which is only produced by EAGER '
                     'encoding. Set DatasetConfig.lazy_tokenize=False (and use a non-streaming, map-style '
                     'dataset) so the length of every sample is precomputed. Legacy swift has the same '
                     'requirement (it reads train_dataset[\'lengths\']).')


def _dp_shard_in_loader(distributed_config: DistributedConfig) -> bool:
    """Should the dataloader shard by data-parallel rank itself? (config-derived, no env/probe)

    Three cases, all read from DistributedConfig (mode + backend) -- never from env vars or runtime
    state. This matters because on the DRIVER (where build_dataset runs) the TWINKLE_MODE env var is
    unset even in ray mode (twinkle only sets it on workers), so an env probe would misreport 'local'
    and double-shard against slice_dp. DistributedConfig.mode is the user/CLI-declared intent:

      - mode=='ray'            -> False. The dataloader is a bare driver loader; DP scatter happens
                                  later in model.forward_backward(dispatch='slice_dp'). (This is why
                                  the twinkle cookbook uses a plain dataloader in ray mode.)
      - mode=='local' + megatron -> True. No driver to scatter (slice_dp no-ops locally), so the
                                  dataloader owns DP sharding and takes the DP rank from the
                                  DeviceMesh (global-rank sharding is wrong once TP/PP/CP>1).
      - mode=='local' + hf     -> False. TP/PP/CP==1, so the global-rank BatchSamplerShard already
                                  equals the DP shard.
    """
    if distributed_config.mode == 'ray':
        return False
    from swift.dev.builders.model import is_megatron_backend
    return is_megatron_backend(distributed_config)


def _drop_last(distributed_config: DistributedConfig, *, is_val: bool) -> bool:
    """Drop a trailing partial batch? megatron yes, hf no -- matching legacy on both.

    Megatron needs it: global_batch_size is an exact invariant, and a remainder batch shifts every
    later step once an epoch boundary falls mid-run. legacy drops on both splits too, by two
    different mechanisms -- the train sampler has no drop_last knob and floors unconditionally
    (batch_sampler.py), the val one defaults to drop_last=True.

    hf keeps the remainder: legacy defers to TrainingArguments.dataloader_drop_last (default False),
    so dropping here would discard real samples and diverge from legacy on that path.
    """
    from swift.dev.builders.model import is_megatron_backend
    if not is_megatron_backend(distributed_config):
        return False
    # is_val stays in the signature: both splits drop today, but for different reasons, so a future
    # change may split them again.
    del is_val
    return True


def _encode(raw: Any, template: Any, *, mode: Literal['lazy', 'eager', 'stream'], num_proc: int, strict: bool,
            data_seed: int) -> Any:
    """Encode a raw messages split into trainable samples.

    Routes:
      - lazy   -> LazyLLMDataset (encode on the fly at __getitem__).
      - stream -> EncodePreprocessor.map (streaming pre-tokenize).
      - eager  -> pre-tokenize via map. The 'split' truncation strategy expands one sample into
                  many, so it must fully encode with EncodePreprocessor; otherwise use
                  AddLengthPreprocessor, which keeps the raw row and only adds a `lengths` column
                  (rows are encoded later by the lazy/collate path).
    """
    from swift.dev.dataset import EncodePreprocessor, LazyLLMDataset, MeasurePreprocessor

    truncation_strategy = getattr(template, 'truncation_strategy', None)
    if truncation_strategy == 'split' and mode != 'eager':
        raise ValueError(f"truncation_strategy='split' requires mode='eager', got mode='{mode}'. "
                         "The 'split' strategy may produce multiple samples per input.")

    if mode == 'lazy':
        random_state = np.random.RandomState(data_seed)
        return LazyLLMDataset(raw, template.encode, strict=strict, random_state=random_state)

    # eager / stream: 'split' needs a full encode (emits multiple samples per input); otherwise
    # AddLengthPreprocessor only writes `lengths` and leaves rows raw.
    if truncation_strategy == 'split':
        preprocessor = EncodePreprocessor(template)
    else:
        preprocessor = MeasurePreprocessor(template)
    return preprocessor(raw, num_proc=num_proc, strict=strict)


def _pack(dataset: Any, template: Any, *, streaming: bool, packing_length: Optional[int], packing_strategy: str,
          num_proc: int, strict: bool) -> Any:
    """Pack an encoded dataset for efficient training (map-style or iterable)."""
    from swift.dev.dataset import IterablePackingDataset, PackingDataset

    length = packing_length or template.max_length
    cls = IterablePackingDataset if streaming else PackingDataset
    return cls(
        template, dataset, num_proc=num_proc, packing_length=length, packing_strategy=packing_strategy, strict=strict)
