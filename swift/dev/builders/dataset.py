# Copyright (c) ModelScope Contributors. All rights reserved.
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, Tuple

from swift.dev.utils.logger import get_logger

if TYPE_CHECKING:
    from twinkle import DeviceMesh

    from swift.dev.config import DatasetConfig, DistributedConfig, TemplateConfig, TrainConfig

logger = get_logger()


def build_dataset(dataset_config: DatasetConfig,
                  template: Any,
                  train_config: TrainConfig,
                  distributed_config: DistributedConfig,
                  *,
                  encode: bool = True,
                  template_config: Optional[TemplateConfig] = None,
                  device_mesh: Optional['DeviceMesh'] = None) -> Any:
    """Load train (+val) once and return ``(train_loader, val_loader)`` (either may be None).

    ``encode`` pre-tokenizes now (SFT) or defers to train/rollout time (GRPO/GKD: False);
    ``cached_dataset`` splits join after encoding and are never re-encoded (legacy SwiftSft).
    """
    from swift.dev.dataset import load_dataset, register_dataset_info
    # legacy base_args.py: a nonzero split ratio alongside val_dataset would carve a slice out of
    # --dataset that is neither trained on (train keeps only its remainder) nor evaluated.
    if dataset_config.val_dataset and dataset_config.split_dataset_ratio:
        raise ValueError(
            f'split_dataset_ratio={dataset_config.split_dataset_ratio} cannot be combined with '
            'val_dataset: the ratio carves a val slice out of --dataset that is then neither trained '
            'on (train keeps only its remainder) nor evaluated (val_dataset overrides it). Drop '
            'split_dataset_ratio, or drop val_dataset to eval on the split-off slice instead.')
    for info_path in dataset_config.custom_dataset_info:
        register_dataset_info(info_path, exist_ok=True)

    load_kwargs = _load_kwargs(dataset_config)

    train_raw, val_raw = (None, None)
    if dataset_config.dataset:
        # load-time shuffle is dataset_shuffle (whole-dataset shuffle at load, legacy semantics)
        train_raw, val_raw = load_dataset(
            dataset_config.dataset,
            split_dataset_ratio=dataset_config.split_dataset_ratio,
            shuffle=dataset_config.dataset_shuffle,
            **load_kwargs)
    if dataset_config.val_dataset:
        # legacy base_args.py: val_dataset replaces any split-off val; interleave_prob is a
        # train-mixing knob and must not sample the val set.
        load_kwargs.pop('interleave_prob', None)
        _, val_raw = load_dataset(
            dataset_config.val_dataset,
            split_dataset_ratio=1.0,
            shuffle=dataset_config.val_dataset_shuffle,
            **load_kwargs)

    cached_train, cached_val = _load_cached_datasets(dataset_config, template_config)
    encode_mode = _encode_mode(dataset_config, template) if encode else None
    train_loader = _build_split_loader(
        train_raw, template, dataset_config, train_config, distributed_config,
        encode=encode, encode_mode=encode_mode, is_val=False, cached=cached_train,
        device_mesh=device_mesh)
    val_loader = _build_split_loader(
        val_raw, template, dataset_config, train_config, distributed_config,
        encode=encode, encode_mode=encode_mode, is_val=True, cached=cached_val,
        device_mesh=device_mesh)
    return train_loader, val_loader


def _load_kwargs(dataset_config: DatasetConfig) -> dict:
    """Full load_dataset kwargs, mirroring legacy DataArguments.get_dataset_kwargs.

    remove_unused_columns / disable_auto_column_mapping are refused rather than forwarded: the dev
    dataset framework always drops consumed source columns and always applies declared renames.
    """
    if not dataset_config.remove_unused_columns:
        raise ValueError('remove_unused_columns=False is not supported by the dev dataset framework: its '
                         'preprocessors always drop the source columns they consumed. Drop the flag, or keep '
                         'the extra columns by loading the dataset without a preprocessor.')
    if dataset_config.disable_auto_column_mapping:
        raise ValueError('disable_auto_column_mapping=True is not supported by the dev dataset framework: a '
                         "dataset's `columns` renames are part of its loader declaration and always apply. "
                         'Pass `columns` to override individual names instead.')
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
    )


def _load_cached_datasets(dataset_config: DatasetConfig, template_config: Any) -> tuple:
    """Load ``cached_dataset`` / ``cached_val_dataset`` from disk as ``(train_splits, val_splits)`` lists.

    DatasetLoader.load_cached_datasets owns the length->lengths rename, the delete-strategy length filter
    and the ``path#N`` subsampling; template_config is optional (None = no length filter applied).
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
    # legacy default (dataset/utils.py): multimodal without cache/packing/length-grouping
    if lazy is None:
        lazy = (
            template.model_meta.is_multimodal and not dataset_config.cached_dataset
            and not dataset_config.cached_val_dataset and not dataset_config.packing
            and not dataset_config.group_by_length)
        logger.info(f'Setting lazy_tokenize: {lazy}')
    return 'lazy' if lazy else 'eager'


def _build_split_loader(raw: Any,
                        template: Any,
                        dataset_config: DatasetConfig,
                        train_config: TrainConfig,
                        distributed_config: DistributedConfig,
                        *,
                        encode: bool,
                        encode_mode: Optional[str],
                        is_val: bool,
                        cached: Optional[list] = None,
                        device_mesh: Optional['DeviceMesh'] = None) -> Any:
    """Encode -> merge cached -> optional pack -> dataloader for one split (None+no cache -> None)."""
    if raw is None and not cached:
        return None

    if raw is None:
        enc = None
    elif not encode:
        enc = raw
    else:
        enc = _encode(
            raw, template, mode=encode_mode, num_proc=dataset_config.dataset_num_proc,
            strict=dataset_config.strict, data_seed=dataset_config.data_seed)

    # cached splits are pre-encoded: merging AFTER encoding avoids re-running the preprocessor
    # (AddLengthPreprocessor would waste it; EncodePreprocessor would corrupt already-tokenized rows)
    if cached:
        enc = _concat_with_cached(enc, cached, encode_mode=encode_mode)

    # map-style packing reads a `lengths` column only eager encoding writes; the iterable packer
    # tokenizes raw rows itself, so packing+encode=False requires streaming (legacy forbids packing+lazy)
    if dataset_config.packing:
        if not encode and not dataset_config.streaming:
            raise ValueError('packing=True with encode=False requires streaming=True: map-style packing needs '
                             'the `lengths` column produced by encoding, which is skipped when encode=False.')
        enc = _pack(
            enc, template, streaming=dataset_config.streaming,
            packing_length=dataset_config.packing_length,
            packing_strategy=dataset_config.packing_strategy,
            num_proc=dataset_config.packing_num_proc,
            strict=dataset_config.strict,
            data_seed=dataset_config.data_seed)

    batch_size = (train_config.per_device_eval_batch_size if is_val else train_config.per_device_train_batch_size)
    dl_shuffle = (not is_val) and dataset_config.train_dataloader_shuffle
    # group_by_length: train split only (legacy disables it for eval); needs the eager `lengths` column
    group_by_length = dataset_config.group_by_length and (not is_val)
    lengths = _extract_lengths(enc) if group_by_length else None
    # data_sharding reshuffles the TRAIN order only; legacy downgrades it (with a warning) rather than
    # failing when group_by_length is also on, so those Megatron scripts keep running
    data_sharding = dataset_config.data_sharding and (not is_val)
    if data_sharding and group_by_length:
        logger.warning('`group_by_length=True` is incompatible with `data_sharding=True`. '
                       'Setting `data_sharding=False` to enable length grouping.')
        data_sharding = False

    global_batch_size, device_mesh = _twinkle_loader_layout(distributed_config, batch_size, device_mesh)

    from twinkle.dataloader import DataLoader
    # the loader is inherently resumable (skip_consumed_samples / get_state); the training loop decides
    # whether to skip -- eval never resumes
    num_workers = dataset_config.dataloader_num_workers or 0
    if dataset_config.streaming and dataset_config.packing and num_workers > 0:
        logger.warning('Streaming packing uses one dataset reader; setting `dataloader_num_workers=0`. '
                       '`packing_num_proc` still controls parallel encoding.')
        num_workers = 0
    worker_kwargs = {}
    if num_workers > 0:
        worker_kwargs['persistent_workers'] = dataset_config.dataloader_persistent_workers
        if dataset_config.dataloader_prefetch_factor is not None:
            worker_kwargs['prefetch_factor'] = dataset_config.dataloader_prefetch_factor
    return DataLoader(
        enc,
        batch_size=global_batch_size,
        device_mesh=device_mesh,
        data_seed=dataset_config.data_seed,
        group_by_length=group_by_length,
        lengths=lengths,
        data_sharding=data_sharding,
        # MapRetryDataset substitutes an unencodable row, seeded so every rank and every resume pick the
        # same replacement; strict (max_retries=0) stops the run on the first failure instead
        retry_seed=dataset_config.data_seed,
        max_retries=0 if dataset_config.strict else 20,
        collate_fn=_identity_collate,
        shuffle=dl_shuffle and not dataset_config.streaming,
        drop_last=_drop_last(distributed_config),
        num_workers=num_workers,
        pin_memory=dataset_config.dataloader_pin_memory,
        **worker_kwargs,
    )


def _identity_collate(batch):
    """Return the batch as a ``list[InputFeature]`` unchanged; dev collates inside the model forward."""
    return list(batch)


def _twinkle_loader_layout(distributed_config: DistributedConfig,
                           per_device_batch_size: int,
                           device_mesh: Optional['DeviceMesh'] = None) -> tuple:
    """``(global_batch_size, loader_device_mesh)``: twinkle's DataLoader takes the GLOBAL batch and its
    ShardedBatchSampler slices it per DP rank, so the batch is scaled by the DP width.
    """
    if device_mesh is not None:
        # HF Ulysses SP: the sampler slices by data_world_size (world/ulysses), so SP peers get
        # identical samples and only DP groups differ
        return per_device_batch_size * device_mesh.data_world_size, device_mesh
    if distributed_config.nproc_per_node is None:
        # plain transformers torchrun (world size lives in the env, not the config): scale by the same
        # source twinkle's local initialize reads; mesh stays None to inherit twinkle's global DP mesh
        from twinkle.utils import Platform
        return per_device_batch_size * Platform.get_world_size(), None
    # Megatron path: build_parallel_mesh sizes the DP layout off nproc_per_node; under ray the loader is
    # a bare driver loader and the DP scatter happens later in model.forward_backward('slice_dp')
    from swift.dev.builders.model import build_parallel_mesh
    mesh = build_parallel_mesh(distributed_config)
    return per_device_batch_size * mesh.data_world_size, (None if distributed_config.mode == 'ray' else mesh)


def _concat_with_cached(enc: Any, cached: list, *, encode_mode: Optional[str]) -> Any:
    """Concatenate the freshly-encoded split with pre-encoded splits loaded from disk.

    lazy/stream wrappers cannot be concatenated with the map-style cache, so mixing `dataset` with
    `cached_dataset` is refused in those modes; a cache ALONE is fine in any mode.
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

    list-of-dict columns come back struct or arrow.json depending on how the split was produced and
    cannot be concatenated mixed; pyarrow casts struct -> Json but not back, so both sides go CACHED.
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


def _drop_last(distributed_config: DistributedConfig) -> bool:
    """Megatron drops a trailing partial batch (exact global_batch_size invariant); hf keeps it (legacy)."""
    return distributed_config.backend == 'megatron'


def _encode(raw: Any, template: Any, *, mode: Literal['lazy', 'eager', 'stream'], num_proc: int, strict: bool,
            materialize: bool = False, data_seed: Optional[int] = None) -> Any:
    """Encode a raw messages split into trainable samples.

    lazy -> LazyLLMDataset (encode at __getitem__); 'split' truncation / an explicit ``materialize``
    need the full EncodePreprocessor; otherwise MeasurePreprocessor only adds `lengths`, rows stay raw.
    """
    from swift.dev.dataset import EncodePreprocessor, LazyLLMDataset, MeasurePreprocessor

    truncation_strategy = getattr(template, 'truncation_strategy', None)
    if truncation_strategy == 'split' and mode != 'eager':
        raise ValueError(f"truncation_strategy='split' requires mode='eager', got mode='{mode}'. "
                         "The 'split' strategy may produce multiple samples per input.")

    if mode == 'lazy':
        return LazyLLMDataset(raw, template.encode)

    # 'split' emits multiple samples per input; materialize (only the cached-dataset exporter sets it)
    # forces input_ids/labels into the rows. data_seed pins build-time random preprocessing per row.
    preprocessor_kwargs = {} if data_seed is None else {'seed': data_seed}
    if truncation_strategy == 'split' or materialize:
        preprocessor = EncodePreprocessor(template, **preprocessor_kwargs)
    else:
        preprocessor = MeasurePreprocessor(template, **preprocessor_kwargs)
    return preprocessor(raw, num_proc=num_proc, strict=strict)


def _pack(dataset: Any, template: Any, *, streaming: bool, packing_length: Optional[int], packing_strategy: str,
          num_proc: int, strict: bool, data_seed: int = 0) -> Any:
    """Pack an encoded dataset for efficient training (map-style or iterable)."""
    from swift.dev.dataset import IterablePackingDataset, PackingDataset

    length = packing_length or template.max_length
    cls = IterablePackingDataset if streaming else PackingDataset
    # data_seed is the streaming protocol's (set_data_seed on a state restore); the map-style
    # PackingDataset forwards **kwargs to SwiftDataset, whose signature is closed, so it must
    # not receive it -- map-style resume rides SampleIndex.epoch instead.
    seed_kwargs = {'data_seed': data_seed} if streaming else {}
    return cls(
        template,
        dataset,
        num_proc=num_proc,
        packing_length=length,
        packing_strategy=packing_strategy,
        strict=strict,
        **seed_kwargs)


def load_prompt_only_rows(dataset_config: DatasetConfig, max_rows: Optional[int],
                          split_dataset_ratio: float) -> List[Dict[str, Any]]:
    """Load the raw prompt rows the rollout/sampling recipes (infer/gkd/grpo) drive generation from.

    ``val_dataset`` wins outright; otherwise only the eval slice of ``dataset`` is used (ratio=0 runs
    all) -- legacy infer's ``_prepare_val_dataset`` semantics.
    """
    from swift.dev.dataset import load_dataset

    kwargs = _load_kwargs(dataset_config)
    # legacy infer: val_dataset loads whole (its own shuffle flag; interleave_prob -- a train-mixing
    # knob -- popped); the eval slice of --dataset uses dataset_shuffle, before the max_rows cut
    if dataset_config.val_dataset:
        kwargs.pop('interleave_prob', None)
        _, rows = load_dataset(
            datasets=list(dataset_config.val_dataset), split_dataset_ratio=1.0,
            shuffle=dataset_config.val_dataset_shuffle, **kwargs)
    elif split_dataset_ratio:
        _, rows = load_dataset(
            datasets=list(dataset_config.dataset), split_dataset_ratio=split_dataset_ratio,
            shuffle=dataset_config.dataset_shuffle, **kwargs)
        logger.info(f'load_prompt_only_rows: using the eval split of --dataset (split_dataset_ratio='
                    f'{split_dataset_ratio}); pass split_dataset_ratio=0 to run over all of it.')
    else:
        rows, _ = load_dataset(
            datasets=list(dataset_config.dataset), split_dataset_ratio=0.0,
            shuffle=dataset_config.dataset_shuffle, **kwargs)

    rows = list(rows or [])
    if max_rows is not None:
        rows = rows[:max_rows]
    return rows


def build_prompt_dataloader(prompts: List[List[dict]],
                            prompt_extras: Optional[List[Dict[str, Any]]],
                            *,
                            seed: int,
                            batch_size: int = 256,
                            shuffle: bool = True) -> Any:
    """A twinkle ``DataLoader`` over the online-RL prompt rows (the prompt-side build_dataset).

    No encoding -- the rollout encodes each prompt itself. ``shuffle=False`` preserves input order;
    otherwise ordering follows the loader seed and epoch. One pass per ``iter()``; drop_last stays
    off so a partial batch is never a dropped prompt.
    """
    from twinkle.dataset import Dataset, DatasetMeta
    from twinkle.dataloader import DataLoader

    if not prompts:
        raise ValueError('build_prompt_dataloader needs at least one prompt, got an empty sequence.')
    if batch_size <= 0:
        raise ValueError(f'batch_size must be > 0, got {batch_size}.')
    if prompt_extras is not None and len(prompt_extras) != len(prompts):
        raise ValueError(
            f'prompt_extras must align with prompts ({len(prompts)} rows), got {len(prompt_extras)}.')
    extras = prompt_extras if prompt_extras is not None else [{} for _ in prompts]
    rows = [{'messages': messages, **extra} for messages, extra in zip(prompts, extras)]
    dataset = Dataset(DatasetMeta(data=rows))
    return DataLoader(
        dataset,
        batch_size=batch_size,
        data_seed=seed,
        num_workers=0,
        collate_fn=_identity_collate,
        shuffle=shuffle)


def to_trajectory(row: Dict[str, Any], messages: List[Dict[str, Any]],
                  template_config: 'TemplateConfig') -> Dict[str, Any]:
    """Build the twinkle Trajectory for one row.

    ``TemplateConfig.system`` REPLACES a system turn the row already has rather than stacking a second
    one, matching legacy: two system messages is not a supported prompt shape for most templates.
    """
    trajectory: Dict[str, Any] = {'messages': messages}
    system = getattr(template_config, 'system', None)
    if system:
        without_system = [message for message in messages if message.get('role') != 'system']
        trajectory['messages'] = [{'role': 'system', 'content': system}] + without_system
    for key in ('images', 'audios', 'videos', 'objects', 'tools'):
        if row.get(key):
            trajectory[key] = row[key]
    return trajectory


def split_prompt_and_reference(rows: List[Dict[str, Any]],
                               template_config: 'TemplateConfig') -> Tuple[List[Dict[str, Any]], List[Optional[str]]]:
    """rows -> ``(trajectories, references)``, with each row's trailing assistant turn moved aside.

    Generation must see a prompt that stops before the reference answer, and the answer itself is
    needed afterwards (metric label / reward ground truth); one place keeps "what the model saw"
    identical across every consumer.
    """
    trajectories, references = [], []
    for row in rows:
        messages = list(row['messages'])
        reference = messages.pop()['content'] if messages and messages[-1]['role'] == 'assistant' else None
        references.append(reference)
        trajectories.append(to_trajectory(row, messages, template_config))
    return trajectories, references
