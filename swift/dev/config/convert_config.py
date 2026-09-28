"""HF <-> Megatron(mcore) weight-conversion configuration."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Literal, Optional


@dataclass
class ConvertConfig:
    """Conversion direction, mcore checkpoint sources, and precision-test options.

    Separate from DistributedConfig on purpose: those fields describe how a model is PARALLELIZED for
    training, whereas these describe a one-shot offline format migration. The parallelism sizes
    (tp/pp/ep/...) still come from DistributedConfig, because the mcore checkpoint layout depends on
    them.
    """

    # === direction ===
    # Exactly one is expected. hf->mcore is the `to_mcore` case with no mcore_model set; mcore->hf is
    # `to_hf`; mcore->mcore (to_mcore WITH mcore_model) is the resharding / LoRA-merge case.
    to_mcore: bool = False
    to_hf: bool = False

    # === sources ===
    mcore_model: Optional[str] = None
    # An mcore-format LoRA checkpoint; when set it is merged into the base weights before saving.
    mcore_adapter: Optional[str] = None

    # === torch-dist sharding ===
    # None => derived from the checkpoint size (one shard-writer thread per ~10GB, min 2), matching
    # legacy. Only affects write throughput, not the produced weights.
    thread_count: Optional[int] = None

    # === verification ===
    # Runs both models on the same input and compares outputs. Costs a second model in memory, so it
    # is opt-in.
    test_convert_precision: bool = False
    test_convert_dtype: Literal['float16', 'bfloat16', 'float32'] = 'float32'

    # === LoRA ===
    #: Fold a LoRA adapter into the base weights and save one plain model. The HF-side counterpart of
    #: ``mcore_adapter`` above; both produce a checkpoint with no adapter left to load.
    merge_lora: bool = False
    #: Save the adapter in PEFT's own layout instead of swift's, for loading by code that only knows
    #: PEFT. A format change only -- it merges nothing.
    to_peft_format: bool = False

    # === Other export targets ===
    # These leave the HF <-> mcore axis entirely: each writes a different artefact from the same model,
    # and at most one applies to a run.
    #: Tokenise the dataset and save the result, so later runs skip encoding. The one target that
    #: produces a dataset rather than a model, hence the template mode below.
    to_cached_dataset: bool = False
    #: Which encoding the cached dataset is built for. It has to be stated because the same rows encode
    #: differently per objective, and a cache built for one is wrong for the others.
    template_mode: Literal['train', 'rlhf', 'kto'] = 'train'
    #: Materialise the encoded tokens (input_ids / labels) into the store instead of keeping the raw row
    #: plus a ``lengths`` column. Off by default, which mirrors the eager training path (tokenisation then
    #: still happens per batch at train time); on, the store is directly trainable -- the model forward's
    #: ``_not_encoded`` guard skips re-encoding any row that already carries ``input_ids``. Text only for
    #: now: a multimodal row's ``input_ids`` depend on the image grid, so the recipe refuses that combo.
    store_encoded: bool = False
    #: Container the store is written in, shared with ``swift infer`` (see swift.dev.dataset.store).
    #: ``arrow`` (default) is a single ``save_to_disk`` table, the directly-trainable form; ``jsonl`` is a
    #: portable line-delimited dump. ``bin`` (the flat mmap pretraining layout) is a named later phase and
    #: raises until it exists rather than silently falling back.
    store_format: Literal['arrow', 'jsonl', 'bin'] = 'arrow'
    #: Restrict the persisted columns to this allow-list; None keeps every column the encoding produced.
    #: A name that matches no column is rejected rather than silently dropped.
    store_fields: Optional[List[str]] = None

    # === Hub upload ===
    #: Message on the commit created by the upload. The hub credentials and repo id are not here: they
    #: are CheckpointConfig's, so training and export push the same way.
    commit_message: str = 'update files'
    #: Write into an output directory that already has files in it. Off by default, since the usual
    #: cause of a non-empty directory is a previous export nobody meant to overwrite.
    exist_ok: bool = False
