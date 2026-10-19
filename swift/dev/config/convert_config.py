# Copyright (c) ModelScope Contributors. All rights reserved.
from dataclasses import dataclass
from typing import List, Literal, Optional


@dataclass
class ConvertConfig:
    """HF <-> mcore conversion; parallel sizes still come from DistributedConfig (the layout needs them)."""

    # === direction ===
    # to_mcore (no mcore_model) = hf->mcore; to_hf = mcore->hf; to_mcore WITH mcore_model = reshard/merge.
    to_mcore: bool = False
    to_hf: bool = False

    # === sources ===
    mcore_model: Optional[str] = None
    # mcore-format LoRA checkpoint; merged into the base weights before saving.
    mcore_adapter: Optional[str] = None

    # === torch-dist sharding ===
    # None => derived from checkpoint size (~one writer thread per 10GB, min 2); throughput only.
    thread_count: Optional[int] = None

    # === verification ===
    # Runs both models on the same input and compares outputs; opt-in (costs a second model).
    test_convert_precision: bool = False
    test_convert_dtype: Literal['float16', 'bfloat16', 'float32'] = 'float32'

    # === LoRA ===
    # Fold a LoRA adapter into the base weights and save one plain model.
    merge_lora: bool = False
    # Save the adapter in PEFT's own layout for PEFT-only consumers; merges nothing.
    to_peft_format: bool = False

    # === Other export targets ===
    # Save the tokenised dataset so later runs skip encoding (the one dataset-producing target).
    to_cached_dataset: bool = False
    # The same rows encode differently per objective; a cache built for one is wrong for the others.
    template_mode: Literal['train', 'rlhf', 'kto'] = 'train'
    # Materialise input_ids/labels (multimodal: the full template.encode output) into the store;
    # the forward's _not_encoded guard then skips re-encoding, so the store is directly trainable.
    store_encoded: bool = False
    # OPEN registry key: any writable format in dataset.storage.STORAGE_MAPPING (arrow/jsonl plus
    # @register_storage plugins); unregistered/read-only names are refused at config validation.
    store_format: str = 'arrow'
    # Allow-list of persisted columns; None keeps every column the encoding produced.
    store_fields: Optional[List[str]] = None

    # === Hub upload ===
    # Commit message only; hub credentials and repo id are CheckpointConfig's.
    commit_message: str = 'update files'
    # Off by default: a non-empty output directory is usually a previous export to not overwrite.
    exist_ok: bool = False
