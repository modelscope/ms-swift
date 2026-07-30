"""Dataset loading, splitting, streaming, packing, and dataloader configuration."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Literal, Optional, Union


@dataclass
class DatasetConfig:
    """Dataset sources, caching, column mapping, packing, and dataloader settings."""

    # === Dataset Sources ===
    dataset: List[str] = field(default_factory=list)
    val_dataset: List[str] = field(default_factory=list)
    cached_dataset: List[str] = field(default_factory=list)
    cached_val_dataset: List[str] = field(default_factory=list)

    # === Splitting & Shuffling ===
    split_dataset_ratio: float = 0.0
    data_seed: int = 42
    dataset_num_proc: int = 1
    load_from_cache_file: bool = False
    dataset_shuffle: bool = True
    val_dataset_shuffle: bool = False

    # === Streaming ===
    streaming: bool = False
    interleave_prob: Optional[List[float]] = None
    stopping_strategy: Literal['first_exhausted', 'all_exhausted'] = 'first_exhausted'
    shuffle_buffer_size: int = 1000
    download_mode: Literal['force_redownload', 'reuse_dataset_if_exists'] = 'reuse_dataset_if_exists'

    # === Column Mapping ===
    columns: Optional[Union[dict, str]] = None
    strict: bool = False
    remove_unused_columns: bool = True
    disable_auto_column_mapping: bool = False
    model_name: Optional[List[str]] = None
    model_author: Optional[List[str]] = None
    custom_dataset_info: List[str] = field(default_factory=list)

    # === Hub ===
    use_hf: bool = False
    hub_token: Optional[str] = None

    # === Packing ===
    packing: bool = False
    packing_length: Optional[int] = None
    packing_num_proc: int = 1
    packing_strategy: Literal['binpack', 'sequential'] = 'binpack'
    lazy_tokenize: Optional[bool] = None  # see default value in dev/builders/dataset _encode_mode

    # === Sampling & DataLoader ===
    train_dataloader_shuffle: bool = True
    group_by_length: bool = False
    data_sharding: bool = False  # megatron only
    dataloader_num_workers: Optional[int] = None
    dataloader_pin_memory: bool = True
    dataloader_persistent_workers: bool = False
    dataloader_prefetch_factor: Optional[int] = None
