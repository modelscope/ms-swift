# Copyright (c) ModelScope Contributors. All rights reserved.
# Dataset -> dataset-store components: encode a dataset once and reuse the store.
from .exporter import CacheDatasetComponent

__all__ = ['CacheDatasetComponent']
