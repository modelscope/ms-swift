# Copyright (c) ModelScope Contributors. All rights reserved.
"""Multimodal resource downloading: pull and cache the media archives datasets reference by id."""
from .base import CompressedDownloader, FileDownloader, MediaDownloader, ShardedDownloader

__all__ = ['MediaDownloader', 'CompressedDownloader', 'FileDownloader', 'ShardedDownloader']
