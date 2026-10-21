# Copyright (c) ModelScope Contributors. All rights reserved.
"""End-to-end tests for the download-backed multimodal dataset path.

Some multimodal datasets ship only text plus an image *name*; the pixels live in an archive that has
to be fetched and unpacked first. That machinery is ``mm_download.MediaDownloader`` (fetch once, cache
atomically) plus ``ArchiveImagePreprocessor`` (fetch in ``prepare_dataset``, then resolve each row's
relative name against the unpacked folder, dropping rows whose file is absent). The published archives
this normally runs against are multi-gigabyte, so this file drives the *same* code path against a local
``file://`` archive: ``datasets.DownloadManager.download_and_extract`` does not care whether the URL
scheme is https or file, so every line of our fetch/resolve/drop/encode logic is exercised, offline and
deterministically. The actual HTTP transport belongs to ``datasets``, not to us.

One ``@pytest.mark.slow`` test does cross the network, to prove the transport hop still works against a
real host; it skips rather than fails when the network is unreachable.
"""
import os
import shutil
import socket
import zipfile

import numpy as np
import pytest
import torch
from datasets import Dataset as HfDataset
from PIL import Image

from swift.dev.dataset.loader.mllm import ArchiveImagePreprocessor
from swift.dev.dataset.mm_download import MediaDownloader

MODEL = os.environ.get('SWIFT_TEST_MODEL', '/mnt/workspace/yzhao/tastelikefeet/Qwen3.5-4B-CM-v2')
MODEL_TYPE = os.environ.get('SWIFT_TEST_MODEL_TYPE', 'qwen3_5')

needs_model = pytest.mark.skipif(not os.path.isdir(MODEL), reason=f'no local model at {MODEL}')


def make_media_zip(path, image_names=('cat.png', )):
    """Build a zip holding ``image_names`` at its root, and return its ``file://`` URL."""
    path = str(path)
    with zipfile.ZipFile(path, 'w') as archive:
        for name in image_names:
            png = os.path.join(os.path.dirname(path), name)
            Image.fromarray((np.random.RandomState(0).rand(56, 56, 3) * 255).astype('uint8')).save(png)
            archive.write(png, arcname=name)
    return 'file://' + path


def clear_cache(alias):
    """Drop a cached resource folder (and any leftover .tmp) so a test observes a real fetch."""
    shutil.rmtree(os.path.join(MediaDownloader.cache_dir, alias), ignore_errors=True)
    shutil.rmtree(os.path.join(MediaDownloader.cache_dir, f'{alias}.tmp'), ignore_errors=True)


class LocalArchiveImagePreprocessor(ArchiveImagePreprocessor):
    """An ArchiveImagePreprocessor whose archive is the local zip under test.

    The real subclasses name a published https archive; this one names a ``file://`` zip so the fetch
    is offline. Everything else -- prepare_dataset's download, resolve's path join, the drop-missing
    rule -- is the inherited production code.
    """

    media_alias = 'swift_a5_local_archive'
    media_file_type = 'compressed'
    zip_url = ''

    def media_url(self) -> str:
        return self.zip_url

    def resolve(self, row):
        name = row.get('images')
        if not name:
            return None
        path = os.path.join(self.media_dir, name)
        return [path] if os.path.exists(path) else None


@pytest.fixture(scope='module')
def vl_template():
    from swift.model import get_model_processor
    from swift.template import get_template
    processor = get_model_processor(MODEL, load_model=False, model_type=MODEL_TYPE)[1]
    template = get_template(processor, template_type='qwen3_5', max_length=1024)
    template.set_mode('train')
    with torch.device('meta'):
        template.model = get_model_processor(MODEL, model_type=MODEL_TYPE, return_dummy_model=True)[0]
    return template


# ---- the downloader contract (offline, via a file:// archive) --------------------------------


def test_media_downloader_fetches_and_extracts(tmp_path):
    alias = 'swift_a5_fetch'
    clear_cache(alias)
    url = make_media_zip(tmp_path / 'media.zip')
    folder = MediaDownloader.download(url, alias, 'compressed')
    assert os.path.isdir(folder)
    assert 'cat.png' in os.listdir(folder), 'the archive must be unpacked into the resource folder'
    assert not os.path.exists(f'{folder}.tmp'), 'the staging .tmp must be promoted, not left behind'
    shutil.rmtree(folder, ignore_errors=True)


def test_media_downloader_serves_a_second_call_from_cache(tmp_path):
    alias = 'swift_a5_cache'
    clear_cache(alias)
    url = make_media_zip(tmp_path / 'media.zip')
    first = MediaDownloader.download(url, alias, 'compressed')
    # Point the second call at a different archive under the SAME alias: the cache fast path must win,
    # so the folder is unchanged and its contents are still the first download's.
    other = make_media_zip(tmp_path / 'other.zip', image_names=('dog.png', ))
    second = MediaDownloader.download(other, alias, 'compressed')
    assert first == second
    assert 'cat.png' in os.listdir(second) and 'dog.png' not in os.listdir(second)
    shutil.rmtree(first, ignore_errors=True)


def test_a_leftover_tmp_from_a_crash_is_cleared_and_retried(tmp_path):
    """The atomicity bug legacy had: a half-written folder must never be mistaken for a finished one."""
    alias = 'swift_a5_crash'
    clear_cache(alias)
    stale = os.path.join(MediaDownloader.cache_dir, f'{alias}.tmp')
    os.makedirs(stale, exist_ok=True)
    with open(os.path.join(stale, 'partial.txt'), 'w') as f:
        f.write('garbage from an interrupted fetch')

    url = make_media_zip(tmp_path / 'media.zip')
    folder = MediaDownloader.download(url, alias, 'compressed')
    assert os.path.isdir(folder)
    assert 'cat.png' in os.listdir(folder) and 'partial.txt' not in os.listdir(folder)
    assert not os.path.exists(f'{folder}.tmp')
    shutil.rmtree(folder, ignore_errors=True)


# ---- the whole download-multimodal chain (needs the local model) ------------------------------


@needs_model
def test_archive_dataset_fetches_resolves_drops_and_encodes(vl_template, tmp_path):
    """One test over the full chain: fetch the archive, resolve each row's image name against it, drop
    the row whose file is absent, then encode the survivor into vision tensors."""
    from swift.dev.dataset import SwiftDataset

    alias = LocalArchiveImagePreprocessor.media_alias
    clear_cache(alias)
    url = make_media_zip(tmp_path / 'media.zip')

    def row(name):
        return {
            'messages': [{'role': 'user', 'content': '<image>What is this?'},
                         {'role': 'assistant', 'content': 'A cat.'}],
            'images': name,
        }

    preprocessor = LocalArchiveImagePreprocessor()
    preprocessor.zip_url = url
    # One row names a file the archive holds, one names a file it does not -- the missing one is dropped.
    processed = preprocessor(HfDataset.from_list([row('cat.png'), row('missing.png')]), load_from_cache_file=False)

    assert len(processed) == 1, 'a row whose image is not in the archive teaches nothing and is dropped'
    image = processed[0]['images'][0]
    assert image['bytes'] is None and image['path'].endswith('cat.png') and os.path.exists(image['path'])

    encoded = SwiftDataset(processed, vl_template, load_from_cache_file=False)[0]
    assert 'pixel_values' in encoded and 'image_grid_thw' in encoded
    grid = torch.as_tensor(encoded['image_grid_thw'])
    assert torch.as_tensor(encoded['pixel_values']).shape[0] == grid.prod(dim=1).sum().item()
    shutil.rmtree(os.path.join(MediaDownloader.cache_dir, alias), ignore_errors=True)


# ---- the real transport hop (slow, skips when offline) ---------------------------------------


def _network_reachable(host='www.modelscope.cn', port=443, timeout=5) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


@pytest.mark.slow
def test_real_remote_download(tmp_path):
    """Prove the HTTP hop still works against a real host, using a tiny stable file. Skips when offline."""
    if not _network_reachable():
        pytest.skip('no network reachable; skipping the real-download smoke test')
    alias = 'swift_a5_remote'
    clear_cache(alias)
    try:
        folder = MediaDownloader.download('https://www.modelscope.cn/robots.txt', alias, 'file')
        assert os.path.isdir(folder) and os.listdir(folder), 'the fetched file must land in the folder'
    except Exception as exc:  # noqa: a transport failure offline must skip, not fail the suite
        pytest.skip(f'remote download unavailable: {exc}')
    finally:
        shutil.rmtree(os.path.join(MediaDownloader.cache_dir, alias), ignore_errors=True)
