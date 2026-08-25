# Copyright (c) ModelScope Contributors. All rights reserved.
"""Tests for dev dataset/dataloader APIs."""
import numpy as np
import pytest
import sys
import torch
from datasets import Dataset as HfDataset
from torch.utils.data import DataLoader, Dataset, IterableDataset
from unittest.mock import MagicMock, patch


def make_mock_template(max_length=2048, truncation_strategy='delete'):
    template = MagicMock()
    template.max_length = max_length
    template.truncation_strategy = truncation_strategy
    template.encode = MagicMock(return_value={'input_ids': [1, 2, 3], 'labels': [1, 2, 3], 'lengths': 3})
    return template


def make_mock_dataset(num_samples=100):
    return HfDataset.from_dict({'messages': [f'msg_{i}' for i in range(num_samples)]})


def make_encoded_dataset(num_samples=50, max_len=100):
    return HfDataset.from_dict({
        'input_ids': [[1, 2, 3]] * num_samples,
        'labels': [[1, 2, 3]] * num_samples,
        'lengths': [np.random.randint(10, max_len) for _ in range(num_samples)],
    })


class SimpleMapDataset(Dataset):

    def __init__(self, size=100):
        self.data = list(range(size))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {'input_ids': torch.tensor([idx])}


class SimpleIterableDataset(IterableDataset):

    def __init__(self, size=100):
        self.size = size

    def __iter__(self):
        for i in range(self.size):
            yield {'input_ids': torch.tensor([i])}


# === TestEncode (build_dataset's private encode helper: mode routing + split guard) ===


class TestEncode:

    @patch('swift.dataset.utils.LazyLLMDataset')
    def test_lazy_mode_returns_lazy_dataset(self, MockLazy):
        from swift.dev.builders.dataset import _encode
        dataset = make_mock_dataset(10)
        template = make_mock_template()
        result = _encode(dataset, template, mode='lazy', num_proc=1, strict=False, data_seed=42)
        MockLazy.assert_called_once()
        assert result == MockLazy.return_value

    @patch('swift.dataset.utils.AddLengthPreprocessor')
    def test_eager_mode_uses_add_length(self, MockAddLength):
        """Non-'split' eager encoding keeps rows raw + only adds `lengths` (AddLengthPreprocessor)."""
        from swift.dev.builders.dataset import _encode
        mock_instance = MagicMock()
        mock_instance.return_value = make_mock_dataset(5)
        MockAddLength.return_value = mock_instance
        dataset = make_mock_dataset(10)
        template = make_mock_template()
        _encode(dataset, template, mode='eager', num_proc=1, strict=False, data_seed=42)
        MockAddLength.assert_called_once_with(template)

    @patch('swift.dataset.utils.AddLengthPreprocessor')
    def test_stream_mode_uses_add_length(self, MockAddLength):
        from swift.dev.builders.dataset import _encode
        mock_instance = MagicMock()
        mock_result = MagicMock()
        mock_instance.return_value = mock_result
        MockAddLength.return_value = mock_instance
        dataset = MagicMock()
        template = make_mock_template()
        result = _encode(dataset, template, mode='stream', num_proc=1, strict=False, data_seed=42)
        MockAddLength.assert_called_once_with(template)
        assert result == mock_result

    def test_split_strategy_requires_eager(self):
        from swift.dev.builders.dataset import _encode
        dataset = make_mock_dataset(10)
        template = make_mock_template(truncation_strategy='split')
        with pytest.raises(ValueError, match="truncation_strategy='split' requires mode='eager'"):
            _encode(dataset, template, mode='lazy', num_proc=1, strict=False, data_seed=42)
        with pytest.raises(ValueError, match="truncation_strategy='split' requires mode='eager'"):
            _encode(dataset, template, mode='stream', num_proc=1, strict=False, data_seed=42)

    @patch('swift.dataset.utils.EncodePreprocessor')
    def test_split_strategy_with_eager_uses_full_encode(self, MockPreprocessor):
        """'split' expands one input into many samples, so it must fully encode (EncodePreprocessor)."""
        from swift.dev.builders.dataset import _encode
        mock_instance = MagicMock()
        mock_instance.return_value = make_mock_dataset(5)
        MockPreprocessor.return_value = mock_instance
        dataset = make_mock_dataset(10)
        template = make_mock_template(truncation_strategy='split')
        _encode(dataset, template, mode='eager', num_proc=1, strict=False, data_seed=42)
        MockPreprocessor.assert_called_once_with(template)


# === TestPack (build_dataset's private pack helper) ===


class TestPack:

    @patch('swift.dataset.packing.PackingDataset')
    def test_static_packing(self, MockPacking):
        from swift.dev.builders.dataset import _pack
        mock_instance = MagicMock()
        MockPacking.return_value = mock_instance
        dataset = make_encoded_dataset(20)
        template = make_mock_template(max_length=512)
        result = _pack(
            dataset,
            template,
            streaming=False,
            packing_length=256,
            packing_strategy='binpack',
            num_proc=1,
            strict=False)
        MockPacking.assert_called_once()
        assert MockPacking.call_args[1]['packing_length'] == 256
        assert result == mock_instance

    @patch('swift.dataset.packing.IterablePackingDataset')
    def test_streaming_packing(self, MockIterPacking):
        from swift.dev.builders.dataset import _pack
        mock_instance = MagicMock()
        MockIterPacking.return_value = mock_instance
        dataset = make_encoded_dataset(20)
        template = make_mock_template(max_length=512)
        result = _pack(
            dataset, template, streaming=True, packing_length=256, packing_strategy='binpack', num_proc=1, strict=False)
        MockIterPacking.assert_called_once()
        assert result == mock_instance

    @patch('swift.dataset.packing.PackingDataset')
    def test_default_packing_length_falls_back_to_template(self, MockPacking):
        from swift.dev.builders.dataset import _pack
        MockPacking.return_value = MagicMock()
        dataset = make_encoded_dataset(20)
        template = make_mock_template(max_length=1024)
        _pack(
            dataset,
            template,
            streaming=False,
            packing_length=None,
            packing_strategy='binpack',
            num_proc=1,
            strict=False)
        assert MockPacking.call_args[1]['packing_length'] == 1024


# === TestBuildDatasetEncode (encode flag + pack dependency) ===


class TestBuildDatasetEncode:
    """build_dataset's encode flag and its interaction with packing.

    encode=False is the RL path (defer tokenization to rollout); it must return the raw dataset
    unencoded, and it must fail fast when combined with map-style packing (which needs the `lengths`
    column that only encoding writes).
    """

    def _configs(self, *, packing=False, streaming=False):
        from swift.dev.config import DatasetConfig, DistributedConfig, TrainConfig
        dc = DatasetConfig(dataset=['d'], packing=packing, streaming=streaming)
        return dc, TrainConfig(), DistributedConfig()

    @patch('swift.dev.legacy_dataloader.build_dataloader')
    @patch('swift.dev.builders.dataset._encode')
    @patch('swift.dataset.load_dataset')
    def test_encode_false_skips_encoding(self, mock_load, mock_encode, mock_build_dl):
        from swift.dev.builders.dataset import build_dataset
        raw = make_mock_dataset(10)
        mock_load.return_value = (raw, None)
        mock_build_dl.return_value = 'LOADER'
        template = make_mock_template()
        dc, tc, dist = self._configs()
        train_loader, val_loader = build_dataset(dc, template, tc, dist, encode=False)
        # _encode must NOT be called; the raw dataset is passed straight to build_dataloader.
        mock_encode.assert_not_called()
        assert mock_build_dl.call_args[0][0] is raw
        assert train_loader == 'LOADER' and val_loader is None

    @patch('swift.dev.legacy_dataloader.build_dataloader')
    @patch('swift.dev.builders.dataset._encode')
    @patch('swift.dataset.load_dataset')
    def test_encode_true_encodes(self, mock_load, mock_encode, mock_build_dl):
        from swift.dev.builders.dataset import build_dataset
        raw = make_mock_dataset(10)
        mock_load.return_value = (raw, None)
        mock_encode.return_value = 'ENCODED'
        mock_build_dl.return_value = 'LOADER'
        template = make_mock_template()
        dc, tc, dist = self._configs()
        build_dataset(dc, template, tc, dist, encode=True)
        mock_encode.assert_called_once()
        assert mock_build_dl.call_args[0][0] == 'ENCODED'

    @patch('swift.dev.legacy_dataloader.build_dataloader')
    @patch('swift.dataset.load_dataset')
    def test_encode_false_with_map_packing_raises(self, mock_load, mock_build_dl):
        from swift.dev.builders.dataset import build_dataset
        mock_load.return_value = (make_mock_dataset(10), None)
        template = make_mock_template()
        dc, tc, dist = self._configs(packing=True, streaming=False)
        with pytest.raises(ValueError, match='requires streaming=True'):
            build_dataset(dc, template, tc, dist, encode=False)

    @patch('swift.dev.builders.dataset._pack', return_value='PACKED')
    @patch('swift.dev.legacy_dataloader.build_dataloader')
    @patch('swift.dataset.load_dataset')
    def test_encode_false_with_streaming_packing_ok(self, mock_load, mock_build_dl, mock_pack):
        from swift.dev.builders.dataset import build_dataset
        raw = make_mock_dataset(10)
        mock_load.return_value = (raw, None)
        template = make_mock_template()
        dc, tc, dist = self._configs(packing=True, streaming=True)
        # streaming packing encodes raw rows itself -> encode=False is legal; _pack gets the raw ds.
        build_dataset(dc, template, tc, dist, encode=False)
        assert mock_pack.call_args[0][0] is raw


# === TestBuildDatasetLoadOnce (problem 1: single load) + kwargs/shuffle (problem 2) ===


class TestBuildDatasetLoadOnce:
    """build_dataset must load the raw data once and pass the full legacy kwargs / shuffle semantics."""

    def _configs(self, **dc_kw):
        from swift.dev.config import DatasetConfig, DistributedConfig, TrainConfig
        return DatasetConfig(**dc_kw), TrainConfig(), DistributedConfig()

    @patch('swift.dev.legacy_dataloader.build_dataloader', return_value='L')
    @patch('swift.dev.builders.dataset._encode', return_value='E')
    @patch('swift.dataset.load_dataset')
    def test_split_val_loads_once(self, mock_load, _enc, _dl):
        """Val split off via split_dataset_ratio must NOT re-load: exactly one load_dataset call."""
        from swift.dev.builders.dataset import build_dataset
        mock_load.return_value = (make_mock_dataset(10), make_mock_dataset(2))
        dc, tc, dist = self._configs(dataset=['d'], split_dataset_ratio=0.2)
        train_loader, val_loader = build_dataset(dc, make_mock_template(), tc, dist)
        assert mock_load.call_count == 1
        assert train_loader == 'L' and val_loader == 'L'

    @patch('swift.dev.legacy_dataloader.build_dataloader', return_value='L')
    @patch('swift.dev.builders.dataset._encode', return_value='E')
    @patch('swift.dataset.load_dataset')
    def test_separate_val_dataset_loads_twice(self, mock_load, _enc, _dl):
        """A separate val_dataset is a distinct source -> legacy loads it with its own call."""
        from swift.dev.builders.dataset import build_dataset
        mock_load.side_effect = [
            (make_mock_dataset(10), None),  # train
            (None, make_mock_dataset(3)),  # separate val (split_ratio=1.0)
        ]
        dc, tc, dist = self._configs(dataset=['d'], val_dataset=['v'])
        build_dataset(dc, make_mock_template(), tc, dist)
        assert mock_load.call_count == 2
        # train load uses dataset_shuffle; val load uses val_dataset_shuffle + split_ratio=1.0
        assert mock_load.call_args_list[0].kwargs['shuffle'] is True
        assert mock_load.call_args_list[1].kwargs['shuffle'] is False
        assert mock_load.call_args_list[1].kwargs['split_dataset_ratio'] == 1.0

    @patch('swift.dev.legacy_dataloader.build_dataloader', return_value='L')
    @patch('swift.dev.builders.dataset._encode', return_value='E')
    @patch('swift.dataset.load_dataset')
    def test_full_kwargs_forwarded(self, mock_load, _enc, _dl):
        """All legacy get_dataset_kwargs keys must reach load_dataset (not the previous 6-key subset)."""
        from swift.dev.builders.dataset import build_dataset
        mock_load.return_value = (make_mock_dataset(10), None)
        dc, tc, dist = self._configs(
            dataset=['d'], interleave_prob=[0.5, 0.5], columns={'a': 'b'}, model_name=['n'], use_hf=True)
        build_dataset(dc, make_mock_template(), tc, dist)
        kw = mock_load.call_args.kwargs
        for key in ('interleave_prob', 'stopping_strategy', 'shuffle_buffer_size', 'columns', 'use_hf', 'hub_token',
                    'download_mode', 'remove_unused_columns', 'disable_auto_column_mapping', 'model_name',
                    'model_author'):
            assert key in kw, f'{key} not forwarded to load_dataset'
        assert kw['interleave_prob'] == [0.5, 0.5]
        assert kw['columns'] == {'a': 'b'}
        assert kw['use_hf'] is True

    @patch('swift.dev.legacy_dataloader.build_dataloader', return_value='L')
    @patch('swift.dev.builders.dataset._encode', return_value='E')
    @patch('swift.dataset.load_dataset')
    def test_load_time_vs_dataloader_shuffle(self, mock_load, _enc, mock_dl):
        """dataset_shuffle drives the load-time shuffle; train_dataloader_shuffle drives the sampler."""
        from swift.dev.builders.dataset import build_dataset
        from swift.dev.config import DatasetConfig, DistributedConfig, TrainConfig
        mock_load.return_value = (make_mock_dataset(10), None)
        dc = DatasetConfig(dataset=['d'], dataset_shuffle=True)
        tc = TrainConfig()
        dc.train_dataloader_shuffle = False
        build_dataset(dc, make_mock_template(), tc, DistributedConfig())
        # load-time shuffle = dataset_shuffle (True)
        assert mock_load.call_args.kwargs['shuffle'] is True
        # dataloader-sampler shuffle = train_dataloader_shuffle (False), decoupled from load-time
        assert mock_dl.call_args.kwargs['shuffle'] is False


# === TestDpShardFromMpu (config-derived shard source: mode + backend, no env/runtime probe) ===


class TestDpShardFromMpu:
    """_dp_shard_in_loader must be derived from DistributedConfig (mode+backend) only -- never env.

    Env probing would be wrong: on the driver TWINKLE_MODE is unset even in ray mode, so it would
    misreport 'local' and double-shard against slice_dp.
    """

    def _dc(self, **kw):
        from swift.dev.config import DistributedConfig
        return DistributedConfig(**kw)

    def test_megatron_local_true(self):
        from swift.dev.builders.dataset import _dp_shard_in_loader
        assert _dp_shard_in_loader(self._dc(backend='megatron', mode='local')) is True

    def test_megatron_ray_false(self):
        from swift.dev.builders.dataset import _dp_shard_in_loader

        # Ray scatters on the driver (slice_dp) -> bare loader; the dataloader must NOT self-shard.
        assert _dp_shard_in_loader(self._dc(backend='megatron', mode='ray')) is False

    def test_hf_local_false(self):
        from swift.dev.builders.dataset import _dp_shard_in_loader

        # transformers path (TP/PP/CP==1) shards by global rank in BatchSamplerShard.
        assert _dp_shard_in_loader(self._dc(backend='hf', mode='local')) is False

    def test_hf_ray_false(self):
        from swift.dev.builders.dataset import _dp_shard_in_loader
        assert _dp_shard_in_loader(self._dc(backend='hf', mode='ray')) is False


# === TestDropLast (partial-batch policy is a backend contract, not a preference) ===


class TestDropLast:
    """drop_last must follow the backend's batch-size contract, matching legacy on both paths.

    On Megatron both splits drop the remainder, by different mechanisms: the train sampler
    (MegatronPretrainingRandomSampler) has no drop_last knob and discards unconditionally via
    last_batch_size/active_total_samples (batch_sampler.py:100,114), while the val sampler
    (MegatronPretrainingSampler) takes drop_last=True by default (batch_sampler.py:18).

    dev previously used the factory default of False everywhere, which trained the remainder batch;
    once an epoch boundary fell mid-run that put dev a full step behind legacy for the rest of the run
    (measured on a 50-step dense.sh comparison: 500 samples at global_batch_size 16 = 31.25
    steps/epoch, and from step 32 on cli[i] matched legacy[i-1], not legacy[i]).
    """

    def _dc(self, **kw):
        from swift.dev.config import DistributedConfig
        return DistributedConfig(**kw)

    def test_megatron_train_drops_remainder(self):
        from swift.dev.builders.dataset import _drop_last
        assert _drop_last(self._dc(backend='megatron', mode='local'), is_val=False) is True

    def test_megatron_ray_also_drops_remainder(self):
        from swift.dev.builders.dataset import _drop_last

        # The global-batch invariant holds regardless of who performs the DP scatter, so ray mode
        # must not differ from local here (unlike _dp_shard_in_loader, which does).
        assert _drop_last(self._dc(backend='megatron', mode='ray'), is_val=False) is True

    def test_hf_train_keeps_remainder(self):
        from swift.dev.builders.dataset import _drop_last

        # legacy defers to TrainingArguments.dataloader_drop_last, whose default is False. Flipping
        # this globally would silently discard training samples on the transformers path.
        assert _drop_last(self._dc(backend='hf', mode='local'), is_val=False) is False

    def test_megatron_eval_also_drops_matching_legacy(self):
        from swift.dev.builders.dataset import _drop_last

        # legacy's val sampler is MegatronPretrainingSampler, whose drop_last defaults to True and
        # which base.py:990-996 constructs without overriding -- so legacy drops the eval remainder.
        # dev matched the opposite ('never drop for eval') until this was checked: with a val set that
        # is not a multiple of micro_batch_size*dp_size, the two pipelines would score DIFFERENT
        # sample sets, which is worse than legacy's known habit of scoring slightly fewer samples.
        assert _drop_last(self._dc(backend='megatron', mode='local'), is_val=True) is True

    def test_hf_eval_keeps_remainder(self):
        from swift.dev.builders.dataset import _drop_last

        # Unchanged on the transformers path: legacy defers to dataloader_drop_last (default False).
        assert _drop_last(self._dc(backend='hf', mode='local'), is_val=True) is False

    @patch('swift.dev.legacy_dataloader.build_dataloader', return_value='L')
    @patch('swift.dev.builders.dataset._encode', return_value='E')
    @patch('swift.dataset.load_dataset')
    def test_build_dataset_forwards_drop_last_per_backend(self, mock_load, _enc, mock_dl):
        """The policy is worthless if build_dataset does not actually pass it down.

        Guards the wiring, which is exactly what was missing before: the factory default (False)
        reached the loader on every path because no caller ever supplied drop_last.
        """
        from swift.dev.builders.dataset import build_dataset
        from swift.dev.config import DatasetConfig, DistributedConfig, TrainConfig
        for backend, expected in (('megatron', True), ('hf', False)):
            mock_load.return_value = (make_mock_dataset(10), None)
            # nproc_per_node is mandatory on the Megatron path (build_device_mesh fails fast without
            # it); harmless on hf, so pass it for both to keep the two cases symmetric.
            build_dataset(
                DatasetConfig(dataset=['d']), make_mock_template(), TrainConfig(),
                DistributedConfig(backend=backend, mode='local', nproc_per_node=1))
            assert mock_dl.call_args.kwargs['drop_last'] is expected, \
                f'backend={backend} should pass drop_last={expected}'


# === TestBuildDataloader ===


class TestBuildDataloader:

    def test_sequence_parallel_not_implemented(self):
        """SP dataloader is intentionally disabled this phase -> fail fast, not a silent wrong path."""
        from swift.dev.legacy_dataloader.factory import build_dataloader
        with pytest.raises(NotImplementedError, match='SP dataloader is not implemented'):
            build_dataloader(SimpleMapDataset(8), lambda x: x, batch_size=4, sequence_parallel_size=2)

    @patch('swift.dev.legacy_dataloader.factory._is_dist_initialized', return_value=False)
    @patch('swift.dev.legacy_dataloader.factory.DataLoaderShard')
    @patch('swift.dev.legacy_dataloader.factory.BatchSamplerShard')
    def test_map_style_dataset(self, MockBatchSampler, MockDataLoaderShard, mock_dist):
        from swift.dev.legacy_dataloader.factory import build_dataloader
        mock_loader = MagicMock(spec=DataLoader)
        MockDataLoaderShard.return_value = mock_loader
        dataset = SimpleMapDataset(50)
        result = build_dataloader(dataset, lambda x: x, batch_size=4, shuffle=True, resumable=False)
        MockBatchSampler.assert_called_once()
        MockDataLoaderShard.assert_called_once()
        assert result == mock_loader

    @patch('swift.dev.legacy_dataloader.factory._is_dist_initialized', return_value=False)
    @patch('swift.dev.legacy_dataloader.factory.DataLoaderDispatcher')
    def test_iterable_dataset(self, MockDispatcher, mock_dist):
        from swift.dev.legacy_dataloader.factory import build_dataloader
        mock_loader = MagicMock()
        MockDispatcher.return_value = mock_loader
        dataset = SimpleIterableDataset(20)
        result = build_dataloader(dataset, lambda x: x, batch_size=4, shuffle=False, resumable=False)
        MockDispatcher.assert_called_once()
        assert result == mock_loader

    @patch('swift.dev.legacy_dataloader.factory._is_dist_initialized', return_value=False)
    @patch('swift.dev.legacy_dataloader.factory.DataLoaderShard')
    @patch('swift.dev.legacy_dataloader.factory.BatchSamplerShard')
    def test_resumable_wrapper(self, MockBatchSampler, MockDataLoaderShard, mock_dist):
        from swift.dev.legacy_dataloader.factory import build_dataloader
        from swift.dev.legacy_dataloader.resumable import ResumableDataLoaderWrapper
        mock_loader = MagicMock(spec=DataLoader)
        mock_loader.batch_size = 4
        MockDataLoaderShard.return_value = mock_loader
        dataset = SimpleMapDataset(50)
        result = build_dataloader(
            dataset, lambda x: x, batch_size=4, shuffle=True, resumable=True, consumed_samples=100)
        assert isinstance(result, ResumableDataLoaderWrapper)
        assert result.consumed_samples == 100

    @patch('swift.dev.legacy_dataloader.factory._is_dist_initialized', return_value=False)
    @patch('swift.dev.legacy_dataloader.factory.DataLoaderShard')
    @patch('swift.dev.legacy_dataloader.factory.BatchSamplerShard')
    def test_non_resumable_default(self, MockBatchSampler, MockDataLoaderShard, mock_dist):
        from swift.dev.legacy_dataloader.factory import build_dataloader
        from swift.dev.legacy_dataloader.resumable import ResumableDataLoaderWrapper
        mock_loader = MagicMock(spec=DataLoader)
        MockDataLoaderShard.return_value = mock_loader
        dataset = SimpleMapDataset(50)
        result = build_dataloader(dataset, lambda x: x, batch_size=4, shuffle=True, resumable=False)
        assert not isinstance(result, ResumableDataLoaderWrapper)
        assert result == mock_loader

    def test_megatron_dp_sampler_shards_by_dp_rank(self):
        """_MegatronDPBatchSampler must shard the dataset across DP ranks with NO overlap and NO
        gap -- the DP-semantics gate for the mode='local' Megatron path.

        Simulates DP=2 by passing each rank's DP coordinate. Each rank's flattened index set must be
        disjoint from the other's, and their union must equal the full dataset (minus the drop_last
        remainder). Uses distinct indices per rank (not identical data) so a regression that shards
        by the wrong rank source, or replicates data to both ranks, is caught. shuffle=False keeps
        the expected order deterministic (natural sequential shard).

        Scope: this gates the shard ARITHMETIC given a DP coordinate. That the coordinate itself is
        right under tp*pp*cp>1 is a separate question, covered without GPUs by
        test_mesh_dp_matches_megatron_rank_generator (mesh coordinate == Megatron's DP grouping).
        """
        size, batch_size, dp_size = 40, 4, 2

        def _indices_for_rank(dp_rank):
            from swift.dev.legacy_dataloader.factory import _MegatronDPBatchSampler
            s = _MegatronDPBatchSampler(
                total_samples=size,
                batch_size=batch_size,
                shuffle=False,
                drop_last=False,
                data_seed=42,
                dp_rank=dp_rank,
                dp_world_size=dp_size)
            return [idx for batch in s for idx in batch]

        r0 = _indices_for_rank(0)
        r1 = _indices_for_rank(1)
        assert r0 and r1, f'empty shard: r0={r0} r1={r1}'
        assert set(r0).isdisjoint(set(r1)), f'DP shards overlap (data replicated?): r0={r0} r1={r1}'
        # natural sequential interleave: rank r gets indices r, r+dp, r+2dp, ...
        assert set(r0) | set(r1) == set(range(size)), \
            f'DP shards do not cover the dataset: union={sorted(set(r0) | set(r1))}'


class TestGroupByLength:
    """group_by_length: length-grouped batching (cut padding), mirroring legacy swift which reads
    a `lengths` column off the eagerly-encoded train dataset (trainers/mixin.py:1306-1309) and
    disables it for eval (_disable_group_by_length)."""

    def test_extract_lengths_reads_column(self):
        """_extract_lengths returns the eager `lengths` column verbatim."""
        from swift.dev.builders.dataset import _extract_lengths
        enc = make_encoded_dataset(num_samples=16)
        lengths = _extract_lengths(enc)
        assert lengths == list(enc['lengths'])

    def test_extract_lengths_fail_fast_when_no_column(self):
        """Lazy/streaming datasets have no `lengths` column -> actionable ValueError, not KeyError."""
        from swift.dev.builders.dataset import _extract_lengths
        no_len = HfDataset.from_dict({'input_ids': [[1, 2, 3]] * 4})
        with pytest.raises(ValueError, match='requires a `lengths` column'):
            _extract_lengths(no_len)

    @patch('swift.dev.legacy_dataloader.factory._is_dist_initialized', return_value=False)
    def test_build_dataloader_group_by_length_batches_similar_lengths(self, mock_dist):
        """End-to-end (real BatchSamplerShard, no GPU): with group_by_length + lengths, batches
        must contain similar-length samples. transformers get_length_grouped_indices groups within
        mega-batches, so we assert the mean intra-batch length spread is far smaller than a random
        batching baseline -- the property group_by_length exists to deliver (less padding), without
        over-claiming perfect global sorting. Needs enough samples for the mega-batch heuristic to
        bite (tiny datasets fall into a single mega-batch and barely group)."""
        import random as _random

        from swift.dev.legacy_dataloader.factory import build_dataloader
        _random.seed(0)
        lengths = [_random.randint(10, 1000) for _ in range(200)]

        def _mean_batch_spread(gbl: bool):
            loader = build_dataloader(
                SimpleMapDataset(len(lengths)),
                lambda x: x,
                batch_size=8,
                shuffle=True,
                group_by_length=gbl,
                lengths=(lengths if gbl else None),
                data_seed=0,
                resumable=False)
            spreads = [max(lengths[i] for i in b) - min(lengths[i] for i in b) for b in loader.batch_sampler]
            return sum(spreads) / len(spreads)

        grouped_spread = _mean_batch_spread(True)
        random_spread = _mean_batch_spread(False)
        # length-grouping must cut the average within-batch length spread substantially.
        assert grouped_spread < random_spread * 0.5, (f'group_by_length did not reduce intra-batch length spread '
                                                      f'(grouped={grouped_spread:.1f} vs random={random_spread:.1f})')

    def test_build_dataloader_group_by_length_requires_lengths(self):
        """BatchSamplerShard guard: group_by_length=True without lengths -> ValueError."""
        from swift.dev.legacy_dataloader.factory import build_dataloader
        with pytest.raises(ValueError, match='lengths must be provided'):
            build_dataloader(
                SimpleMapDataset(8), lambda x: x, batch_size=4, shuffle=True, group_by_length=True, lengths=None)


class TestDataSharding:
    """data_sharding (Megatron-only): shuffle WITHIN a DP shard instead of globally-then-stride.

    Ported from legacy MegatronPretrainingRandomSampler (batch_sampler.py:121-129) so existing
    Megatron user scripts that set --data_sharding keep working after the dev refactor.
    """

    def _sampler(self, total, batch_size, dp_rank, dp_world, data_sharding, **kw):
        """_MegatronDPBatchSampler with the DP coordinate passed in (no GPU / no megatron init)."""
        from swift.dev.legacy_dataloader.factory import _MegatronDPBatchSampler

        return _MegatronDPBatchSampler(
            total_samples=total,
            batch_size=batch_size,
            shuffle=True,
            drop_last=False,
            data_seed=0,
            dp_rank=dp_rank,
            dp_world_size=dp_world,
            data_sharding=data_sharding,
            **kw)

    def test_data_sharding_keeps_each_rank_in_its_own_contiguous_bucket(self):
        """shard-then-shuffle: rank r draws only from its own contiguous block, and the two ranks
        together still cover the dataset exactly once (no overlap, no loss)."""
        total, dp_world = 40, 2
        per_rank = total // dp_world
        idx = {}
        for r in range(dp_world):
            s = self._sampler(total, batch_size=4, dp_rank=r, dp_world=dp_world, data_sharding=True)
            idx[r] = [i for b in s for i in b]
            expected = set(range(r * per_rank, (r + 1) * per_rank))
            assert set(idx[r]) == expected, (
                f'rank {r} left its bucket: got {sorted(set(idx[r]))}, expected {sorted(expected)}')
        assert set(idx[0]).isdisjoint(idx[1])
        assert set(idx[0]) | set(idx[1]) == set(range(total))

    def test_data_sharding_shuffles_within_the_bucket(self):
        """It must still be a shuffle -- a contiguous bucket read in order would defeat the purpose."""
        s = self._sampler(40, batch_size=4, dp_rank=0, dp_world=2, data_sharding=True)
        order = [i for b in s for i in b]
        assert order != sorted(order), 'data_sharding did not shuffle within the bucket'

    def test_data_sharding_off_uses_global_permutation(self):
        """With data_sharding=False the base class strides a GLOBAL permutation, so a rank's indices
        are NOT confined to a contiguous bucket (this is the behavioural difference being ported)."""
        s = self._sampler(40, batch_size=4, dp_rank=0, dp_world=2, data_sharding=False)
        order = [i for b in s for i in b]
        assert not set(order).issubset(set(range(20))), \
            'data_sharding=False should draw from the whole dataset, not one contiguous bucket'

    def test_data_sharding_rejects_group_by_length(self):
        """Length grouping needs the global order; the sampler must not silently pick one.
        (build_dataset downgrades this combination earlier with a warning, matching legacy.)"""
        with pytest.raises(ValueError, match='incompatible with group_by_length'):
            self._sampler(
                40,
                batch_size=4,
                dp_rank=0,
                dp_world=2,
                data_sharding=True,
                group_by_length=True,
                lengths=list(range(40)))

    def test_build_dataloader_rejects_data_sharding_without_megatron(self):
        """data_sharding is meaningless without loader-side DP sharding -> fail, never ignore."""
        from swift.dev.legacy_dataloader.factory import build_dataloader
        with pytest.raises(ValueError, match='requires the Megatron backend'):
            build_dataloader(
                SimpleMapDataset(8),
                lambda x: x,
                batch_size=4,
                shuffle=True,
                data_sharding=True,
                dp_shard_in_loader=False)


# === TestResumableDataLoaderWrapper ===


class TestResumableDataLoaderWrapper:

    def _make_simple_dataloader(self, size=20, batch_size=4):
        return DataLoader(SimpleMapDataset(size), batch_size=batch_size)

    def test_get_state(self):
        from swift.dev.legacy_dataloader.resumable import ResumableDataLoaderWrapper
        wrapper = ResumableDataLoaderWrapper(self._make_simple_dataloader(), consumed_samples=0)
        state = wrapper.get_state()
        # state carries the resume triple (consumed_samples/consumed_batches/epoch)
        assert state['consumed_samples'] == 0
        assert state['epoch'] == 0
        assert state['consumed_batches'] == 0

    def test_skip_consumed_samples(self):
        from swift.dev.legacy_dataloader.resumable import ResumableDataLoaderWrapper
        wrapper = ResumableDataLoaderWrapper(self._make_simple_dataloader(), consumed_samples=0)
        wrapper.skip_consumed_samples(100)
        assert wrapper.consumed_samples == 100
        wrapper.skip_consumed_samples(-5)
        assert wrapper.consumed_samples == 0

    def test_consumed_samples_increments(self):
        from swift.dev.legacy_dataloader.resumable import ResumableDataLoaderWrapper
        wrapper = ResumableDataLoaderWrapper(self._make_simple_dataloader(20, 4), consumed_samples=0)
        list(wrapper)
        assert wrapper.consumed_samples == 20

    def test_initial_consumed_samples_skips(self):
        from swift.dev.legacy_dataloader.resumable import ResumableDataLoaderWrapper
        wrapper = ResumableDataLoaderWrapper(self._make_simple_dataloader(20, 4), consumed_samples=8)
        assert wrapper.consumed_samples == 8
        batches = list(wrapper)
        assert len(batches) == 3
        assert wrapper.consumed_samples == 20

    def test_set_epoch(self):
        from swift.dev.legacy_dataloader.resumable import ResumableDataLoaderWrapper
        wrapper = ResumableDataLoaderWrapper(self._make_simple_dataloader(), consumed_samples=0)
        wrapper.set_epoch(5)
        assert wrapper.get_state()['epoch'] == 5

    def test_len(self):
        from swift.dev.legacy_dataloader.resumable import ResumableDataLoaderWrapper
        loader = self._make_simple_dataloader(20, 4)
        wrapper = ResumableDataLoaderWrapper(loader, consumed_samples=0)
        assert len(wrapper) == len(loader)

    def test_dataset_property(self):
        from swift.dev.legacy_dataloader.resumable import ResumableDataLoaderWrapper
        loader = self._make_simple_dataloader(20, 4)
        wrapper = ResumableDataLoaderWrapper(loader, consumed_samples=0)
        assert wrapper.dataset is loader.dataset


# === TestResumeGuards ===


class TestEpochReshuffle:
    """Guards the per-epoch reshuffle contract: shuffle order must change per epoch.

    SFTLoop.fit calls dataloader.set_epoch(epoch) each epoch; BatchSamplerShard derives
    its shuffle seed as base_seed + epoch. A regression where set_epoch is dropped (the
    original SFTLoop bug) would make every epoch train on the identical order — silent
    training-quality loss. These assertions freeze the correct reshuffle behavior.
    """

    def _indices(self, epoch, seed=42, size=40, batch_size=4):
        from swift.dataloader.shard import BatchSamplerShard
        s = BatchSamplerShard(total_samples=size, batch_size=batch_size, shuffle=True, drop_last=False, data_seed=seed)
        s.set_epoch(epoch)
        return [idx for batch in s for idx in batch]

    def test_different_epochs_reshuffle(self):
        # epoch 0 and epoch 1 must NOT produce the same order (this is the bug we fixed).
        assert self._indices(0) != self._indices(1)

    def test_set_epoch_is_reproducible(self):
        # set_epoch(e) must be deterministic: same epoch -> same order (bit-identical).
        assert self._indices(3) == self._indices(3)

    def test_each_epoch_matches_independent_set_epoch(self):
        # The order at epoch e must equal an independently constructed sampler set to e.
        for e in (0, 1, 2):
            assert self._indices(e) == self._indices(e)

    def test_no_shuffle_is_natural_order(self):
        """shuffle=False must yield the dataset in natural sequential order (0,1,2,...).

        This is the property the dev-vs-legacy Megatron loss comparison
        (test_run_sft_e2e.py::test_run_sft_megatron_vs_legacy_loss) relies on to feed BOTH
        backends the same samples in the same order -- dev's dataloader must not silently reorder
        under dataset_shuffle=False, or that gate would compare different data. A regression that
        applied shuffling (or a non-identity permutation) despite shuffle=False would break the
        comparison's premise; this freezes the natural-order contract at the data layer (no GPU).
        """
        from swift.dataloader.shard import BatchSamplerShard
        size, batch_size = 40, 4
        s = BatchSamplerShard(total_samples=size, batch_size=batch_size, shuffle=False, drop_last=False, data_seed=42)
        order = [idx for batch in s for idx in batch]
        assert order == list(range(size)), f'shuffle=False reordered samples: {order[:8]}...'
        # Order is stable across epochs too (no per-epoch reshuffle when shuffle=False).
        s.set_epoch(1)
        assert [idx for batch in s for idx in batch] == list(range(size))

    @pytest.mark.parametrize('world,tp,pp,cp', [(2, 1, 1, 1), (4, 2, 1, 1), (4, 1, 2, 1), (4, 1, 1, 2), (8, 2, 2, 1),
                                                (8, 2, 1, 2)])
    def test_mesh_dp_matches_megatron_rank_generator(self, world, tp, pp, cp, monkeypatch):
        """The DeviceMesh's DP coordinate must equal the DP rank Megatron itself would assign.

        This is the load-bearing assumption of the mode='local' data path: the dataloader shards by
        DP rank taken from the mesh (build_device_mesh) because mpu does not exist yet, while the
        model later initializes mpu from that same mesh. If the two disagreed under TP/PP/CP, ranks
        that must see IDENTICAL data would silently get different data -- corrupt training, no
        error. Compared against Megatron's own RankGenerator with the mesh's order, so it needs no
        GPU and no distributed init, which is why it can cover layouts a 2-GPU test cannot.
        """
        from megatron.core.parallel_state import RankGenerator

        from swift.dev.builders import build_device_mesh
        from swift.dev.config import DistributedConfig

        dc = DistributedConfig(
            backend='megatron',
            mode='local',
            nproc_per_node=world,
            tensor_model_parallel_size=tp,
            pipeline_model_parallel_size=pp,
            context_parallel_size=cp)
        mesh_rank, order, dp_size = {}, None, None
        for rank in range(world):
            monkeypatch.setenv('RANK', str(rank))
            monkeypatch.setenv('WORLD_SIZE', str(world))
            mesh = build_device_mesh(dc)
            mesh_rank[rank] = mesh.dp_rank
            order, dp_size = mesh.order, mesh.dp_world_size

        # Megatron's DP rank of a global rank == its index inside its own DP group.
        groups = RankGenerator(tp=tp, ep=1, dp=world // (tp * pp * cp), pp=pp, cp=cp, order=order).get_ranks('dp')
        megatron_rank = {r: i for g in groups for i, r in enumerate(g)}

        assert dp_size == world // (tp * pp * cp)
        assert mesh_rank == megatron_rank, (f'mesh DP coordinate disagrees with Megatron (order={order}): '
                                            f'mesh={mesh_rank} megatron={megatron_rank} groups={groups}')


class TestResumeGuards:

    def test_streaming_resume_raises(self):
        """Resume + streaming/iterable must be refused (no deterministic epoch-aware skip).

        Iterable datasets can't reproduce the shuffle order for a cross-epoch skip, so run_sft
        raises rather than silently mis-skipping.
        """
        from swift.dev.config import (CheckpointConfig, DatasetConfig, DistributedConfig, ModelConfig, TemplateConfig,
                                       TrainConfig)
        from swift.dev.recipe import run_sft
        with pytest.raises(NotImplementedError, match='streaming/iterable'):
            run_sft(
                ModelConfig(model='dummy'),
                TemplateConfig(template='qwen2_5'),
                DatasetConfig(dataset=['dummy'], streaming=True),
                TrainConfig(max_steps=1),
                DistributedConfig(),
                CheckpointConfig(resume_from_checkpoint='/nonexistent/ckpt'),
            )


# === TestValidateConfigs (cross-config rules; pure function of the Configs, no I/O) ===


class TestValidateConfigs:
    """validate_configs is the single seam for constraints spanning several atomic Configs.

    Each rule exists to turn a SILENTLY-IGNORED knob into an actionable error: the user believing a
    feature is on while it does nothing is worse than a failed launch.
    """

    def _configs(self, **overrides):
        from swift.dev.config import DatasetConfig, DistributedConfig, ModelConfig, TemplateConfig, TrainConfig
        kw = {
            'model_config': ModelConfig(model='dummy'),
            'template_config': TemplateConfig(template='qwen2_5'),
            'dataset_config': DatasetConfig(dataset=['dummy']),
            'train_config': TrainConfig(),
            'distributed_config': DistributedConfig(backend='hf'),
        }
        kw.update(overrides)
        return kw

    def _validate(self, **overrides):
        from swift.dev.config import validate_configs
        validate_configs(**self._configs(**overrides))

    def test_defaults_pass(self):
        """A default config set must validate -- the rules may not block the common path."""
        self._validate()

    # --- mutual exclusion: group_by_length ---

    def test_group_by_length_rejects_padding_free(self):
        """padding_free already removes all padding, so length grouping buys nothing and raises
        peak activation memory. Legacy Megatron rejected this; we enforce it on BOTH backends
        because padding_free is the same Template._data_collator path on each."""
        from swift.dev.config import DatasetConfig, TemplateConfig
        with pytest.raises(ValueError, match='incompatible with padding_free'):
            self._validate(
                dataset_config=DatasetConfig(dataset=['d'], group_by_length=True, lazy_tokenize=False),
                template_config=TemplateConfig(template='qwen2_5', padding_free=True))

    def test_group_by_length_rejects_padding_free_on_megatron_too(self):
        """Same rule on the Megatron backend (where padding_free defaults to True)."""
        from swift.dev.config import DatasetConfig, DistributedConfig, TemplateConfig
        with pytest.raises(ValueError, match='incompatible with padding_free'):
            self._validate(
                dataset_config=DatasetConfig(dataset=['d'], group_by_length=True, lazy_tokenize=False),
                template_config=TemplateConfig(template='qwen2_5', padding_free=True),
                distributed_config=DistributedConfig(backend='megatron'))

    def test_group_by_length_rejects_packing(self):
        """packing bin-packs to ~packing_length (a stronger form of length organisation)."""
        from swift.dev.config import DatasetConfig
        with pytest.raises(ValueError, match='incompatible with packing'):
            self._validate(
                dataset_config=DatasetConfig(dataset=['d'], group_by_length=True, packing=True, lazy_tokenize=False))

    def test_group_by_length_rejects_streaming(self):
        """Previously silently dropped in build_dataset; streaming has no `lengths` column."""
        from swift.dev.config import DatasetConfig
        with pytest.raises(ValueError, match='requires a map-style dataset'):
            self._validate(dataset_config=DatasetConfig(dataset=['d'], group_by_length=True, streaming=True))

    def test_group_by_length_rejects_lazy_tokenize(self):
        """`lengths` is only written by the EAGER encode; report the cause, not a missing column."""
        from swift.dev.config import DatasetConfig
        with pytest.raises(ValueError, match='requires lazy_tokenize=False'):
            self._validate(dataset_config=DatasetConfig(dataset=['d'], group_by_length=True, lazy_tokenize=True))

    # --- data_sharding ---

    def test_data_sharding_requires_shuffle(self):
        """data_sharding only changes the SCOPE of the reshuffle -> useless without shuffling."""
        from swift.dev.config import DatasetConfig, DistributedConfig
        with pytest.raises(ValueError, match='requires train_dataloader_shuffle=True'):
            self._validate(
                dataset_config=DatasetConfig(dataset=['d'], data_sharding=True, train_dataloader_shuffle=False),
                distributed_config=DistributedConfig(backend='megatron'))

    def test_data_sharding_with_group_by_length_is_not_fatal(self):
        """Legacy downgrades this pair with a warning (batch_sampler.py:86-90). Existing Megatron
        scripts setting both must keep launching -- build_dataset performs the downgrade."""
        from swift.dev.config import DatasetConfig, DistributedConfig
        self._validate(
            dataset_config=DatasetConfig(dataset=['d'], data_sharding=True, group_by_length=True, lazy_tokenize=False),
            distributed_config=DistributedConfig(backend='megatron'))

    # --- backend-specific knobs ---

    def test_megatron_only_knob_on_hf_backend_raises(self):
        """data_sharding is Megatron-only; on the transformers backend it would be ignored."""
        from swift.dev.config import DatasetConfig
        with pytest.raises(ValueError, match='only implemented by the megatron backend'):
            self._validate(dataset_config=DatasetConfig(dataset=['d'], data_sharding=True))

    def test_hf_only_knob_on_megatron_backend_raises(self):
        """deepspeed is a transformers-path concept; Megatron has its own distributed optimizer."""
        from swift.dev.config import DistributedConfig
        with pytest.raises(ValueError, match='only implemented by the transformers backend'):
            self._validate(distributed_config=DistributedConfig(backend='megatron', deepspeed='zero2'))

    def test_hf_only_tuner_knob_on_megatron_backend_raises(self):
        """use_galore lives on TunerConfig (it drives the optimizer branch), so the backend check
        must read it from there -- a stale 'train_config' holder would AttributeError instead."""
        from swift.dev.config import DistributedConfig, TunerConfig
        with pytest.raises(ValueError, match='only implemented by the transformers backend'):
            self._validate(
                distributed_config=DistributedConfig(backend='megatron'), tuner_config=TunerConfig(use_galore=True))

    def test_tuner_only_knobs_skipped_when_tuner_config_is_none(self):
        """tuner_config=None is full-param training: its knobs cannot be set, so the check must
        skip rather than blow up on a None holder."""
        from swift.dev.config import DistributedConfig
        self._validate(distributed_config=DistributedConfig(backend='megatron'), tuner_config=None)

    def test_normalized_falsy_off_value_is_not_an_opt_in(self):
        """SftArguments._init_fsdp rewrites an unset fsdp to `[]` while the Config default is None.
        A strict `!=` comparison would flag EVERY Megatron CLI run as having enabled FSDP, so the
        off-state test must treat all falsy values as 'not requested'."""
        from swift.dev.config import DistributedConfig
        self._validate(distributed_config=DistributedConfig(backend='megatron', fsdp=[]))

    def test_real_opt_in_still_caught_for_falsy_defaulted_knob(self):
        """The falsy tolerance above must not swallow a genuine value."""
        from swift.dev.config import DistributedConfig
        with pytest.raises(ValueError, match='only implemented by the transformers backend'):
            self._validate(distributed_config=DistributedConfig(backend='megatron', fsdp='fsdp2'))

    def test_lisa_knob_is_backend_checked_from_tuner_config(self):
        """LISA moved to TunerConfig with it; the backend check must follow the field."""
        from swift.dev.config import DistributedConfig, TunerConfig
        with pytest.raises(ValueError, match='only implemented by the transformers backend'):
            self._validate(
                distributed_config=DistributedConfig(backend='megatron'),
                tuner_config=TunerConfig(lisa_activated_layers=4))

    def test_parallel_size_gt_1_on_hf_backend_raises(self):
        """TP/PP/CP/EP > 1 needs Megatron; the transformers path cannot provide that layout.
        Default 1 == disabled, so the sizes are checked by the same _MEGATRON_ONLY table as the
        other single-backend knobs (no separate loop)."""
        from swift.dev.config import DistributedConfig
        with pytest.raises(ValueError, match='only implemented by the megatron backend'):
            self._validate(distributed_config=DistributedConfig(backend='hf', tensor_model_parallel_size=2))

    def test_parallel_size_1_on_hf_backend_passes(self):
        """The off-value (1) must not be mistaken for an opt-in."""
        from swift.dev.config import DistributedConfig
        self._validate(
            distributed_config=DistributedConfig(
                backend='hf',
                tensor_model_parallel_size=1,
                pipeline_model_parallel_size=1,
                context_parallel_size=1,
                expert_model_parallel_size=1))

    def test_megatron_backend_accepts_its_own_knobs(self):
        """The positive case: Megatron-only knobs validate on the Megatron backend."""
        from swift.dev.config import DatasetConfig, DistributedConfig
        self._validate(
            dataset_config=DatasetConfig(dataset=['d'], data_sharding=True),
            distributed_config=DistributedConfig(
                backend='megatron', tensor_model_parallel_size=2, sequence_parallel=True))

    # --- table integrity: the _HF_ONLY / _MEGATRON_ONLY rows themselves ---

    def test_single_backend_table_defaults_match_the_config_declarations(self):
        """Every row's 'off value' must equal the field's real default, or the rule silently dies.

        The tables reject a knob only when its value differs from the off-value recorded in the row.
        If a row's off-value drifts from the dataclass default (someone changes the default, or
        mistypes the row), then EITHER the rule never fires (row value == whatever users pass) OR it
        fires on every default run. Both failure modes are silent, which is why this is checked
        against the dataclass rather than restated by hand.

        This also catches a row naming a field that no longer exists.
        """
        import dataclasses

        from swift.dev.config import validate as validate_module
        holders = {
            'dataset_config': 'DatasetConfig',
            'train_config': 'TrainConfig',
            'distributed_config': 'DistributedConfig',
            'tuner_config': 'TunerConfig',
        }
        import swift.dev.config as configs_module
        tables = {
            '_HF_ONLY': validate_module._HF_ONLY,
            '_MEGATRON_ONLY': getattr(validate_module, '_MEGATRON_ONLY', ())
        }
        problems = []
        for table_name, table in tables.items():
            for holder, field_name, off_value in table:
                cls = getattr(configs_module, holders[holder], None)
                if cls is None:
                    problems.append(f'{table_name}: unknown holder {holder!r}')
                    continue
                fields = {f.name: f for f in dataclasses.fields(cls)}
                if field_name not in fields:
                    problems.append(
                        f'{table_name}: {holders[holder]} has no field {field_name!r} (renamed or removed?)')
                    continue
                declared = fields[field_name].default
                if declared is dataclasses.MISSING:
                    continue  # no simple default to compare against
                if declared != off_value:
                    problems.append(f'{table_name}: {holders[holder]}.{field_name} row says off-value '
                                    f'{off_value!r} but the dataclass default is {declared!r}')
        assert not problems, 'single-backend tables are out of sync with the Configs:\n  ' + \
            '\n  '.join(problems)

    def test_every_hf_only_knob_passes_at_its_off_value_on_megatron(self):
        """A default Megatron run must not trip any _HF_ONLY row.

        This is the guard that was missing when _HF_ONLY was introduced: rows were added without
        sweeping the existing call sites, so runs that happened to pass a knob at a NON-default value
        started failing one at a time as their tests were run (_megatron_sft_runner and, separately,
        test_run_sft_megatron_ga_equivalence both hit it with optim='adamw'). Pinning the off-value
        behaviour here means a newly added row can no longer break the default Megatron path
        unnoticed -- and the message points at the sweep that a new row requires.
        """
        from swift.dev.config import DistributedConfig, TrainConfig, TunerConfig
        from swift.dev.config import validate as validate_module
        for holder, field_name, off_value in validate_module._HF_ONLY:
            if holder == 'train_config':
                overrides = {'train_config': TrainConfig(**{field_name: off_value})}
            elif holder == 'tuner_config':
                overrides = {'tuner_config': TunerConfig(**{field_name: off_value})}
            elif holder == 'distributed_config':
                overrides = {
                    'distributed_config':
                    DistributedConfig(backend='megatron', nproc_per_node=1, **{field_name: off_value})
                }
            else:
                continue
            overrides.setdefault('distributed_config', DistributedConfig(backend='megatron', nproc_per_node=1))
            self._validate(**overrides)
