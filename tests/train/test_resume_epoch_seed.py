from transformers import trainer as hf_trainer
from types import SimpleNamespace

from swift.trainers.mixin import DataLoaderMixin, SwiftMixin

DATA_SEED = 42
NUM_SAMPLES = 1000
SKIP_BATCHES = 600


class _DummyTrainer(DataLoaderMixin, SwiftMixin):
    """Expose only what get_train_dataloader/_patch_skip_first_batches need."""


def _make_trainer():
    trainer = _DummyTrainer.__new__(_DummyTrainer)
    trainer.template = SimpleNamespace(sequence_parallel_size=1)
    trainer.train_dataset = list(range(NUM_SAMPLES))
    trainer._train_batch_size = 1
    trainer.data_collator = lambda batch: batch
    trainer.accelerator = SimpleNamespace(device=None)
    trainer.args = SimpleNamespace(
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
        dataloader_persistent_workers=False,
        dataloader_prefetch_factor=None,
        dataloader_drop_last=False,
        train_dataloader_shuffle=True,
        data_seed=DATA_SEED,
        deepspeed=None,
        group_by_length=False,
        process_index=0,
    )
    return trainer


def _iterate(dataloader):
    return [batch[0] for batch in dataloader]


def _epoch_order(trainer, epoch):
    dataloader = trainer.get_train_dataloader()
    dataloader.set_epoch(epoch)
    return _iterate(dataloader)


def test_skip_first_batches_keeps_in_progress_epoch_permutation():
    # HF Trainer (transformers <= 4.x) calls set_epoch on the original dataloader
    # before skip_first_batches; the rebuilt dataloader must keep the epoch
    # permutation instead of replaying the epoch-0 order. Fixes #10050.
    trainer = _make_trainer()
    epoch1 = _epoch_order(trainer, 1)
    epoch0 = _epoch_order(trainer, 0)
    assert epoch1 != epoch0  # sanity: per-epoch permutations differ

    dataloader = trainer.get_train_dataloader()
    dataloader.set_epoch(1)
    with trainer._patch_skip_first_batches():
        resumed = hf_trainer.skip_first_batches(dataloader, SKIP_BATCHES)
    tail = _iterate(resumed)

    assert tail == epoch1[SKIP_BATCHES:]
    assert tail != epoch0[SKIP_BATCHES:]


def test_skip_first_batches_propagates_curr_seed():
    trainer = _make_trainer()
    dataloader = trainer.get_train_dataloader()
    dataloader.set_epoch(2)
    with trainer._patch_skip_first_batches():
        resumed = hf_trainer.skip_first_batches(dataloader, SKIP_BATCHES)
    # SkipBatchSampler wraps the rebuilt BatchSamplerShard
    assert resumed.batch_sampler.batch_sampler.curr_seed == DATA_SEED + 2


def test_skip_first_batches_without_set_epoch_keeps_epoch0_order():
    # Resuming inside epoch 0 (or without ever calling set_epoch) must keep
    # the original behavior: the epoch-0 permutation with the prefix skipped.
    trainer = _make_trainer()
    epoch0 = _epoch_order(trainer, 0)

    dataloader = trainer.get_train_dataloader()
    with trainer._patch_skip_first_batches():
        resumed = hf_trainer.skip_first_batches(dataloader, SKIP_BATCHES)

    assert _iterate(resumed) == epoch0[SKIP_BATCHES:]


def test_skip_first_batches_compatible_with_post_skip_set_epoch():
    # transformers >= 5.x applies set_epoch after skip_first_batches; the fix
    # must stay idempotent under that ordering.
    trainer = _make_trainer()
    epoch1 = _epoch_order(trainer, 1)

    dataloader = trainer.get_train_dataloader()
    dataloader.set_epoch(1)
    with trainer._patch_skip_first_batches():
        resumed = hf_trainer.skip_first_batches(dataloader, SKIP_BATCHES)
    resumed.set_epoch(1)

    assert _iterate(resumed) == epoch1[SKIP_BATCHES:]


def test_skip_first_batches_epoch0_resume_is_unchanged():
    # The bug is invisible when resuming inside epoch 0 because
    # data_seed + 0 == data_seed; keep it that way.
    trainer = _make_trainer()
    epoch0 = _epoch_order(trainer, 0)

    dataloader = trainer.get_train_dataloader()
    dataloader.set_epoch(0)
    with trainer._patch_skip_first_batches():
        resumed = hf_trainer.skip_first_batches(dataloader, SKIP_BATCHES)

    assert _iterate(resumed) == epoch0[SKIP_BATCHES:]


if __name__ == '__main__':
    for fn in [
            test_skip_first_batches_keeps_in_progress_epoch_permutation,
            test_skip_first_batches_propagates_curr_seed,
            test_skip_first_batches_without_set_epoch_keeps_epoch0_order,
            test_skip_first_batches_compatible_with_post_skip_set_epoch,
            test_skip_first_batches_epoch0_resume_is_unchanged,
    ]:
        fn()
        print(f'{fn.__name__}: PASSED')
