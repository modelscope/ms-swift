# Copyright (c) ModelScope Contributors. All rights reserved.
"""Shared fixtures for the dev SFT feature tests.

The SFT suite drives the real recipes (``run_sft`` / ``run_seq_cls`` / ``run_embedding`` /
``run_reranker``) on real weights and asserts one task-type-agnostic post-condition: the recipe ran
optimizer steps, the loss stayed finite and in a sane (normalized, non-divergent) range, and a
self-describing checkpoint was written. Those checks and the tiny-dataset writers are factored here
so every test file states only what is specific to its dimension.

No mocks live in this suite: a test that passes must mean the capability really trains end to end.
Heavy tests are gated with ``@pytest.mark.slow`` (+ ``@pytest.mark.accel(N)`` when multi-GPU); run
them on a free card with ``CUDA_VISIBLE_DEVICES=<card> pytest swift/dev/tests/feature/sft -m slow``.
"""
import json
import os

import pytest

#: Smallest viable text model, shared by every task_type in this suite. A classification head
#: (seq_cls / reranker) is built on any causal LM, embedding rides last-token pooling, and
#: generative_reranker scores off the vocab head -- so one cached 0.5B covers all of them without a
#: per-type download. Dedicated embedding/reranker checkpoints are exercised separately where a
#: dimension calls for a real one.
MODEL_TEXT = 'Qwen/Qwen2.5-0.5B-Instruct'

#: A tiny balanced sentiment set with a REAL signal (positive/negative Chinese reviews), shared by
#: every task_type below. Real signal matters because these tests assert a normalized, non-divergent
#: loss and (in test_parity_grid) a legacy-vs-dev loss trajectory: on learnable data the loss is a
#: meaningful number rather than noise around ln(num_labels).
SENTIMENT = [
    ('这个包装很差，容易被调包。', 0),
    ('质量很好，物流也快，满意。', 1),
    ('收到就是坏的，客服还不理人。', 0),
    ('物超所值，会回购。', 1),
    ('味道不对，像是过期了。', 0),
    ('和描述一致，推荐购买。', 1),
    ('用了两天就坏了。', 0),
    ('非常棒，五星好评。', 1),
]


@pytest.fixture(scope='session')
def text_model_path():
    """Local path to the shared 0.5B text model, downloaded once per session."""
    from modelscope import snapshot_download
    return snapshot_download(MODEL_TEXT)


@pytest.fixture
def write_jsonl():
    """Return a writer that dumps a list of row dicts to a ``.jsonl`` path."""

    def _write(path, rows):
        with open(path, 'w') as f:
            for row in rows:
                f.write(json.dumps(row) + '\n')
        return str(path)

    return _write


@pytest.fixture
def assert_trained():
    """Return the task-type-agnostic post-condition checker every SFT recipe must satisfy.

    ``max_loss`` is the divergence / wrong-reduction guard, applied to the CONVERGED (final) loss.
    It is task-specific because the loss scales differ (seq_cls CE ~ ln(num_labels), embedding
    InfoNCE ~ ln(group), reranker BCE ~ ln(2)), so each caller passes a ceiling a correctly-normalized
    run settles under while a sum-reduced (scales with sequence count) or diverging one does not.

    The ceiling is on the final loss, not every step, on purpose: a freshly-initialized head starts
    far from the target scale -- a num_labels=1 regression head emits logits of magnitude ~6 (the same
    large init the seq_cls head shows), so its first MSE against 0/1 targets is ~37 before converging
    within a few steps. Bounding every step would be brittle to that init scale; bounding the converged
    loss still catches a wrong reduction (every step, hence the last, scales up) and a diverged run.
    """

    def _assert(history, out_dir, task_type, *, max_loss):
        assert history, f'{task_type}: recipe produced no optimizer steps'
        losses = [r['loss'] for r in history]
        # loss == loss is the NaN test (NaN != itself); not a typo.
        assert all(loss == loss and abs(loss) != float('inf') for loss in losses), \
            f'{task_type}: non-finite loss: {losses}'
        assert abs(losses[-1]) < max_loss, \
            f'{task_type}: converged loss out of sane range (<{max_loss}): {losses}'
        assert all('grad_norm' in r for r in history), f'{task_type}: grad_norm missing from history'

        ckpt = os.path.join(out_dir, 'checkpoint-final')
        assert os.path.isdir(ckpt), f'{task_type}: no final checkpoint at {ckpt}'
        files = set(os.listdir(ckpt))
        assert any(f.endswith('.safetensors') for f in files), f'{task_type}: no weights in {sorted(files)}'
        assert 'args.json' in files, f'{task_type}: checkpoint not self-describing: {sorted(files)}'
        with open(os.path.join(ckpt, 'args.json')) as f:
            args = json.load(f)
        # task_type is a force_load key: without it `swift infer <ckpt>` silently downgrades a
        # seq_cls/reranker/embedding checkpoint back to causal_lm.
        assert args.get('task_type') == task_type, \
            f'args.json task_type={args.get("task_type")!r}, expected {task_type!r} (infer would mis-load)'
        return losses

    return _assert


# --- task-type tiny dataset writers -------------------------------------------------------
# Row shapes are the documented dev/legacy formats (see the registered loaders in
# swift/dataset/dataset/llm.py and swift/dev/dataset/loader/llm.py). embedding and reranker both
# carry an anchor ``messages`` plus ``positive_messages``/``negative_messages``, where each of those
# is a LIST OF message-lists (one conversation per candidate), not a single conversation.

_POSITIVE_DOC = '这是一条正面评价，表示满意。'
_NEGATIVE_DOC = '这是一条负面评价，表示不满意。'


def seq_cls_rows(kind):
    """seq_cls rows in the ``{messages, label}`` format.

    ``kind='single_label'`` emits an integer class index; ``kind='regression'`` emits the same target
    as a float (a num_labels=1 head fits it with MSE). One scalar label per sequence -- this is the
    path that regressed before twinkle's ``to_tensor`` wrapped bare scalars (see processor/base.py).
    """
    def label_of(gold):
        return gold if kind == 'single_label' else float(gold)

    return [{'messages': [{'role': 'user', 'content': text}], 'label': label_of(gold)} for text, gold in SENTIMENT]


def embedding_rows():
    """embedding rows: anchor review vs a sentiment-matching positive and an opposite negative.

    InfoNCE contrasts the anchor against its positive/negative, so the positive shares the anchor's
    sentiment and the negative flips it -- a real signal rather than a constant 'unrelated' string.
    """
    return [{
        'messages': [{'role': 'user', 'content': text}],
        'positive_messages': [[{'role': 'user', 'content': _POSITIVE_DOC if gold == 1 else _NEGATIVE_DOC}]],
        'negative_messages': [[{'role': 'user', 'content': _NEGATIVE_DOC if gold == 1 else _POSITIVE_DOC}]],
    } for text, gold in SENTIMENT]


def reranker_rows():
    """reranker / generative_reranker rows: a query plus a relevant and an irrelevant candidate doc.

    Same nested shape as embedding. The reranker encode prepends the query to each candidate and
    labels relevant=1 / irrelevant=0; a pointwise (BCE) or listwise loss then trains the score.
    """
    return embedding_rows()


@pytest.fixture
def tiny_data(tmp_path, write_jsonl):
    """Write each task_type's tiny dataset under ``tmp_path`` and return the paths.

    A namespace of no-arg writers so a test names only the dataset it needs; each returns the written
    ``.jsonl`` path ready to hand to ``DatasetConfig``.
    """

    class _TinyData:

        def seq_cls(self, kind, tag):
            return write_jsonl(tmp_path / f'seq_cls_{tag}.jsonl', seq_cls_rows(kind))

        def embedding(self):
            return write_jsonl(tmp_path / 'embedding.jsonl', embedding_rows())

        def reranker(self):
            return write_jsonl(tmp_path / 'reranker.jsonl', reranker_rows())

    return _TinyData()
