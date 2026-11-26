# Copyright (c) ModelScope Contributors. All rights reserved.
"""Media-sensitive sidecar keys (G31 media collision regression).

``_prompt_key`` used to hash ONLY the chat messages, so two dataset rows asking the same question
about DIFFERENT images shared one prompt id -- and therefore one set of ``{id}_c{j}.npz`` sidecar
filenames. The second group overwrote the first's rollout-token payload (data loss), and both jsonl
rows referenced the same NPZs. These tests pin the fix:

1. same messages + different media -> different keys AND different NPZ filenames;
2. text-only rows keep the historical md5(messages) key byte-identically (resume / GRPO group id /
   candidate-cache contract);
3. the media fingerprint is content-based: the same media re-loaded as another object instance,
   another ``{bytes,path}`` dict, or another path with identical bytes digests identically;
4. reverse control: dropping the media term reintroduces the collision, and genuinely identical
   media rows still share one key (correct dedup).

All pure: synthetic candidates and tiny in-memory / tmp-file media, no sampler, no model.
"""

import hashlib
import os

import json

from swift.dev.components.infer.component import (
    _CandidateCache,
    _dpo_row,
    _emit_grpo,
    _media_fingerprint,
    _prompt_key,
    _RolloutRecorder,
)
from swift.dev.config import InferConfig
from tests.v5.infer.conftest import make_candidate, write_cache_file

MESSAGES = [{"role": "user", "content": "<image>How many red dots are in this image?"}]
IMG_A = b"\x89PNG-fake-image-A" * 4
IMG_B = b"\x89PNG-fake-image-B" * 4


def _messages_md5(messages):
    """The historical (pre-fix) key: md5 of the messages json, nothing else."""
    return hashlib.md5(
        json.dumps(messages, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")
    ).hexdigest()


def _group(text):
    """One recorded candidate: the recorder needs input_ids/labels plus the response frame."""
    candidate = make_candidate(
        text,
        messages=MESSAGES + [{"role": "assistant", "content": text}],
        response_token_ids=[7, 8],
        rollout_logprobs=[-0.1, -0.2],
    )
    candidate["input_ids"] = [1, 2, 7, 8]
    candidate["labels"] = [-100, -100, 7, 8]
    return candidate


def _emit_media_row(tmp_path, image, subdir):
    row = {"messages": list(MESSAGES), "images": [image]}
    trajectory = {"messages": list(MESSAGES), "images": [image]}
    out_dir = tmp_path / subdir
    out_dir.mkdir()
    recorder = _RolloutRecorder(str(out_dir) + ".rollout_tokens", base_dir=str(out_dir), enabled=True)
    out = _emit_grpo(row, trajectory, None, [_group("3"), _group("5")], None, InferConfig(), recorder)[0]
    return out, sorted(os.listdir(str(out_dir) + ".rollout_tokens"))


def test_same_messages_different_media_get_different_keys_and_npz_names(tmp_path):
    row_a = {"messages": MESSAGES, "images": [IMG_A]}
    row_b = {"messages": MESSAGES, "images": [IMG_B]}
    key_a = _prompt_key(MESSAGES, row_a)
    key_b = _prompt_key(MESSAGES, row_b)
    assert key_a != key_b  # the collision is gone at the key level
    out_a, files_a = _emit_media_row(tmp_path, IMG_A, "a")
    out_b, files_b = _emit_media_row(tmp_path, IMG_B, "b")
    assert out_a["id"] == key_a and out_b["id"] == key_b
    assert set(files_a).isdisjoint(files_b)  # no shared NPZ filename -> no overwrite
    assert all(os.path.basename(p).startswith(out_a["id"]) for p in out_a["rollout_tokens"])
    assert all(os.path.basename(p).startswith(out_b["id"]) for p in out_b["rollout_tokens"])


def test_text_only_key_is_byte_identical_to_historical_md5():
    assert _prompt_key(MESSAGES) == _messages_md5(MESSAGES)
    assert _prompt_key(MESSAGES, {"messages": MESSAGES}) == _messages_md5(MESSAGES)
    # empty / null media columns are still text-only: the key must not move
    for empty in ([], None, {}):
        row = {"messages": MESSAGES, "images": empty, "videos": empty, "audios": empty, "objects": empty}
        assert _prompt_key(MESSAGES, row) == _messages_md5(MESSAGES)
        assert _media_fingerprint(row) == ""


def test_media_fingerprint_is_content_based_not_instance_or_path(tmp_path):
    # same bytes as another object instance, and as an HF {bytes,path} dict with a transient path
    assert _prompt_key(MESSAGES, {"images": [IMG_A]}) == _prompt_key(MESSAGES, {"images": [bytes(IMG_A)]})
    assert _prompt_key(MESSAGES, {"images": [IMG_A]}) == _prompt_key(
        MESSAGES, {"images": [{"bytes": IMG_A, "path": "/tmp/whatever-a.png"}]}
    )
    assert _prompt_key(MESSAGES, {"images": [IMG_A]}) == _prompt_key(
        MESSAGES, {"images": [{"path": "/tmp/another-name.png", "bytes": IMG_A}]}
    )  # dict order irrelevant
    # same content at two different real paths -> one key; different content, same basename -> two
    p1, p2, p3 = tmp_path / "one.png", tmp_path / "two.png", tmp_path / "sub" / "one.png"
    p3.parent.mkdir()
    p1.write_bytes(IMG_A)
    p2.write_bytes(IMG_A)
    p3.write_bytes(IMG_B)
    assert _prompt_key(MESSAGES, {"images": [str(p1)]}) == _prompt_key(MESSAGES, {"images": [str(p2)]})
    assert _prompt_key(MESSAGES, {"images": [str(p1)]}) != _prompt_key(MESSAGES, {"images": [str(p3)]})
    # a path-only dict reads the file's bytes, matching the raw-bytes form of the same content
    assert _prompt_key(MESSAGES, {"images": [{"bytes": None, "path": str(p1)}]}) == _prompt_key(
        MESSAGES, {"images": [IMG_A]}
    )
    # PIL instances: two opens of the same file are one prompt; a different image is another
    from PIL import Image

    real_a, real_a_copy, real_b = tmp_path / "a.png", tmp_path / "a_copy.png", tmp_path / "b.png"
    Image.new("RGB", (4, 4), "red").save(real_a)
    Image.new("RGB", (4, 4), "red").save(real_a_copy)
    Image.new("RGB", (4, 4), "blue").save(real_b)
    pil_a1, pil_a2, pil_b = Image.open(real_a), Image.open(real_a_copy), Image.open(real_b)
    assert _prompt_key(MESSAGES, {"images": [pil_a1]}) == _prompt_key(MESSAGES, {"images": [pil_a2]})
    assert _prompt_key(MESSAGES, {"images": [pil_a1]}) != _prompt_key(MESSAGES, {"images": [pil_b]})


def test_media_fingerprint_distinguishes_columns_and_order():
    # the same bytes under a different media column is a different prompt
    assert _prompt_key(MESSAGES, {"images": [IMG_A]}) != _prompt_key(MESSAGES, {"videos": [IMG_A]})
    # list order maps to placeholder order: swapping two media items swaps the prompt
    assert _prompt_key(MESSAGES, {"images": [IMG_A, IMG_B]}) != _prompt_key(MESSAGES, {"images": [IMG_B, IMG_A]})
    # the grounding objects column rides the rendered prompt too
    assert _prompt_key(MESSAGES, {"objects": [{"bbox": [1, 2, 3, 4]}]}) != _prompt_key(MESSAGES)


def test_reverse_control_dropping_media_term_reintroduces_collision(tmp_path, monkeypatch):
    row_a = {"messages": MESSAGES, "images": [IMG_A]}
    row_b = {"messages": MESSAGES, "images": [IMG_B]}
    # WITHOUT the media term (the pre-fix computation) both rows collide on md5(messages) ...
    assert _messages_md5(row_a["messages"]) == _messages_md5(row_b["messages"])
    # ... and with it they diverge: the fingerprint is load-bearing.
    assert _media_fingerprint(row_a) and _media_fingerprint(row_b)
    assert _prompt_key(MESSAGES, row_a) != _prompt_key(MESSAGES, row_b)
    # genuinely identical media rows still share one key (correct dedup, not row-ordinal keys)
    twin = {"messages": MESSAGES, "images": [bytes(IMG_A)]}
    assert _prompt_key(MESSAGES, row_a) == _prompt_key(MESSAGES, twin)
    # end-to-end single point: neutralising the fingerprint brings the NPZ overwrite BACK ...
    import swift.dev.components.infer.component as component

    monkeypatch.setattr(component, "_media_fingerprint", lambda row: "")
    out_a, files_a = _emit_media_row(tmp_path, IMG_A, "red_a")
    out_b, files_b = _emit_media_row(tmp_path, IMG_B, "red_b")
    assert out_a["id"] == out_b["id"] == _messages_md5(MESSAGES)
    assert set(files_a) == set(files_b)  # same filenames -> second group overwrites the first
    # ... and restoring it removes the collision again.
    monkeypatch.undo()
    out_a, files_a = _emit_media_row(tmp_path, IMG_A, "green_a")
    out_b, files_b = _emit_media_row(tmp_path, IMG_B, "green_b")
    assert out_a["id"] != out_b["id"]
    assert set(files_a).isdisjoint(files_b)


def test_dpo_row_id_uses_the_same_media_sensitive_key():
    row = {"messages": MESSAGES, "images": [IMG_A]}
    trajectory = {"messages": MESSAGES, "images": [IMG_A]}
    out = _dpo_row(row, trajectory, _group("3"), _group("5"), "3")
    assert out["id"] == _prompt_key(MESSAGES, row)
    text_only = _dpo_row({"messages": MESSAGES}, {"messages": MESSAGES}, _group("3"), _group("5"), "3")
    assert text_only["id"] == _messages_md5(MESSAGES)


def test_candidate_cache_text_only_hits_unchanged_and_media_does_not_cross_serve(tmp_path):
    # text-only: a cache written by an OLD run (md5(messages) keys) still serves this run
    text_cache = write_cache_file(str(tmp_path / "text.jsonl"), [(MESSAGES, ["3", "5"])])
    cache = _CandidateCache([text_cache])
    assert cache.lookup({"messages": MESSAGES}, 2) is not None
    # media: an entry stored for image A (as a path-only dict, the jsonl-safe reload form) must
    # NOT serve a prompt asking about image B, and must serve image A given as raw bytes
    img_path = tmp_path / "stored.png"
    img_path.write_bytes(IMG_A)
    media_entry = {
        "messages": MESSAGES + [{"role": "assistant", "content": "3"}],
        "images": [{"bytes": None, "path": str(img_path)}],
    }
    media_cache_path = tmp_path / "media.jsonl"
    media_cache_path.write_text(json.dumps(media_entry) + "\n", encoding="utf-8")
    media_cache = _CandidateCache([str(media_cache_path)])
    assert media_cache.lookup({"messages": MESSAGES, "images": [IMG_A]}, 1) is not None
    assert media_cache.lookup({"messages": MESSAGES, "images": [IMG_B]}, 1) is None
    assert media_cache.lookup({"messages": MESSAGES}, 1) is None
