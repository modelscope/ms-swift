import pytest
import torch


def _assert_preprocessed_logprobs(result, batched_logits, sampled_ids, top_logprobs):
    for step, logits in enumerate(batched_logits):
        expected = torch.log_softmax(logits, dim=-1)
        k = min(top_logprobs or 1, logits.shape[-1])
        expected_top_ids = torch.topk(expected, k, dim=-1).indices
        for batch in range(logits.shape[0]):
            sampled_id = sampled_ids[batch, step].item()
            expected_ids = {sampled_id}
            expected_ids.update(expected_top_ids[batch].tolist())

            assert result[batch][step].keys() == expected_ids
            for token_id, logprob in result[batch][step].items():
                assert logprob == pytest.approx(expected[batch, token_id].item(), abs=1e-6)


@pytest.mark.parametrize('dtype', [torch.float16, torch.float64])
@pytest.mark.parametrize('top_logprobs', [None, 0, 2, 10])
def test_preprocess_logits_matches_log_softmax(top_logprobs, dtype):
    from swift.infer_engine import TransformersEngine

    batched_logits = [
        torch.tensor([[0.1, 0.2, 2.0, -1.0], [3.0, 0.5, -0.5, 1.0]], dtype=dtype),
        torch.tensor([[1.5, -1.0, 0.0, 2.5], [-2.0, 1.0, 2.0, 0.5]], dtype=dtype),
    ]
    # Cover sampled tokens both inside and outside the requested top-k.
    sampled_ids = torch.tensor([[0, 3], [3, 0]])

    result = TransformersEngine.preprocess_logits(batched_logits, sampled_ids, top_logprobs)

    _assert_preprocessed_logprobs(result, batched_logits, sampled_ids, top_logprobs)


def test_preprocess_logits_handles_missing_and_empty_logits():
    from swift.infer_engine import TransformersEngine

    sampled_ids = torch.empty((2, 0), dtype=torch.long)
    assert TransformersEngine.preprocess_logits(None, sampled_ids, 2) is None
    assert TransformersEngine.preprocess_logits([], sampled_ids, 2) == [[], []]
