# Copyright (c) ModelScope Contributors. All rights reserved.
import torch
import unittest
from types import SimpleNamespace

from swift.model import MODEL_MAPPING, LLMModelType
from swift.model.models.jina import patch_jina_reranker_v3_forward
from swift.template import TEMPLATE_MAPPING, TemplateType
from swift.template.templates.llm import JinaRerankerV3Template


class DummyJinaReranker:

    def __init__(self):
        self.received_kwargs = None

    def forward(self, *args, **kwargs):
        self.received_kwargs = kwargs
        return SimpleNamespace(scores=torch.tensor([0.25, 0.75]), logits=None)


class TestJinaRerankerV3(unittest.TestCase):

    def test_registration(self):
        meta = MODEL_MAPPING[LLMModelType.jina_reranker_v3]
        self.assertEqual(meta.template, TemplateType.jina_reranker_v3)
        self.assertEqual(meta.task_type, 'reranker')
        self.assertEqual(meta.architectures, ['JinaForRanking'])
        model_ids = [model.hf_model_id for group in meta.model_groups for model in group.models]
        self.assertEqual(model_ids, ['jinaai/jina-reranker-v3', 'jinaai/jina-reranker-v3.5'])
        self.assertIn(TemplateType.jina_reranker_v3, TEMPLATE_MAPPING)

    def test_prompt_format(self):
        prompt = JinaRerankerV3Template.format_prompt(
            'What is RAG?<|rerank_token|>',
            'RAG augments LLMs with retrieved documents.<|embed_token|>',
            'Prefer technically precise passages.',
        )
        self.assertIn('<|im_start|>system\n', prompt)
        self.assertIn('<instruct>\nPrefer technically precise passages.\n</instruct>', prompt)
        self.assertIn('<passage id="0">\nRAG augments LLMs with retrieved documents.<|embed_token|>\n</passage>',
                      prompt)
        self.assertIn('<query>\nWhat is RAG?<|rerank_token|>\n</query>', prompt)
        self.assertTrue(prompt.endswith('<|im_start|>assistant\n<think>\n\n</think>\n\n'))
        self.assertEqual(prompt.count('<|embed_token|>'), 1)
        self.assertEqual(prompt.count('<|rerank_token|>'), 1)

    def test_forward_patch_exposes_scores_as_logits(self):
        model = DummyJinaReranker()
        patch_jina_reranker_v3_forward(model)
        output = model.forward(input_ids=torch.tensor([[1, 2]]), labels=torch.tensor([1]))
        self.assertNotIn('labels', model.received_kwargs)
        self.assertEqual(tuple(output.logits.shape), (2, 1))
        torch.testing.assert_close(output.logits.squeeze(-1), output.scores)


if __name__ == '__main__':
    unittest.main()
