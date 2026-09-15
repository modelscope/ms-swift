import os
import torch
import unittest

from swift.utils import get_device

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'


def test_qwen2():
    import os

    from swift.model import get_model_processor
    model, tokenizer = get_model_processor('Qwen/Qwen2-7B-Instruct', load_model=False)
    print(f'model: {model}, tokenizer: {tokenizer}')
    # test hf
    model, tokenizer = get_model_processor('Qwen/Qwen2-7B-Instruct', load_model=False, use_hf=True)

    model, tokenizer = get_model_processor(
        'Qwen/Qwen2-7B-Instruct', torch_dtype=torch.float32, device_map=get_device(), attn_impl='flash_attn')
    print(f'model: {model}, tokenizer: {tokenizer}')


def test_modelscope_hub():
    from swift.model import get_model_processor
    model, tokenizer = get_model_processor('Qwen/Qwen2___5-Math-1___5B-Instruct/', load_model=False)


class TestMolmo2Registration(unittest.TestCase):

    def test_registration(self):
        from swift.model import MODEL_MAPPING, MLLMModelType
        from swift.template import TEMPLATE_MAPPING, TemplateType

        model_meta = MODEL_MAPPING[MLLMModelType.molmo2]
        self.assertEqual(model_meta.template, TemplateType.molmo2)
        self.assertEqual(model_meta.model_arch.arch_name, 'molmo')
        self.assertIn('Molmo2ForConditionalGeneration', model_meta.architectures)

        hf_model_ids = []
        for group in model_meta.model_groups:
            for model in group.models:
                hf_model_ids.append(model.hf_model_id)

        self.assertIn('allenai/Molmo2-4B', hf_model_ids)
        self.assertIn('allenai/Molmo2-8B', hf_model_ids)
        self.assertIn('allenai/Molmo2-O-7B', hf_model_ids)
        self.assertIn(TemplateType.molmo2, TEMPLATE_MAPPING)


class TestSeqClsArchitecturesRewrite(unittest.TestCase):
    """Regression coverage for #9704.

    The seq_cls / reranker patcher monkey-patches a ``score`` head onto the
    generation model class without swapping the class itself, so
    ``PreTrainedModel.save_pretrained`` would otherwise write the wrong
    ``architectures`` to ``config.json`` and break downstream vLLM deployment.
    """

    def test_seq_cls_architectures_rewrite(self):
        from swift.model.patcher import _seq_cls_architectures

        # Generation -> SequenceClassification
        self.assertEqual(_seq_cls_architectures(['Qwen2ForCausalLM']), ['Qwen2ForSequenceClassification'])
        self.assertEqual(
            _seq_cls_architectures(['Qwen3VLForConditionalGeneration']),
            ['Qwen3VLForSequenceClassification'])
        self.assertEqual(_seq_cls_architectures(['LlamaForCausalLM']), ['LlamaForSequenceClassification'])
        self.assertEqual(
            _seq_cls_architectures(['Qwen2VLForConditionalGeneration']),
            ['Qwen2VLForSequenceClassification'])

        # Already-seq_cls class is preserved (idempotent).
        self.assertEqual(
            _seq_cls_architectures(['BertForSequenceClassification']),
            ['BertForSequenceClassification'])

        # Multi-arch list: only the matching suffix is rewritten.
        self.assertEqual(
            _seq_cls_architectures(['FooForCausalLM', 'BarForConditionalGeneration']),
            ['FooForSequenceClassification', 'BarForSequenceClassification'])

        # Empty / None inputs are returned unchanged.
        self.assertEqual(_seq_cls_architectures([]), [])
        self.assertIsNone(_seq_cls_architectures(None))

    def test_seq_cls_architectures_unknown_suffix(self):
        """Unknown suffixes (custom architectures) are left alone — we don't
        silently invent a class name that may not exist in transformers."""
        from swift.model.patcher import _seq_cls_architectures

        self.assertEqual(_seq_cls_architectures(['MyCustomLMHeadModel']), ['MyCustomLMHeadModel'])
        self.assertEqual(
            _seq_cls_architectures(['MyCustomLMHeadModel', 'LlamaForCausalLM']),
            ['MyCustomLMHeadModel', 'LlamaForSequenceClassification'])


if __name__ == '__main__':
    test_qwen2()
    # test_modelscope_hub()
