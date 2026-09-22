import copy
import os
import pickle
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


def _patched_qwen2_seq_cls(architectures=('Qwen2ForCausalLM', )):
    """A tiny patched seq_cls model (regression helper for #9704)."""
    from transformers import Qwen2Config, Qwen2ForCausalLM
    from types import SimpleNamespace

    from swift.model.patcher import _patch_sequence_classification

    config = Qwen2Config(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        num_labels=2,
        architectures=list(architectures) if architectures else architectures)
    model = Qwen2ForCausalLM(config)
    _patch_sequence_classification(model, SimpleNamespace(model_arch=None))
    return model


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
            _seq_cls_architectures(['Qwen3VLForConditionalGeneration']), ['Qwen3VLForSequenceClassification'])
        self.assertEqual(_seq_cls_architectures(['LlamaForCausalLM']), ['LlamaForSequenceClassification'])
        self.assertEqual(
            _seq_cls_architectures(['Qwen2VLForConditionalGeneration']), ['Qwen2VLForSequenceClassification'])

        # Already-seq_cls class is preserved (idempotent).
        self.assertEqual(_seq_cls_architectures(['BertForSequenceClassification']), ['BertForSequenceClassification'])

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

    def test_seq_cls_architectures_saved_config(self):
        """The seq_cls architecture must survive ``save_pretrained``.

        ``PreTrainedModel.save_pretrained`` resets ``config.architectures`` to the
        model's class name, so without covering the save path the checkpoint keeps
        advertising the generation architecture (see #9704).
        """
        import json
        import tempfile
        from transformers import AutoConfig, Qwen2Config, Qwen2ForCausalLM
        from types import SimpleNamespace

        from swift.model.patcher import _patch_sequence_classification

        for arch, expected in [('Qwen2ForCausalLM', 'Qwen2ForSequenceClassification'),
                               ('Qwen2VLForConditionalGeneration', 'Qwen2VLForSequenceClassification'),
                               ('MyCustomLMHeadModel', 'MyCustomLMHeadModel')]:
            with self.subTest(arch=arch):
                config = Qwen2Config(
                    vocab_size=32,
                    hidden_size=16,
                    intermediate_size=32,
                    num_hidden_layers=1,
                    num_attention_heads=2,
                    num_key_value_heads=2,
                    num_labels=2,
                    architectures=[arch])
                model = Qwen2ForCausalLM(config)
                _patch_sequence_classification(model, SimpleNamespace(model_arch=None))

                with tempfile.TemporaryDirectory() as tmp_dir:
                    model.save_pretrained(tmp_dir)
                    with open(os.path.join(tmp_dir, 'config.json'), 'r') as f:
                        saved_config = json.load(f)
                    reloaded = AutoConfig.from_pretrained(tmp_dir)

                self.assertEqual(saved_config['architectures'], [expected])
                self.assertEqual(reloaded.architectures, [expected])

    def test_seq_cls_architectures_saved_config_deepcopy(self):
        """A deep-copied patched model must save its own weights.

        Binding the hook onto the instance captured the original model in a
        closure, so ``deepcopy(model).save_pretrained()`` silently wrote the
        original weights while looking fine on disk.
        """
        import json
        import tempfile
        from transformers import AutoModelForSequenceClassification

        model = _patched_qwen2_seq_cls()
        copied = copy.deepcopy(model)
        with torch.no_grad():
            copied.score.weight.fill_(0.123)

        with tempfile.TemporaryDirectory() as tmp_dir:
            copied.save_pretrained(tmp_dir)
            with open(os.path.join(tmp_dir, 'config.json'), 'r') as f:
                saved_config = json.load(f)
            reloaded = AutoModelForSequenceClassification.from_pretrained(tmp_dir)

        self.assertEqual(saved_config['architectures'], ['Qwen2ForSequenceClassification'])
        self.assertTrue(torch.allclose(reloaded.score.weight, torch.full_like(reloaded.score.weight, 0.123)))
        self.assertNotIn('save_pretrained', copied.config.__dict__)

    def test_seq_cls_architectures_saved_config_pickle(self):
        """A patched model preserves classifier metadata after a pickle round trip."""
        import json
        import tempfile

        model = _patched_qwen2_seq_cls()
        copied = pickle.loads(pickle.dumps(model))

        with tempfile.TemporaryDirectory() as tmp_dir:
            copied.save_pretrained(tmp_dir)
            with open(os.path.join(tmp_dir, 'config.json'), 'r') as f:
                saved_config = json.load(f)

        self.assertEqual(saved_config['architectures'], ['Qwen2ForSequenceClassification'])

    def test_seq_cls_architectures_non_main_then_main(self):
        import json
        import tempfile

        model = _patched_qwen2_seq_cls()
        with tempfile.TemporaryDirectory() as tmp_dir:
            model.save_pretrained(os.path.join(tmp_dir, 'non_main'), is_main_process=False)
            output_dir = os.path.join(tmp_dir, 'main')
            model.save_pretrained(output_dir)
            with open(os.path.join(output_dir, 'config.json')) as f:
                saved_config = json.load(f)
        self.assertEqual(saved_config['architectures'], ['Qwen2ForSequenceClassification'])
        self.assertEqual(model.config.architectures, ['Qwen2ForSequenceClassification'])
        self.assertNotIn('save_pretrained', model.config.__dict__)

    def test_seq_cls_architectures_pickle_fresh_process(self):
        import subprocess
        import sys
        import tempfile
        import textwrap

        model = _patched_qwen2_seq_cls()
        with torch.no_grad():
            model.score.weight.fill_(0.123)
        script = textwrap.dedent('''\
            import json
            import pickle
            import sys
            import torch
            from transformers import AutoModelForSequenceClassification

            with open(sys.argv[1], 'rb') as f:
                model = pickle.load(f)
            model.save_pretrained(sys.argv[2])
            with open(sys.argv[2] + '/config.json') as f:
                assert json.load(f)['architectures'] == ['Qwen2ForSequenceClassification']
            restored = AutoModelForSequenceClassification.from_pretrained(sys.argv[2])
            torch.testing.assert_close(restored.score.weight, torch.full_like(restored.score.weight, 0.123))
        ''')
        with tempfile.TemporaryDirectory() as tmp_dir:
            model_path = os.path.join(tmp_dir, 'model.pkl')
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            result = subprocess.run([sys.executable, '-c', script, model_path,
                                     os.path.join(tmp_dir, 'saved')],
                                    capture_output=True,
                                    text=True,
                                    timeout=60)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_seq_cls_architectures_save_failure_cleanup(self):
        import json
        import tempfile
        from unittest.mock import patch

        model = _patched_qwen2_seq_cls()
        with tempfile.TemporaryDirectory() as tmp_dir:
            with patch.object(type(model.config), 'save_pretrained', side_effect=OSError('save failed')):
                with self.assertRaisesRegex(OSError, 'save failed'):
                    model.save_pretrained(tmp_dir)
            self.assertEqual(model.config.architectures, ['Qwen2ForSequenceClassification'])
            self.assertNotIn('save_pretrained', model.config.__dict__)
            model.save_pretrained(tmp_dir)
            with open(os.path.join(tmp_dir, 'config.json')) as f:
                self.assertEqual(json.load(f)['architectures'], ['Qwen2ForSequenceClassification'])

    def test_seq_cls_architectures_failure_before_config_save(self):
        import json
        import tempfile
        from unittest.mock import patch

        model = _patched_qwen2_seq_cls()
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Custom-code export runs after Transformers resets architectures, before config saving.
            # Newer Transformers checks _auto_class through a classmethod, not the instance.
            with patch.object(type(model), '_auto_class', 'AutoModel'):
                with patch('transformers.modeling_utils.custom_object_save', side_effect=OSError('copy failed')):
                    with self.assertRaisesRegex(OSError, 'copy failed'):
                        model.save_pretrained(tmp_dir)
            self.assertEqual(model.config.architectures, ['Qwen2ForSequenceClassification'])
            self.assertNotIn('save_pretrained', model.config.__dict__)
            model.save_pretrained(tmp_dir)
            with open(os.path.join(tmp_dir, 'config.json')) as f:
                self.assertEqual(json.load(f)['architectures'], ['Qwen2ForSequenceClassification'])

    def test_seq_cls_architectures_existing_save_wrapper(self):
        import json
        import tempfile
        from functools import partial

        from swift.model.patcher import _patch_save_pretrained_architectures

        model = _patched_qwen2_seq_cls()
        model.save_pretrained = partial(type(model).save_pretrained, model, max_shard_size='1KB')
        _patch_save_pretrained_architectures(model)
        _patch_save_pretrained_architectures(model)
        with tempfile.TemporaryDirectory() as tmp_dir:
            model.save_pretrained(tmp_dir)
            self.assertTrue(os.path.isfile(os.path.join(tmp_dir, 'model.safetensors.index.json')))
            with open(os.path.join(tmp_dir, 'config.json')) as f:
                self.assertEqual(json.load(f)['architectures'], ['Qwen2ForSequenceClassification'])

    def test_seq_cls_architectures_none_keeps_class_name(self):
        """A missing ``architectures`` must not be written back as ``null``.

        The pre-save snapshot only restores a value that existed before the
        save, so transformers' own ``[class_name]`` is kept when there is
        nothing to restore.
        """
        import json
        import tempfile

        model = _patched_qwen2_seq_cls(architectures=None)
        self.assertIsNone(model.config.architectures)

        with tempfile.TemporaryDirectory() as tmp_dir:
            model.save_pretrained(tmp_dir)
            with open(os.path.join(tmp_dir, 'config.json'), 'r') as f:
                saved_config = json.load(f)

        self.assertEqual(saved_config['architectures'], ['Qwen2ForCausalLM'])

    def test_seq_cls_architectures_post_patch_assignment_kept(self):
        """An ``architectures`` value assigned after patching is preserved."""
        import json
        import tempfile

        model = _patched_qwen2_seq_cls()
        model.config.architectures = ['MySeqClsModel']

        with tempfile.TemporaryDirectory() as tmp_dir:
            model.save_pretrained(tmp_dir)
            with open(os.path.join(tmp_dir, 'config.json'), 'r') as f:
                saved_config = json.load(f)

        self.assertEqual(saved_config['architectures'], ['MySeqClsModel'])

    def test_seq_cls_architectures_unpatched_instance_unaffected(self):
        """Patching one model does not affect other instances of its class."""
        import json
        import tempfile
        from transformers import Qwen2Config, Qwen2ForCausalLM

        _patched_qwen2_seq_cls()

        config = Qwen2Config(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=2)
        plain = Qwen2ForCausalLM(config)

        with tempfile.TemporaryDirectory() as tmp_dir:
            plain.save_pretrained(tmp_dir)
            with open(os.path.join(tmp_dir, 'config.json'), 'r') as f:
                saved_config = json.load(f)

        self.assertEqual(saved_config['architectures'], ['Qwen2ForCausalLM'])


if __name__ == '__main__':
    test_qwen2()
    # test_modelscope_hub()
