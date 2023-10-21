import tempfile
import unittest

from modelscope import AutoTokenizer, snapshot_download

from swift.llm import MODEL_MAPPING, ModelType, get_model_tokenizer


class TestModel(unittest.TestCase):

    def test_model(self):
        model, tokenizer = get_model_tokenizer(ModelType.qwen_7b_chat_int4)
        assert model.__class__.__name__ == 'QWenLMHeadModel'
        assert tokenizer.__class__.__name__ == 'QWenTokenizer'


if __name__ == '__main__':
    unittest.main()
