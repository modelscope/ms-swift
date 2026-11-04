import json
import os
import tempfile
import unittest
from transformers import AutoConfig, LlamaConfig, PretrainedConfig
from transformers.models.auto.configuration_auto import CONFIG_MAPPING

from swift.utils import HfConfigFactory


class OverriddenLlamaConfig(PretrainedConfig):
    """Mimics a third-party config class that only keeps the keys in config.json."""

    model_type = 'llama'

    def __init__(self, hidden_size=64, **kwargs):
        self.hidden_size = hidden_size
        super().__init__(**kwargs)


class TestLoadHfConfig(unittest.TestCase):

    def setUp(self):
        self._tmp_dir = tempfile.TemporaryDirectory()
        self.model_dir = self._tmp_dir.name
        self._write_config({'model_type': 'llama', 'hidden_size': 128, 'num_hidden_layers': 2})
        self._origin = CONFIG_MAPPING._extra_content.get('llama')
        CONFIG_MAPPING.register('llama', OverriddenLlamaConfig, exist_ok=True)

    def tearDown(self):
        if self._origin is None:
            CONFIG_MAPPING._extra_content.pop('llama', None)
        else:
            CONFIG_MAPPING._extra_content['llama'] = self._origin
        self._tmp_dir.cleanup()

    def _write_config(self, config):
        with open(os.path.join(self.model_dir, 'config.json'), 'w') as f:
            json.dump(config, f)

    def test_load_hf_config_bypasses_auto_config_override(self):
        self.assertIs(type(AutoConfig.from_pretrained(self.model_dir)), OverriddenLlamaConfig)
        config = HfConfigFactory.load_hf_config(self.model_dir)
        self.assertIs(type(config), LlamaConfig)
        self.assertEqual(config.hidden_size, 128)
        self.assertEqual(config.num_hidden_layers, 2)

    def test_load_hf_config_without_override(self):
        CONFIG_MAPPING._extra_content.pop('llama', None)
        config = HfConfigFactory.load_hf_config(self.model_dir)
        self.assertIs(type(config), LlamaConfig)

    def test_load_hf_config_falls_back_for_remote_code(self):
        self._write_config({'model_type': 'llama', 'auto_map': {'AutoConfig': 'configuration_x.X'}})
        self.assertIsNone(HfConfigFactory.load_hf_config(self.model_dir))

    def test_load_hf_config_falls_back_for_unknown_model_type(self):
        self._write_config({'model_type': 'swift_unregistered_model_type'})
        self.assertIsNone(HfConfigFactory.load_hf_config(self.model_dir))


if __name__ == '__main__':
    unittest.main()
