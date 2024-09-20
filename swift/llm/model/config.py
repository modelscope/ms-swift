import os.path

from modelscope import AutoConfig

from swift.llm.model.loader import safe_snapshot_download


class ConfigReader:

    @staticmethod
    def read_config(key, model_type, model_id_or_path, revision):
        model_dir = safe_snapshot_download(model_type, model_id_or_path, revision, download_model=False)
        if os.path.exists(os.path.join(model_dir, 'config.json')):
            return ConfigReader.read_config_from_hf(key, model_dir)
        else:
            # For Mistral
            raise NotImplementedError

    @staticmethod
    def read_config_from_hf(key, model_dir):
        config = AutoConfig.from_pretrained(model_dir)
        for k in key.split('.'):
            config = getattr(config, k, None)
            if config is None:
                return None
        return config
