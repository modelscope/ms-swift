def test_model_arch():
    from swift.llm import MODEL_MAPPING, safe_snapshot_download
    from transformers import PretrainedConfig
    from swift.utils import JsonlWriter
    import random
    jsonl_writer = JsonlWriter('model_arch.jsonl')
    for i, (model_type, model_meta) in enumerate(MODEL_MAPPING.items()):
        if i < 0:
            continue
        arch_list = model_meta.architectures
        for model_group in model_meta.model_groups:
            model = random.choice(model_group.models).ms_model_id
            config_dict = None
            try:
                model_dir = safe_snapshot_download(model, download_model=False)
                config_dict = PretrainedConfig.get_config_dict(model_dir)[0]
            except Exception:
                pass
            finally:
                msg = None
                if config_dict:
                    arch = config_dict.get('architectures')
                    if arch and arch[0] not in arch_list:
                        msg = {
                            'model_type': model_type,
                            'model': model,
                            'config_arch': arch,
                            'architectures': arch_list
                        }
                    elif not arch and arch_list:
                        msg = {
                            'model_type': model_type,
                            'model': model,
                            'config_arch': arch,
                            'architectures': arch_list
                        }
                else:
                    msg = {'msg': 'error', 'model_type': model_type, 'model': model, 'arch_list': arch_list}
                if msg:
                    jsonl_writer.append(msg)


if __name__ == '__main__':
    test_model_arch()
