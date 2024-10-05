# [TODO:not impl]
# def load_by_unsloth(model_dir, torch_dtype, **kwargs):
#     """Load model by unsloth"""
#     assert is_unsloth_available(), 'please install unsloth if using `use_unsloth=True`'
#     from unsloth import FastLanguageModel
#     return FastLanguageModel.from_pretrained(
#         model_name=model_dir,
#         max_seq_length=kwargs.get('max_length', None),
#         dtype=torch_dtype,
#         load_in_4bit=kwargs.get('load_in_4bit', True),
#         trust_remote_code=True,
#     )
#

# def load_by_transformers(automodel_class, model_dir, model_config, torch_dtype, is_aqlm, is_training, model_kwargs,
#                          **kwargs):
#     """Load model by transformers"""
#     context = kwargs.get('context', None)
#     if is_aqlm and is_training:
#         require_version('transformers>=4.39')
#         import aqlm
#         context = aqlm.optimize_for_training()
#     if context is None:
#         context = nullcontext()
#     with context:
#         model = automodel_class.from_pretrained(
#             model_dir, config=model_config, torch_dtype=torch_dtype, trust_remote_code=True, **model_kwargs)
#     return model
