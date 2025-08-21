from evalscope import TaskConfig, run_task

task_cfg_dict = TaskConfig(
    work_dir='outputs',
    eval_backend='VLMEvalKit',
    eval_config={
        'data': ['MMMU_DEV_VAL'],
        'mode': 'all',
        'model': [
            {'api_base': 'http://localhost:8000/v1/chat/completions',
            'key': 'EMPTY',
            'name': 'CustomAPIModel',
            'temperature': 0.6,
            'type': 'Qwen3-VL',
            'img_size': -1,
            'video_llm': False,
            'max_tokens': 512,}
            ],
        'reuse': False,
        'nproc': 64,
        'judge': 'exact_matching'},
)

run_task(task_cfg=task_cfg_dict)
