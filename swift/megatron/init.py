import os
import shutil
import sys

from swift.llm import git_clone_github
from swift.utils import is_megatron_available, safe_ddp_context, subprocess_run


def _rename_files():
    megatron_patch_path = os.environ['PAI_MEGATRON_PATCH_PATH']
    qwen_folders = ['toolkits/model_checkpoints_convertor/qwen']
    for folder in qwen_folders:
        dir_path = os.path.join(megatron_patch_path, folder)
        for fname in os.listdir(dir_path):
            old_path = os.path.join(dir_path, fname)
            fname = fname.replace('qwen1.', 'qwen1_')
            fname = fname.replace('qwen2.', 'qwen2_')
            new_path = os.path.join(dir_path, fname)
            if old_path != new_path and os.path.exists(old_path):
                shutil.move(old_path, new_path)


def init_megatron_env() -> None:
    if 'MEGATRON_LM_PATH' not in os.environ:
        os.environ['MEGATRON_LM_PATH'] = git_clone_github(
            'https://github.com/NVIDIA/Megatron-LM', branch='core_r0.11.0')
    if not is_megatron_available():
        subprocess_run([sys.executable, '-m', 'pip', 'install', '-e', os.environ['MEGATRON_LM_PATH']])
    sys.path.append(os.environ['MEGATRON_LM_PATH'])

    if 'PAI_MEGATRON_PATCH_PATH' not in os.environ:
        os.environ['PAI_MEGATRON_PATCH_PATH'] = git_clone_github(
            'https://github.com/alibaba/Pai-Megatron-Patch', commit_hash='v0.10.3')
    sys.path.append(os.environ['PAI_MEGATRON_PATCH_PATH'])

    # rename qwen1.5/2.5->qwen1_5/2_5 files
    with safe_ddp_context('rename_files'):
        _rename_files()
