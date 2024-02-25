import os
import shutil
import time
from typing import Optional

from requests.exceptions import HTTPError

from swift.hub import HubApi, ModelScopeConfig
from swift.hub.constants import ModelVisibility
from .logger import get_logger
from .utils import subprocess_run

logger = get_logger()


def create_ms_repo(hub_model_id: str,
                   hub_token: Optional[str] = None,
                   hub_private_repo: bool = False) -> str:
    assert hub_model_id is not None, 'Please enter a valid hub_model_id'

    api = HubApi()
    if hub_token is None:
        hub_token = os.environ.get('MODELSCOPE_API_TOKEN')
    if hub_token is not None:
        api.login(hub_token)
    visibility = ModelVisibility.PRIVATE if hub_private_repo else ModelVisibility.PUBLIC

    if '/' not in hub_model_id:
        user_name = ModelScopeConfig.get_user_info()[0]
        assert isinstance(user_name, str)
        hub_model_id = f'{user_name}/{hub_model_id}'
        logger.info(
            f"'/' not in hub_model_id, setting hub_model_id: {hub_model_id}")
    try:
        api.create_model(hub_model_id, visibility)
    except HTTPError:
        # The remote repository has been created
        pass
    return hub_model_id


def push_to_ms_hub(ckpt_dir: str,
                   hub_model_id: str,
                   hub_token: Optional[str] = None,
                   hub_private_repo: bool = False,
                   commit_message: str = 'Push `ckpt_dir` to ModelScope Hub'):
    logger.info(f'Starting push to hub. ckpt_dir: {ckpt_dir}.')
    subprocess_run(['git', 'lfs', 'env'])  # check git-lfs install

    hub_model_id = create_ms_repo(hub_model_id, hub_token, hub_private_repo)
    git_token = ModelScopeConfig.get_token()
    ms_url = f'https://oauth2:{git_token}@www.modelscope.cn/{hub_model_id}.git'
    subprocess_run(['git', '-C', ckpt_dir, 'clone', ms_url, 'tmp'],
                   env={'GIT_LFS_SKIP_SMUDGE': '1'})
    tmp_dir = os.path.join(ckpt_dir, 'tmp')
    subprocess_run(['git', '-C', tmp_dir, 'lfs', 'pull'])
    # mv .git
    dst_git_path = os.path.join(ckpt_dir, '.git')
    if os.path.exists(dst_git_path):
        shutil.rmtree(dst_git_path)
    shutil.copytree(os.path.join(tmp_dir, '.git'), dst_git_path)
    shutil.copy(
        os.path.join(tmp_dir, '.gitattributes'),
        os.path.join(ckpt_dir, '.gitattributes'))
    shutil.rmtree(tmp_dir)
    # add commit push
    subprocess_run(['git', '-C', ckpt_dir, 'lfs', 'install'])
    time.sleep(0.5)
    subprocess_run(['git', '-C', ckpt_dir, 'add', '.'])
    subprocess_run(['git', '-C', ckpt_dir, 'lfs', 'status'])
    subprocess_run(['git', '-C', ckpt_dir, 'commit', '-m', commit_message])
    subprocess_run(['git', '-C', ckpt_dir, 'push'])
    url = f'https://www.modelscope.cn/models/{hub_model_id}/summary'
    logger.info(f'Push to Modelscope successful. url: `{url}`.')
