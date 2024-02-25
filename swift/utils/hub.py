import os
import shutil
import time
from typing import Optional

from requests.exceptions import HTTPError

from swift.hub import HubApi, ModelScopeConfig
from swift.hub.constants import ModelVisibility
from swift.utils import create_ms_repo
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


def _init_ms_repo(local_dir: str,
                  hub_model_id: str,
                  hub_token: Optional[str] = None,
                  hub_private_repo: bool = False) -> None:
    create_ms_repo(hub_model_id, hub_token, hub_private_repo)
    git_token = ModelScopeConfig.get_token()
    ms_url = f'https://oauth2:{git_token}@www.modelscope.cn/{hub_model_id}.git'
    subprocess_run(['git', '-C', local_dir, 'clone', ms_url, 'ms'],
                   env={'GIT_LFS_SKIP_SMUDGE': '1'})
    subprocess_run(['git', '-C', local_dir, 'lfs', 'pull'])


def push_to_ms_hub(ckpt_dir: str,
                   hub_model_id: str,
                   hub_token: Optional[str] = None,
                   hub_private_repo: bool = False):
    assert isinstance(hub_model_id, str)
    subprocess_run(['git', 'lfs', 'env'])  # check git-lfs install
    tmp_dir = os.path.join(ckpt_dir, 'tmp')
    _init_ms_repo(tmp_dir, hub_model_id, hub_token, hub_private_repo)
    # mv .git
    shutil.copytree(
        os.path.join(tmp_dir, '.git'), os.path.join(ckpt_dir, '.git'))
    shutil.rmtree(tmp_dir)
    # add
    subprocess_run(['git', '-C', ckpt_dir, 'lfs', 'install'])
    time.sleep(0.5)
    subprocess_run(['git', '-C', ckpt_dir, 'add', '.'])
    subprocess_run(f'git -C {ckpt_dir} lfs status')
    subprocess_run(f'git -C {ckpt_dir} commit -m "first commit"')
    subprocess_run(f'git -C {ckpt_dir} push')
    url = f'https://www.modelscope.cn/models/{hub_model_id}/summary'
    logger.info(f'Push to Modelscope successful. url: {url}.')
