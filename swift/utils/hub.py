import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Optional, Union, List

from requests.exceptions import HTTPError

from swift.hub import HubApi, ModelScopeConfig
from swift.hub.constants import ModelVisibility
from .logger import get_logger
from .utils import subprocess_run

logger = get_logger()


def create_ms_repo(
        repo_id: str,
        *,
        token: Optional[str] = None,
        private: bool = False,
        **kwargs) -> str:
    assert repo_id is not None, 'Please enter a valid hub_model_id'

    api = HubApi()
    if token is None:
        hub_token = os.environ.get('MODELSCOPE_API_TOKEN')
    if token is not None:
        api.login(token)
    visibility = ModelVisibility.PRIVATE if private else ModelVisibility.PUBLIC

    if '/' not in repo_id:
        user_name = ModelScopeConfig.get_user_info()[0]
        assert isinstance(user_name, str)
        repo_id = f'{user_name}/{repo_id}'
        logger.info(f"'/' not in hub_model_id, setting hub_model_id: {repo_id}")
    try:
        api.create_model(repo_id, visibility)
    except HTTPError:
        # The remote repository has been created
        pass
    return repo_id


def push_to_ms_hub(self,
                    *,
                    repo_id: str,
                    folder_path: Union[str, Path],
                    path_in_repo: Optional[str] = None,
                    commit_message: Optional[str] = None,
                    token: Union[str, bool, None] = None):
    logger.info(f'Starting push to hub. ckpt_dir: {folder_path}.')
    tmp_file_name = tempfile.TemporaryDirectory().name
    subprocess_run(['git', 'lfs', 'env'], stdout=subprocess.PIPE)  # check git-lfs install

    path_in_repo = path_in_repo or ''
    if not folder_path.endswith(path_in_repo):
        folder_path = os.path.join(folder_path, path_in_repo)

    git_token = ModelScopeConfig.get_token()
    ms_url = f'https://oauth2:{git_token}@www.modelscope.cn/{repo_id}.git'
    subprocess_run(['git', '-C', folder_path, 'clone', ms_url, tmp_file_name], env={'GIT_LFS_SKIP_SMUDGE': '1'})
    tmp_dir = os.path.join(folder_path, tmp_file_name)
    subprocess_run(['git', '-C', tmp_dir, 'lfs', 'pull'])
    logger.info('Git clone the repo successfully.')
    # mv .git
    dst_git_path = os.path.join(folder_path, '.git')
    if os.path.exists(dst_git_path):
        shutil.rmtree(dst_git_path)
    shutil.copytree(os.path.join(tmp_dir, '.git'), dst_git_path)
    shutil.copy(os.path.join(tmp_dir, '.gitattributes'), os.path.join(folder_path, '.gitattributes'))
    shutil.rmtree(tmp_dir)
    # add commit push
    subprocess_run(['git', '-C', folder_path, 'lfs', 'install'])
    time.sleep(0.5)
    logger.info('Start `git add .`')
    subprocess_run(['git', '-C', folder_path, 'add', '.'])
    if is_repo_clean(folder_path):
        logger.info('Repo currently clean. Ignoring commit and push_to_hub')
    else:
        subprocess_run(['git', '-C', folder_path, 'commit', '-m', commit_message])
        subprocess_run(['git', '-C', folder_path, 'push'])
        url = f'https://www.modelscope.cn/models/{repo_id}/summary'
        logger.info(f'Push to Modelscope successful. url: `{url}`.')


def is_repo_clean(ckpt_dir: str) -> bool:
    resp = subprocess_run(['git', '-C', ckpt_dir, 'status', '--porcelain'], stdout=subprocess.PIPE)
    return len(resp.stdout.strip()) == 0
