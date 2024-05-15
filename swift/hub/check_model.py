# Copyright (c) Alibaba, Inc. and its affiliates.

import os
from typing import Dict, Optional, Union
from urllib.parse import urlparse

from swift.utils.logger import get_logger
from .api import HubApi, ModelScopeConfig
from .constants import FILE_HASH
from .git import GitCommandWrapper
from .utils.caching import ModelFileSystemCache
from .utils.utils import compute_hash

logger = get_logger()


def check_local_model_is_latest(
    model_root_path: str,
    user_agent: Optional[Union[Dict, str]] = None,
):
    """Check local model repo is latest.
    Check local model repo is same as hub latest version.
    """
    try:
        model_cache = None
        # download with git
        if os.path.exists(os.path.join(model_root_path, '.git')):
            git_cmd_wrapper = GitCommandWrapper()
            git_url = git_cmd_wrapper.get_repo_remote_url(model_root_path)
            if git_url.endswith('.git'):
                git_url = git_url[:-4]
            u_parse = urlparse(git_url)
            model_id = u_parse.path[1:]
        else:  # snapshot_download
            model_cache = ModelFileSystemCache(model_root_path)
            model_id = model_cache.get_model_id()

        # make headers
        headers = {'user-agent': ModelScopeConfig.get_user_agent(user_agent=user_agent, )}
        cookies = ModelScopeConfig.get_cookies()

        snapshot_header = headers if 'CI_TEST' in os.environ else {**headers, **{'Snapshot': 'True'}}
        _api = HubApi()
        try:
            _, revisions = _api.get_model_branches_and_tags(model_id=model_id, use_cookies=cookies)
            if len(revisions) > 0:
                latest_revision = revisions[0]
            else:
                latest_revision = 'master'
        except:  # noqa: E722
            latest_revision = 'master'

        model_files = _api.get_model_files(
            model_id=model_id,
            revision=latest_revision,
            recursive=True,
            headers=snapshot_header,
            use_cookies=cookies,
        )
        for model_file in model_files:
            if model_file['Type'] == 'tree':
                continue
            # check model_file updated
            if model_cache is not None:
                if model_cache.exists(model_file):
                    continue
                else:
                    logger.info(
                        f'Model file {model_file["Name"]} is different from the latest version `{latest_revision}`,'
                        f'This is because you are using an older version or the file is updated manually.')
                    break
            else:
                if FILE_HASH in model_file:
                    local_file_hash = compute_hash(os.path.join(model_root_path, model_file['Path']))
                    if local_file_hash == model_file[FILE_HASH]:
                        continue
                    else:
                        logger.info(
                            f'Model file {model_file["Name"]} is different from the latest version `{latest_revision}`,'
                            f'This is because you are using an older version or the file is updated manually.')
                        break
    except:  # noqa: E722
        pass  # ignore


def check_model_is_id(model_id: str, token=None):
    if token is None:
        token = os.environ.get('MODELSCOPE_API_TOKEN')
    if model_id is None or os.path.exists(model_id):
        return False
    else:
        _api = HubApi()
        if token is not None:
            _api.login(token)
        try:
            _api.get_model(model_id=model_id, )
            return True
        except Exception:
            return False
