# Copyright (c) Alibaba, Inc. and its affiliates.

import os
import time
from typing import Optional

from swift.utils.logger import get_logger
from .constants import DEFAULT_DATASET_REVISION, DEFAULT_REPOSITORY_REVISION, MASTER_MODEL_BRANCH
from .errors import GitError, InvalidParameter, NotLoginException
from .git import GitCommandWrapper
from .utils.utils import get_endpoint

logger = get_logger()


class Repository:
    """A local representation of the model git repository.
    """

    def __init__(self,
                 model_dir: str,
                 clone_from: str,
                 revision: Optional[str] = DEFAULT_REPOSITORY_REVISION,
                 auth_token: Optional[str] = None,
                 git_path: Optional[str] = None):
        """Instantiate a Repository object by cloning the remote ModelScopeHub repo

        Args:
            model_dir (str): The model root directory.
            clone_from (str): model id in ModelScope-hub from which git clone
            revision (str, optional): revision of the model you want to clone from.
                     Can be any of a branch, tag or commit hash
            auth_token (str, optional): token obtained when calling `HubApi.login()`.
                        Usually you can safely ignore the parameter as the token is already
                        saved when you login the first time, if None, we will use saved token.
            git_path (str, optional): The git command line path, if None, we use 'git'

        Raises:
            InvalidParameter: revision is None.
        """
        self.model_dir = model_dir
        self.model_base_dir = os.path.dirname(model_dir)
        if self.model_base_dir == '':
            self.model_base_dir = '.'
        self.model_repo_name = os.path.basename(model_dir)

        if not revision:
            err_msg = 'a non-default value of revision cannot be empty.'
            raise InvalidParameter(err_msg)

        from .api import ModelScopeConfig
        if auth_token:
            self.auth_token = auth_token
        else:
            self.auth_token = ModelScopeConfig.get_token()

        git_wrapper = GitCommandWrapper()
        if not git_wrapper.is_lfs_installed():
            logger.error('git lfs is not installed, please install.')

        self.git_wrapper = GitCommandWrapper(git_path)
        os.makedirs(self.model_dir, exist_ok=True)
        url = self._get_model_id_url(clone_from)
        if os.listdir(self.model_dir):  # directory not empty.
            remote_url = self._get_remote_url()
            remote_url = self.git_wrapper.remove_token_from_url(remote_url)
            if remote_url and remote_url == url:  # need not clone again
                return
        self.git_wrapper.clone(self.model_base_dir, self.auth_token, url, self.model_repo_name, revision)

        if git_wrapper.is_lfs_installed():
            git_wrapper.git_lfs_install(self.model_dir)  # init repo lfs

        # add user info if login
        self.git_wrapper.add_user_info(self.model_base_dir, self.model_repo_name)
        if self.auth_token:  # config remote with auth token
            self.git_wrapper.config_auth_token(self.model_dir, self.auth_token)

    def _get_model_id_url(self, model_id):
        url = f'{get_endpoint()}/{model_id}.git'
        return url

    def _get_remote_url(self):
        try:
            remote = self.git_wrapper.get_repo_remote_url(self.model_dir)
        except GitError:
            remote = None
        return remote

    def pull(self, remote: str = 'origin', branch: str = 'master'):
        """Pull remote branch

        Args:
            remote (str, optional): The remote name. Defaults to 'origin'.
            branch (str, optional): The remote branch. Defaults to 'master'.
        """
        self.git_wrapper.pull(self.model_dir, remote=remote, branch=branch)

    def add_lfs_type(self, file_name_suffix: str):
        """Add file suffix to lfs list.

        Args:
            file_name_suffix (str): The file name suffix.
                examples '*.safetensors'
        """
        os.system("printf '%s filter=lfs diff=lfs merge=lfs -text\n'>>%s" %
                  (file_name_suffix, os.path.join(self.model_dir, '.gitattributes')))

    def push(self,
             commit_message: str,
             local_branch: Optional[str] = DEFAULT_REPOSITORY_REVISION,
             remote_branch: Optional[str] = DEFAULT_REPOSITORY_REVISION,
             force: Optional[bool] = False):
        """Push local files to remote, this method will do.
        Execute git pull, git add, git commit, git push in order.

        Args:
            commit_message (str): commit message
            local_branch(str, optional): The local branch, default master.
            remote_branch (str, optional): The remote branch to push, default master.
            force (bool, optional): whether to use forced-push.

        Raises:
            InvalidParameter: no commit message.
            NotLoginException: no auth token.
        """
        if commit_message is None or not isinstance(commit_message, str):
            msg = 'commit_message must be provided!'
            raise InvalidParameter(msg)
        if not isinstance(force, bool):
            raise InvalidParameter('force must be bool')

        if not self.auth_token:
            raise NotLoginException('Must login to push, please login first.')

        self.git_wrapper.config_auth_token(self.model_dir, self.auth_token)
        self.git_wrapper.add_user_info(self.model_base_dir, self.model_repo_name)

        url = self.git_wrapper.get_repo_remote_url(self.model_dir)
        assert 'modelscope' in url  # Avoid unexpected pushes
        self.git_wrapper.add(self.model_dir, all_files=True)
        # avoid race condition with git status
        time.sleep(0.5)
        if self.is_repo_clean():
            logger.info('Repo currently clean. Ignoring commit and push_to_hub')
        else:
            self.git_wrapper.commit(self.model_dir, commit_message)
            self.git_wrapper.push(
                repo_dir=self.model_dir,
                token=self.auth_token,
                url=url,
                local_branch=local_branch,
                remote_branch=remote_branch)

    def tag(self, tag_name: str, message: str, ref: Optional[str] = MASTER_MODEL_BRANCH):
        """Create a new tag.

        Args:
            tag_name (str): The name of the tag
            message (str): The tag message.
            ref (str, optional): The tag reference, can be commit id or branch.

        Raises:
            InvalidParameter: no commit message.
        """
        if tag_name is None or tag_name == '':
            msg = 'We use tag-based revision, therefore tag_name cannot be None or empty.'
            raise InvalidParameter(msg)
        if message is None or message == '':
            msg = 'We use annotated tag, therefore message cannot None or empty.'
            raise InvalidParameter(msg)
        self.git_wrapper.tag(repo_dir=self.model_dir, tag_name=tag_name, message=message, ref=ref)

    def tag_and_push(self, tag_name: str, message: str, ref: Optional[str] = MASTER_MODEL_BRANCH):
        """Create tag and push to remote

        Args:
            tag_name (str): The name of the tag
            message (str): The tag message.
            ref (str, optional): The tag ref, can be commit id or branch. Defaults to MASTER_MODEL_BRANCH.
        """
        self.tag(tag_name, message, ref)

        self.git_wrapper.push_tag(repo_dir=self.model_dir, tag_name=tag_name)

    def is_repo_clean(self) -> bool:
        response = self.git_wrapper._run_git_command('-C', self.model_dir, 'status', '--porcelain')
        return len(response.stdout.strip()) == 0


class DatasetRepository:
    """A local representation of the dataset (metadata) git repository.
    """

    def __init__(self,
                 repo_work_dir: str,
                 dataset_id: str,
                 revision: Optional[str] = DEFAULT_DATASET_REVISION,
                 auth_token: Optional[str] = None,
                 git_path: Optional[str] = None):
        """
        Instantiate a Dataset Repository object by cloning the remote ModelScope dataset repo

        Args:
            repo_work_dir (str): The dataset repo root directory.
            dataset_id (str): dataset id in ModelScope from which git clone
            revision (str, optional): revision of the dataset you want to clone from.
                                      Can be any of a branch, tag or commit hash
            auth_token (str, optional): token obtained when calling `HubApi.login()`.
                                        Usually you can safely ignore the parameter as the token is
                                        already saved when you login the first time, if None, we will use saved token.
            git_path (str, optional): The git command line path, if None, we use 'git'

        Raises:
            InvalidParameter: parameter invalid.
        """
        self.dataset_id = dataset_id
        if not repo_work_dir or not isinstance(repo_work_dir, str):
            err_msg = 'dataset_work_dir must be provided!'
            raise InvalidParameter(err_msg)
        self.repo_work_dir = repo_work_dir.rstrip('/')
        if not self.repo_work_dir:
            err_msg = 'dataset_work_dir can not be root dir!'
            raise InvalidParameter(err_msg)
        self.repo_base_dir = os.path.dirname(self.repo_work_dir)
        self.repo_name = os.path.basename(self.repo_work_dir)

        if not revision:
            err_msg = 'a non-default value of revision cannot be empty.'
            raise InvalidParameter(err_msg)
        self.revision = revision
        from .api import ModelScopeConfig
        if auth_token:
            self.auth_token = auth_token
        else:
            self.auth_token = ModelScopeConfig.get_token()

        self.git_wrapper = GitCommandWrapper(git_path)
        os.makedirs(self.repo_work_dir, exist_ok=True)
        self.repo_url = self._get_repo_url(dataset_id=dataset_id)

    def clone(self) -> str:
        # check local repo dir, directory not empty.
        if os.listdir(self.repo_work_dir):
            remote_url = self._get_remote_url()
            remote_url = self.git_wrapper.remove_token_from_url(remote_url)
            # no need clone again
            if remote_url and remote_url == self.repo_url:
                return ''

        logger.info('Cloning repo from {} '.format(self.repo_url))
        self.git_wrapper.clone(self.repo_base_dir, self.auth_token, self.repo_url, self.repo_name, self.revision)
        return self.repo_work_dir

    def push(self,
             commit_message: str,
             branch: Optional[str] = DEFAULT_DATASET_REVISION,
             force: Optional[bool] = False):
        """Push local files to remote, this method will do.
           git pull
           git add
           git commit
           git push

        Args:
            commit_message (str): commit message
            branch (str, optional): which branch to push.
            force (bool, optional): whether to use forced-push.

        Raises:
            InvalidParameter: no commit message.
            NotLoginException: no access token.
        """
        if commit_message is None or not isinstance(commit_message, str):
            msg = 'commit_message must be provided!'
            raise InvalidParameter(msg)

        if not isinstance(force, bool):
            raise InvalidParameter('force must be bool')

        if not self.auth_token:
            raise NotLoginException('Must login to push, please login first.')

        self.git_wrapper.config_auth_token(self.repo_work_dir, self.auth_token)
        self.git_wrapper.add_user_info(self.repo_base_dir, self.repo_name)

        remote_url = self._get_remote_url()
        remote_url = self.git_wrapper.remove_token_from_url(remote_url)

        self.git_wrapper.pull(self.repo_work_dir)
        self.git_wrapper.add(self.repo_work_dir, all_files=True)
        self.git_wrapper.commit(self.repo_work_dir, commit_message)
        self.git_wrapper.push(
            repo_dir=self.repo_work_dir,
            token=self.auth_token,
            url=remote_url,
            local_branch=branch,
            remote_branch=branch)

    def _get_repo_url(self, dataset_id):
        return f'{get_endpoint()}/datasets/{dataset_id}.git'

    def _get_remote_url(self):
        try:
            remote = self.git_wrapper.get_repo_remote_url(self.repo_work_dir)
        except GitError:
            remote = None
        return remote
