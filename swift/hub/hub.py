# Copyright (c) Alibaba, Inc. and its affiliates.
import os
import tempfile
from contextlib import contextmanager
from functools import partial
from pathlib import Path
from typing import List, Literal, Optional, Union

import huggingface_hub
from huggingface_hub import RepoUrl
from huggingface_hub.hf_api import api, future_compatible
from requests.exceptions import HTTPError
from transformers import trainer
from transformers.utils import logging, strtobool

from swift.utils.env import use_hf_hub

logger = logging.get_logger(__name__)


class HubOperation:

    @classmethod
    @contextmanager
    def patch_hub(cls):
        yield

    @classmethod
    def try_login(cls, token: Optional[str] = None) -> bool:
        """Try to login to the hub

        Args:
            token: The hub token to use

        Returns:
            bool: Whether login is successful
        """
        raise NotImplementedError

    @classmethod
    def create_model_repo(cls, repo_id: str, token: Optional[str] = None, private: bool = False):
        """Create a model repo on the hub

        Args:
            repo_id: The model id of the hub
            token: The hub token to use
            private: If is a private repo
        """
        raise NotImplementedError

    @classmethod
    def push_to_hub(cls,
                    repo_id: str,
                    folder_path: Union[str, Path],
                    path_in_repo: Optional[str] = None,
                    commit_message: Optional[str] = None,
                    commit_description: Optional[str] = None,
                    token: Union[str, bool, None] = None,
                    private: bool = False,
                    revision: Optional[str] = 'master',
                    ignore_patterns: Optional[Union[List[str], str]] = None,
                    **kwargs):
        """Push a model-like folder to the hub

        Args:
            repo_id: The repo id
            folder_path: The local folder path
            path_in_repo: Which remote folder to put the local files in
            commit_message: The commit message of git
            commit_description: The commit description
            token: The hub token
            private: Private hub or not
            revision: The revision to push to
            ignore_patterns: The ignore file patterns
        """
        raise NotImplementedError

    @classmethod
    def load_dataset(cls,
                     dataset_id: str,
                     subset_name: str,
                     split: str,
                     streaming: bool = False,
                     revision: Optional[str] = None):
        """Load a dataset from the repo

        Args:
            dataset_id: The dataset id
            subset_name: The subset name of the dataset
            split: The split info
            streaming: Streaming mode
            revision: The revision of the dataset

        Returns:
            The Dataset instance
        """
        raise NotImplementedError

    @classmethod
    def download_model(cls,
                       model_id_or_path: Optional[str] = None,
                       revision: Optional[str] = None,
                       download_model: bool = True,
                       ignore_patterns: Optional[List[str]] = None,
                       **kwargs):
        """Download model from the hub

        Args:
            model_id_or_path: The model id
            revision: The model revision
            download_model: Whether downloading bin/safetensors files, this is usually useful when only
                using tokenizer
            ignore_patterns: Custom ignore pattern
            **kwargs:

        Returns:
            The local dir
        """
        raise NotImplementedError


class MSHub(HubOperation):
    ms_token = None

    @staticmethod
    def create_repo(repo_id: str, *, token: Union[str, bool, None] = None, private: bool = False, **kwargs) -> RepoUrl:
        """
        Create a new repository on the hub.

        Args:
            repo_id: The ID of the repository to create.
            token: The authentication token to use.
            private: Whether the repository should be private.
            **kwargs: Additional arguments.

        Returns:
            RepoUrl: The URL of the created repository.
        """
        hub_model_id = MSHub.create_model_repo(repo_id, token, private)
        return RepoUrl(url=hub_model_id, )

    @staticmethod
    @future_compatible
    def upload_folder(
        self,
        *,
        repo_id: str,
        folder_path: Union[str, Path],
        path_in_repo: Optional[str] = None,
        commit_message: Optional[str] = None,
        commit_description: Optional[str] = None,
        token: Union[str, bool, None] = None,
        revision: Optional[str] = 'master',
        ignore_patterns: Optional[Union[List[str], str]] = None,
        **kwargs,
    ):
        from modelscope.utils.repo_utils import CommitInfo
        MSHub.push_to_hub(repo_id, folder_path, path_in_repo, commit_message, commit_description, token, True, revision,
                          ignore_patterns)
        return CommitInfo(
            commit_url=f'https://www.modelscope.cn/models/{repo_id}/files',
            commit_message=commit_message,
            commit_description=commit_description,
            oid=None,
        )

    @classmethod
    @contextmanager
    def patch_hub(cls):
        hub_create_repo = huggingface_hub.create_repo
        hub_upload_folder = huggingface_hub.upload_folder
        trainer_create_repo = trainer.create_repo
        trainer_upload_folder = trainer.upload_folder

        huggingface_hub.create_repo = cls.create_repo
        huggingface_hub.upload_folder = partial(cls.upload_folder, api)
        trainer.create_repo = cls.create_repo
        trainer.upload_folder = partial(cls.upload_folder, api)
        try:
            yield
        finally:
            huggingface_hub.create_repo = hub_create_repo
            huggingface_hub.upload_folder = hub_upload_folder
            trainer.create_repo = trainer_create_repo
            trainer.upload_folder = trainer_upload_folder

    @classmethod
    def try_login(cls, token: Optional[str] = None) -> bool:
        from modelscope import HubApi
        if token is None:
            token = os.environ.get('MODELSCOPE_API_TOKEN')
        if token:
            api = HubApi()
            api.login(token)
            return True
        return False

    @classmethod
    def create_model_repo(cls, repo_id: str, token: Optional[str] = None, private: bool = False) -> str:
        from modelscope import HubApi
        from modelscope.hub.api import ModelScopeConfig
        from modelscope.hub.constants import ModelVisibility
        assert repo_id is not None, 'Please enter a valid hub_model_id'

        if not cls.try_login(token):
            raise ValueError('Please specify a token by `--hub_token` or `MODELSCOPE_API_TOKEN=xxx`')
        cls.ms_token = token
        visibility = ModelVisibility.PRIVATE if private else ModelVisibility.PUBLIC
        api = HubApi()
        if '/' not in repo_id:
            user_name = ModelScopeConfig.get_user_info()[0]
            assert isinstance(user_name, str)
            hub_model_id = f'{user_name}/{repo_id}'
            logger.info(f"'/' not in hub_model_id, pushing to personal repo {hub_model_id}")
        try:
            api.create_model(repo_id, visibility)
        except HTTPError:
            # The remote repository has been created
            pass

        with tempfile.TemporaryDirectory() as temp_cache_dir:
            from modelscope.hub.repository import Repository
            repo = Repository(temp_cache_dir, repo_id)
            cls.add_patterns_to_gitattributes(repo, ['*.safetensors', '*.bin', '*.pt'])
            # Add 'runs/' to .gitignore, ignore tensorboard files
            cls.add_patterns_to_gitignore(repo, ['runs/', 'images/'])
            cls.add_patterns_to_file(
                repo,
                'configuration.json', ['{"framework": "pytorch", "task": "text-generation", "allow_remote": true}'],
                ignore_push_error=True)
            # Add '*.sagemaker' to .gitignore if using SageMaker
            if os.environ.get('SM_TRAINING_ENV'):
                cls.add_patterns_to_gitignore(repo, ['*.sagemaker-uploading', '*.sagemaker-uploaded'],
                                              'Add `*.sagemaker` patterns to .gitignore')
        return repo_id

    @classmethod
    def push_to_hub(cls,
                    repo_id: str,
                    folder_path: Union[str, Path],
                    path_in_repo: Optional[str] = None,
                    commit_message: Optional[str] = None,
                    commit_description: Optional[str] = None,
                    token: Union[str, bool, None] = None,
                    private: bool = False,
                    revision: Optional[str] = 'master',
                    ignore_patterns: Optional[Union[List[str], str]] = None,
                    **kwargs):
        cls.create_model_repo(repo_id, token, private)
        from modelscope import push_to_hub
        commit_message = commit_message or 'Upload folder using api'
        if commit_description:
            commit_message = commit_message + '\n' + commit_description
        if not os.path.exists(os.path.join(folder_path, 'configuration.json')):
            with open(os.path.join(folder_path, 'configuration.json'), 'w', encoding='utf-8') as f:
                f.write('{"framework": "pytorch", "task": "text-generation", "allow_remote": true}')
        if ignore_patterns:
            ignore_patterns = [p for p in ignore_patterns if p != '_*']
        if path_in_repo:
            # We don't support part submit for now
            path_in_repo = os.path.basename(folder_path)
            folder_path = os.path.dirname(folder_path)
            ignore_patterns = []
        if revision is None or revision == 'main':
            revision = 'master'
        push_to_hub(
            repo_id,
            folder_path,
            token or cls.ms_token,
            private,
            commit_message=commit_message,
            ignore_file_pattern=ignore_patterns,
            revision=revision,
            tag=path_in_repo)

    @classmethod
    def load_dataset(cls,
                     dataset_id: str,
                     subset_name: str,
                     split: str,
                     streaming: bool = False,
                     revision: Optional[str] = None,
                     download_mode: Literal['force_redownload', 'reuse_dataset_if_exists'] = 'reuse_dataset_if_exists',
                     token: Optional[str] = None,
                     **kwargs):
        from modelscope import MsDataset
        cls.try_login(token)
        if revision is None or revision == 'main':
            revision = 'master'

        return MsDataset.load(
            dataset_id,
            subset_name=subset_name,
            split=split,
            version=revision,
            download_mode=download_mode,
            use_streaming=streaming,
            trust_remote_code=True)

    @classmethod
    def download_model(cls,
                       model_id_or_path: Optional[str] = None,
                       revision: Optional[str] = None,
                       ignore_patterns: Optional[List[str]] = None,
                       token: Optional[str] = None,
                       **kwargs):
        cls.try_login(token)
        if revision is None or revision == 'main':
            revision = 'master'
        logger.info(f'Downloading the model from ModelScope Hub, model_id: {model_id_or_path}')
        from modelscope import snapshot_download
        return snapshot_download(model_id_or_path, revision, ignore_patterns=ignore_patterns, **kwargs)

    @staticmethod
    def add_patterns_to_file(repo,
                             file_name: str,
                             patterns: List[str],
                             commit_message: Optional[str] = None,
                             ignore_push_error=False) -> None:
        if isinstance(patterns, str):
            patterns = [patterns]
        if commit_message is None:
            commit_message = f'Add `{patterns[0]}` patterns to {file_name}'

        # Get current file content
        repo_dir = repo.model_dir
        file_path = os.path.join(repo_dir, file_name)
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                current_content = f.read()
        else:
            current_content = ''
        # Add the patterns to file
        content = current_content
        for pattern in patterns:
            if pattern not in content:
                if len(content) > 0 and not content.endswith('\n'):
                    content += '\n'
                content += f'{pattern}\n'

        # Write the file if it has changed
        if content != current_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                logger.debug(f'Writing {file_name} file. Content: {content}')
                f.write(content)
        try:
            repo.push(commit_message)
        except Exception as e:
            if ignore_push_error:
                pass
            else:
                raise e

    @staticmethod
    def add_patterns_to_gitignore(repo, patterns: List[str], commit_message: Optional[str] = None) -> None:
        MSHub.add_patterns_to_file(repo, '.gitignore', patterns, commit_message, ignore_push_error=True)

    @staticmethod
    def add_patterns_to_gitattributes(repo, patterns: List[str], commit_message: Optional[str] = None) -> None:
        new_patterns = []
        suffix = 'filter=lfs diff=lfs merge=lfs -text'
        for pattern in patterns:
            if suffix not in pattern:
                pattern = f'{pattern} {suffix}'
            new_patterns.append(pattern)
        file_name = '.gitattributes'
        if commit_message is None:
            commit_message = f'Add `{patterns[0]}` patterns to {file_name}'
        MSHub.add_patterns_to_file(repo, file_name, new_patterns, commit_message, ignore_push_error=True)


class HFHub(HubOperation):

    @classmethod
    def try_login(cls, token: Optional[str] = None) -> bool:
        pass

    @classmethod
    def create_model_repo(cls, repo_id: str, token: Optional[str] = None, private: bool = False) -> str:
        return api.create_repo(repo_id, token=token, private=private)

    @classmethod
    def push_to_hub(cls,
                    repo_id: str,
                    folder_path: Union[str, Path],
                    path_in_repo: Optional[str] = None,
                    commit_message: Optional[str] = None,
                    commit_description: Optional[str] = None,
                    token: Union[str, bool, None] = None,
                    private: bool = False,
                    revision: Optional[str] = 'master',
                    ignore_patterns: Optional[Union[List[str], str]] = None,
                    **kwargs):
        cls.create_model_repo(repo_id, token, private)
        if revision is None or revision == 'master':
            revision = 'main'
        return api.upload_folder(
            repo_id=repo_id,
            folder_path=folder_path,
            path_in_repo=path_in_repo,
            commit_message=commit_message,
            commit_description=commit_description,
            token=token,
            revision=revision,
            ignore_patterns=ignore_patterns,
            **kwargs)

    @classmethod
    def load_dataset(cls,
                     dataset_id: str,
                     subset_name: str,
                     split: str,
                     streaming: bool = False,
                     revision: Optional[str] = None,
                     download_mode: Literal['force_redownload', 'reuse_dataset_if_exists'] = 'reuse_dataset_if_exists',
                     num_proc: Optional[int] = None,
                     **kwargs):
        from datasets import load_dataset
        if revision is None or revision == 'master':
            revision = 'main'
        return load_dataset(
            dataset_id,
            name=subset_name,
            split=split,
            streaming=streaming,
            revision=revision,
            download_mode=download_mode,
            num_proc=num_proc)

    @classmethod
    def download_model(cls,
                       model_id_or_path: Optional[str] = None,
                       revision: Optional[str] = None,
                       ignore_patterns: Optional[List[str]] = None,
                       **kwargs):
        if revision is None or revision == 'master':
            revision = 'main'
        logger.info(f'Downloading the model from HuggingFace Hub, model_id: {model_id_or_path}')
        use_hf_transfer = strtobool(os.environ.get('USE_HF_TRANSFER', 'False'))
        if use_hf_transfer:
            from huggingface_hub import _snapshot_download
            _snapshot_download.HF_HUB_ENABLE_HF_TRANSFER = True
        from huggingface_hub import snapshot_download
        return snapshot_download(
            model_id_or_path, repo_type='model', revision=revision, ignore_patterns=ignore_patterns, **kwargs)


def get_hub(use_hf: Optional[bool] = None):
    if use_hf is None:
        use_hf = True if use_hf_hub() else False
    return {True: HFHub, False: MSHub}[use_hf]
