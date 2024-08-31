import os
import tempfile
from concurrent.futures import Future
from pathlib import Path
from typing import List, Optional, Union

from huggingface_hub import RepoUrl
from huggingface_hub.hf_api import CommitInfo, future_compatible
from modelscope import HubApi, push_to_hub
from modelscope.hub.api import ModelScopeConfig
from modelscope.hub.constants import ModelVisibility
from modelscope.hub.repository import Repository
from modelscope.hub.utils.utils import get_cache_dir
from requests.exceptions import HTTPError
from transformers.utils import logging, strtobool

logger = logging.get_logger(__name__)


class PushToMsHubMixin:

    _use_hf_hub = strtobool(os.environ.get('USE_HF', 'False'))
    repo = None
    _cache_dir = get_cache_dir()

    @staticmethod
    def create_repo(repo_id: str, *, token: Union[str, bool, None] = None, private: bool = False, **kwargs) -> RepoUrl:
        hub_model_id = PushToMsHubMixin._create_ms_repo(repo_id, token, private)
        with tempfile.TemporaryDirectory(dir=PushToMsHubMixin._cache_dir) as temp_cache_dir:
            repo = Repository(temp_cache_dir, hub_model_id)
            PushToMsHubMixin._add_patterns_to_gitattributes(repo, ['*.safetensors', '*.bin', '*.pt'])
            # Add 'runs/' to .gitignore, ignore tensorboard files
            PushToMsHubMixin._add_patterns_to_gitignore(repo, ['runs/'])
            # Add '*.sagemaker' to .gitignore if using SageMaker
            if os.environ.get('SM_TRAINING_ENV'):
                PushToMsHubMixin._add_patterns_to_gitignore(repo, ['*.sagemaker-uploading', '*.sagemaker-uploaded'],
                                                            'Add `*.sagemaker` patterns to .gitignore')
        return RepoUrl(url=hub_model_id, )

    @staticmethod
    @future_compatible
    def upload_folder(
        *,
        repo_id: str,
        folder_path: Union[str, Path],
        path_in_repo: Optional[str] = None,
        commit_message: Optional[str] = None,
        commit_description: Optional[str] = None,
        token: Union[str, bool, None] = None,
        revision: Optional[str] = None,
        ignore_patterns: Optional[Union[List[str], str]] = None,
        run_as_future: bool = False,
        **kwargs,
    ) -> Union[CommitInfo, str, Future[CommitInfo], Future[str]]:
        if path_in_repo is not None:
            # This is a different logic from transformers
            folder_path = os.path.join(folder_path, path_in_repo)
        commit_message = commit_message or 'Upload folder using api'
        if commit_description:
            commit_message = commit_message + '\n' + commit_description
        push_to_hub(
            repo_id,
            folder_path,
            token,
            commit_message=commit_message,
            ignore_file_pattern=ignore_patterns,
            revision=revision)
        return CommitInfo(
            commit_url=f'https://www.modelscope.cn/models/{repo_id}/files',
            commit_message=commit_message,
            commit_description=commit_description,
        )

    if not _use_hf_hub:
        import huggingface_hub
        huggingface_hub.create_repo = create_repo
        huggingface_hub.upload_folder = upload_folder

    @staticmethod
    def _create_ms_repo(hub_model_id: str, hub_token: Optional[str] = None, hub_private_repo: bool = False) -> str:
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
            logger.info(f"'/' not in hub_model_id, setting hub_model_id: {hub_model_id}")
        try:
            api.create_model(hub_model_id, visibility)
        except HTTPError:
            # The remote repository has been created
            pass
        return hub_model_id

    @staticmethod
    def _add_patterns_to_file(repo: Repository,
                              file_name: str,
                              patterns: List[str],
                              commit_message: Optional[str] = None) -> None:
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
        repo.push(commit_message)

    @staticmethod
    def _add_patterns_to_gitignore(repo: Repository, patterns: List[str], commit_message: Optional[str] = None) -> None:
        PushToMsHubMixin._add_patterns_to_file(repo, '.gitignore', patterns, commit_message)

    @staticmethod
    def _add_patterns_to_gitattributes(repo: Repository,
                                       patterns: List[str],
                                       commit_message: Optional[str] = None) -> None:
        new_patterns = []
        suffix = 'filter=lfs diff=lfs merge=lfs -text'
        for pattern in patterns:
            if suffix not in pattern:
                pattern = f'{pattern} {suffix}'
            new_patterns.append(pattern)
        file_name = '.gitattributes'
        if commit_message is None:
            commit_message = f'Add `{patterns[0]}` patterns to {file_name}'
        PushToMsHubMixin._add_patterns_to_file(repo, file_name, new_patterns, commit_message)
