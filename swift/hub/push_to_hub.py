# Copyright (c) Alibaba, Inc. and its affiliates.

import concurrent.futures
import os
import shutil
from multiprocessing import Manager, Process, Value

from swift.utils.logger import get_logger
from .api import HubApi
from .constants import DEFAULT_REPOSITORY_REVISION, ModelVisibility

logger = get_logger()

_executor = concurrent.futures.ProcessPoolExecutor(max_workers=8)
_queues = dict()
_flags = dict()
_tasks = dict()
_manager = None


def _api_push_to_hub(repo_name,
                     output_dir,
                     token,
                     private=True,
                     commit_message='',
                     tag=None,
                     source_repo='',
                     ignore_file_pattern=None,
                     revision=DEFAULT_REPOSITORY_REVISION):
    try:
        api = HubApi()
        api.login(token)
        api.push_model(
            repo_name,
            output_dir,
            visibility=ModelVisibility.PUBLIC if not private else ModelVisibility.PRIVATE,
            chinese_name=repo_name,
            commit_message=commit_message,
            tag=tag,
            original_model_id=source_repo,
            ignore_file_pattern=ignore_file_pattern,
            revision=revision)
        commit_message = commit_message or 'No commit message'
        logger.info(f'Successfully upload the model to {repo_name} with message: {commit_message}')
        return True
    except Exception as e:
        logger.error(f'Error happens when uploading model {repo_name} with message: {commit_message}: {e}')
        return False


def push_to_hub(repo_name,
                output_dir,
                token=None,
                private=True,
                retry=3,
                commit_message='',
                tag=None,
                source_repo='',
                ignore_file_pattern=None,
                revision=DEFAULT_REPOSITORY_REVISION):
    """
    Args:
        repo_name: The repo name for the modelhub repo
        output_dir: The local output_dir for the checkpoint
        token: The user api token, function will check the `MODELSCOPE_API_TOKEN` variable if this argument is None
        private: If is a private repo, default True
        retry: Retry times if something error in uploading, default 3
        commit_message: The commit message
        tag: The tag of this commit
        source_repo: The source repo (model id) which this model comes from
        ignore_file_pattern: The file pattern to be ignored in uploading.
        revision: The branch to commit to
    Returns:
        The boolean value to represent whether the model is uploaded.
    """
    if token is None:
        token = os.environ.get('MODELSCOPE_API_TOKEN')
    if ignore_file_pattern is None:
        ignore_file_pattern = os.environ.get('UPLOAD_IGNORE_FILE_PATTERN')
    assert repo_name is not None
    assert token is not None, 'Either pass in a token or to set `MODELSCOPE_API_TOKEN` in the environment variables.'
    assert os.path.isdir(output_dir)
    assert 'configuration.json' in os.listdir(output_dir) or 'configuration.yaml' in os.listdir(output_dir) \
           or 'configuration.yml' in os.listdir(output_dir)

    logger.info(f'Uploading {output_dir} to {repo_name} with message {commit_message}')
    for i in range(retry):
        if _api_push_to_hub(repo_name, output_dir, token, private, commit_message, tag, source_repo,
                            ignore_file_pattern, revision):
            return True
    return False


def push_to_hub_async(repo_name,
                      output_dir,
                      token=None,
                      private=True,
                      commit_message='',
                      tag=None,
                      source_repo='',
                      ignore_file_pattern=None,
                      revision=DEFAULT_REPOSITORY_REVISION):
    """
    Args:
        repo_name: The repo name for the modelhub repo
        output_dir: The local output_dir for the checkpoint
        token: The user api token, function will check the `MODELSCOPE_API_TOKEN` variable if this argument is None
        private: If is a private repo, default True
        commit_message: The commit message
        tag: The tag of this commit
        source_repo: The source repo (model id) which this model comes from
        ignore_file_pattern: The file pattern to be ignored in uploading
        revision: The branch to commit to
    Returns:
        A handler to check the result and the status
    """
    if token is None:
        token = os.environ.get('MODELSCOPE_API_TOKEN')
    if ignore_file_pattern is None:
        ignore_file_pattern = os.environ.get('UPLOAD_IGNORE_FILE_PATTERN')
    assert repo_name is not None
    assert token is not None, 'Either pass in a token or to set `MODELSCOPE_API_TOKEN` in the environment variables.'
    assert os.path.isdir(output_dir)
    assert 'configuration.json' in os.listdir(output_dir) or 'configuration.yaml' in os.listdir(output_dir) \
           or 'configuration.yml' in os.listdir(output_dir)

    logger.info(f'Uploading {output_dir} to {repo_name} with message {commit_message}')
    return _executor.submit(_api_push_to_hub, repo_name, output_dir, token, private, commit_message, tag, source_repo,
                            ignore_file_pattern, revision)


def submit_task(q, b):
    while True:
        b.value = False
        item = q.get()
        logger.info(item)
        b.value = True
        if not item.pop('done', False):
            delete_dir = item.pop('delete_dir', False)
            output_dir = item.get('output_dir')
            try:
                push_to_hub(**item)
                if delete_dir and os.path.exists(output_dir):
                    shutil.rmtree(output_dir)
            except Exception as e:
                logger.error(e)
        else:
            break


class UploadStrategy:
    cancel = 'cancel'
    wait = 'wait'


def push_to_hub_in_queue(queue_name, strategy=UploadStrategy.cancel, **kwargs):
    assert queue_name is not None and len(queue_name) > 0, 'Please specify a valid queue name!'
    global _manager
    if _manager is None:
        _manager = Manager()
    if queue_name not in _queues:
        _queues[queue_name] = _manager.Queue()
        _flags[queue_name] = Value('b', False)
        process = Process(target=submit_task, args=(_queues[queue_name], _flags[queue_name]))
        process.start()
        _tasks[queue_name] = process

    queue = _queues[queue_name]
    flag: Value = _flags[queue_name]
    if kwargs.get('done', False):
        queue.put(kwargs)
    elif flag.value and strategy == UploadStrategy.cancel:
        logger.error(f'Another uploading is running, '
                     f'this uploading with message {kwargs.get("commit_message")} will be canceled.')
    else:
        queue.put(kwargs)


def wait_for_done(queue_name):
    process: Process = _tasks.pop(queue_name, None)
    if process is None:
        return
    process.join()

    _queues.pop(queue_name)
    _flags.pop(queue_name)
