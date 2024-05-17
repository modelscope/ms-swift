# Copyright (c) Alibaba, Inc. and its affiliates.

import os
import subprocess
from typing import List, Optional

from swift.utils.logger import get_logger
from .constants import MASTER_MODEL_BRANCH
from .errors import GitError

logger = get_logger()


class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class GitCommandWrapper(metaclass=Singleton):
    """Some git operation wrapper
    """
    default_git_path = 'git'  # The default git command line

    def __init__(self, path: str = None):
        self.git_path = path or self.default_git_path

    def _run_git_command(self, *args) -> subprocess.CompletedProcess:
        """Run git command, if command return 0, return subprocess.response
             otherwise raise GitError, message is stdout and stderr.

        Args:
            args: List of command args.

        Raises:
            GitError: Exception with stdout and stderr.

        Returns:
            subprocess.CompletedProcess: the command response
        """
        logger.debug(' '.join(args))
        git_env = os.environ.copy()
        git_env['GIT_TERMINAL_PROMPT'] = '0'
        response = subprocess.run(
            [self.git_path, *args],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=git_env,
        )  # compatible for python3.6
        try:
            response.check_returncode()
            return response
        except subprocess.CalledProcessError as error:
            logger.error('There are error run git command.')
            raise GitError('stdout: %s, stderr: %s' % (response.stdout.decode('utf8'), error.stderr.decode('utf8')))

    def config_auth_token(self, repo_dir, auth_token):
        url = self.get_repo_remote_url(repo_dir)
        if '//oauth2' not in url:
            auth_url = self._add_token(auth_token, url)
            cmd_args = '-C %s remote set-url origin %s' % (repo_dir, auth_url)
            cmd_args = cmd_args.split(' ')
            rsp = self._run_git_command(*cmd_args)
            logger.debug(rsp.stdout.decode('utf8'))

    def _add_token(self, token: str, url: str):
        if token:
            if '//oauth2' not in url:
                url = url.replace('//', '//oauth2:%s@' % token)
        return url

    def remove_token_from_url(self, url: str):
        if url and '//oauth2' in url:
            start_index = url.find('oauth2')
            end_index = url.find('@')
            url = url[:start_index] + url[end_index + 1:]
        return url

    def is_lfs_installed(self):
        cmd = ['lfs', 'env']
        try:
            self._run_git_command(*cmd)
            return True
        except GitError:
            return False

    def git_lfs_install(self, repo_dir):
        cmd = ['-C', repo_dir, 'lfs', 'install']
        try:
            self._run_git_command(*cmd)
            return True
        except GitError:
            return False

    def clone(self, repo_base_dir: str, token: str, url: str, repo_name: str, branch: Optional[str] = None):
        """ git clone command wrapper.
        For public project, token can None, private repo, there must token.

        Args:
            repo_base_dir (str): The local base dir, the repository will be clone to local_dir/repo_name
            token (str): The git token, must be provided for private project.
            url (str): The remote url
            repo_name (str): The local repository path name.
            branch (str, optional): _description_. Defaults to None.

        Returns:
            The popen response.
        """
        url = self._add_token(token, url)
        if branch:
            clone_args = '-C %s clone %s %s --branch %s' % (repo_base_dir, url, repo_name, branch)
        else:
            clone_args = '-C %s clone %s' % (repo_base_dir, url)
        logger.debug(clone_args)
        clone_args = clone_args.split(' ')
        response = self._run_git_command(*clone_args)
        logger.debug(response.stdout.decode('utf8'))
        return response

    def add_user_info(self, repo_base_dir, repo_name):
        from .api import ModelScopeConfig
        user_name, user_email = ModelScopeConfig.get_user_info()
        if user_name and user_email:
            # config user.name and user.email if exist
            config_user_name_args = '-C %s/%s config user.name %s' % (repo_base_dir, repo_name, user_name)
            response = self._run_git_command(*config_user_name_args.split(' '))
            logger.debug(response.stdout.decode('utf8'))
            config_user_email_args = '-C %s/%s config user.email %s' % (repo_base_dir, repo_name, user_email)
            response = self._run_git_command(*config_user_email_args.split(' '))
            logger.debug(response.stdout.decode('utf8'))

    def add(self, repo_dir: str, files: List[str] = list(), all_files: bool = False):
        if all_files:
            add_args = '-C %s add -A' % repo_dir
        elif len(files) > 0:
            files_str = ' '.join(files)
            add_args = '-C %s add %s' % (repo_dir, files_str)
        add_args = add_args.split(' ')
        rsp = self._run_git_command(*add_args)
        logger.debug(rsp.stdout.decode('utf8'))
        return rsp

    def commit(self, repo_dir: str, message: str):
        """Run git commit command

        Args:
            repo_dir (str): the repository directory.
            message (str): commit message.

        Returns:
            The command popen response.
        """
        commit_args = ['-C', '%s' % repo_dir, 'commit', '-m', "'%s'" % message]
        rsp = self._run_git_command(*commit_args)
        logger.info(rsp.stdout.decode('utf8'))
        return rsp

    def checkout(self, repo_dir: str, revision: str):
        cmds = ['-C', '%s' % repo_dir, 'checkout', '%s' % revision]
        return self._run_git_command(*cmds)

    def new_branch(self, repo_dir: str, revision: str):
        cmds = ['-C', '%s' % repo_dir, 'checkout', '-b', revision]
        return self._run_git_command(*cmds)

    def get_remote_branches(self, repo_dir: str):
        cmds = ['-C', '%s' % repo_dir, 'branch', '-r']
        rsp = self._run_git_command(*cmds)
        info = [line.strip() for line in rsp.stdout.decode('utf8').strip().split(os.linesep)]
        if len(info) == 1:
            return ['/'.join(info[0].split('/')[1:])]
        else:
            return ['/'.join(line.split('/')[1:]) for line in info[1:]]

    def pull(self, repo_dir: str, remote: str = 'origin', branch: str = 'master'):
        cmds = ['-C', repo_dir, 'pull', remote, branch]
        return self._run_git_command(*cmds)

    def push(self, repo_dir: str, token: str, url: str, local_branch: str, remote_branch: str, force: bool = False):
        url = self._add_token(token, url)

        push_args = '-C %s push %s %s:%s' % (repo_dir, url, local_branch, remote_branch)
        if force:
            push_args += ' -f'
        push_args = push_args.split(' ')
        rsp = self._run_git_command(*push_args)
        logger.debug(rsp.stdout.decode('utf8'))
        return rsp

    def get_repo_remote_url(self, repo_dir: str):
        cmd_args = '-C %s config --get remote.origin.url' % repo_dir
        cmd_args = cmd_args.split(' ')
        rsp = self._run_git_command(*cmd_args)
        url = rsp.stdout.decode('utf8')
        return url.strip()

    def list_lfs_files(self, repo_dir: str):
        cmd_args = '-C %s lfs ls-files' % repo_dir
        cmd_args = cmd_args.split(' ')
        rsp = self._run_git_command(*cmd_args)
        out = rsp.stdout.decode('utf8').strip()
        files = []
        for line in out.split(os.linesep):
            files.append(line.split(' ')[-1])

        return files

    def tag(self, repo_dir: str, tag_name: str, message: str, ref: str = MASTER_MODEL_BRANCH):
        cmd_args = ['-C', repo_dir, 'tag', tag_name, '-m', '"%s"' % message, ref]
        rsp = self._run_git_command(*cmd_args)
        logger.debug(rsp.stdout.decode('utf8'))
        return rsp

    def push_tag(self, repo_dir: str, tag_name):
        cmd_args = ['-C', repo_dir, 'push', 'origin', tag_name]
        rsp = self._run_git_command(*cmd_args)
        logger.debug(rsp.stdout.decode('utf8'))
        return rsp
