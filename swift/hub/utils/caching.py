# Copyright (c) Alibaba, Inc. and its affiliates.

import hashlib
import os
import pickle
import tempfile
from shutil import move, rmtree

from swift.hub.constants import MODEL_META_FILE_NAME, MODEL_META_MODEL_ID
from swift.utils.logger import get_logger

logger = get_logger()
"""Implements caching functionality, used internally only
"""


class FileSystemCache(object):
    KEY_FILE_NAME = '.msc'
    """Local file cache.
    """

    def __init__(
        self,
        cache_root_location: str,
        **kwargs,
    ):
        """Base file system cache interface.

        Args:
            cache_root_location (str): The root location to store files.
            kwargs(dict): The keyword arguments.
        """
        os.makedirs(cache_root_location, exist_ok=True)
        self.cache_root_location = cache_root_location
        self.load_cache()

    def get_root_location(self):
        return self.cache_root_location

    def load_cache(self):
        self.cached_files = []
        cache_keys_file_path = os.path.join(self.cache_root_location, FileSystemCache.KEY_FILE_NAME)
        if os.path.exists(cache_keys_file_path):
            with open(cache_keys_file_path, 'rb') as f:
                self.cached_files = pickle.load(f)

    def save_cached_files(self):
        """Save cache metadata."""
        # save new meta to tmp and move to KEY_FILE_NAME
        cache_keys_file_path = os.path.join(self.cache_root_location, FileSystemCache.KEY_FILE_NAME)
        # TODO: Sync file write
        fd, fn = tempfile.mkstemp()
        with open(fd, 'wb') as f:
            pickle.dump(self.cached_files, f)
        move(fn, cache_keys_file_path)

    def get_file(self, key):
        """Check the key is in the cache, if exist, return the file, otherwise return None.

        Args:
            key(str): The cache key.

        Raises:
            None
        """
        pass

    def put_file(self, key, location):
        """Put file to the cache.

        Args:
            key (str): The cache key
            location (str): Location of the file, we will move the file to cache.

        Raises:
            None
        """
        pass

    def remove_key(self, key):
        """Remove cache key in index, The file is removed manually

        Args:
            key (dict): The cache key.
        """
        if key in self.cached_files:
            self.cached_files.remove(key)
            self.save_cached_files()

    def exists(self, key):
        for cache_file in self.cached_files:
            if cache_file == key:
                return True

        return False

    def clear_cache(self):
        """Remove all files and metadata from the cache
        In the case of multiple cache locations, this clears only the last one,
        which is assumed to be the read/write one.
        """
        rmtree(self.cache_root_location)
        self.load_cache()

    def hash_name(self, key):
        return hashlib.sha256(key.encode()).hexdigest()


class ModelFileSystemCache(FileSystemCache):
    """Local cache file layout
       cache_root/owner/model_name/individual cached files and cache index file '.mcs'
       Save only one version for each file.
    """

    def __init__(self, cache_root, owner=None, name=None):
        """Put file to the cache
        Args:
            cache_root(`str`): The modelscope local cache root(default: ~/.cache/modelscope/)
            owner(`str`): The model owner.
            name('str'): The name of the model
        Returns:
        Raises:
            None
        <Tip>
            model_id = {owner}/{name}
        </Tip>
        """
        if owner is None or name is None:
            # get model meta from
            super().__init__(os.path.join(cache_root))
            self.load_model_meta()
        else:
            super().__init__(os.path.join(cache_root, owner, name))
            self.model_meta = {MODEL_META_MODEL_ID: '%s/%s' % (owner, name)}
            self.save_model_meta()

    def load_model_meta(self):
        meta_file_path = os.path.join(self.cache_root_location, MODEL_META_FILE_NAME)
        if os.path.exists(meta_file_path):
            with open(meta_file_path, 'rb') as f:
                self.model_meta = pickle.load(f)
        else:
            self.model_meta = {MODEL_META_MODEL_ID: 'unknown'}

    def get_model_id(self):
        return self.model_meta[MODEL_META_MODEL_ID]

    def save_model_meta(self):
        meta_file_path = os.path.join(self.cache_root_location, MODEL_META_FILE_NAME)
        with open(meta_file_path, 'wb') as f:
            pickle.dump(self.model_meta, f)

    def get_file_by_path(self, file_path):
        """Retrieve the cache if there is file match the path.

        Args:
            file_path (str): The file path in the model.

        Returns:
            path: the full path of the file.
        """
        for cached_file in self.cached_files:
            if file_path == cached_file['Path']:
                cached_file_path = os.path.join(self.cache_root_location, cached_file['Path'])
                if os.path.exists(cached_file_path):
                    return cached_file_path
                else:
                    self.remove_key(cached_file)

        return None

    def get_file_by_path_and_commit_id(self, file_path, commit_id):
        """Retrieve the cache if there is file match the path.

        Args:
            file_path (str): The file path in the model.
            commit_id (str): The commit id of the file

        Returns:
            path: the full path of the file.
        """
        for cached_file in self.cached_files:
            if file_path == cached_file['Path'] and \
               (cached_file['Revision'].startswith(commit_id) or commit_id.startswith(cached_file['Revision'])):
                cached_file_path = os.path.join(self.cache_root_location, cached_file['Path'])
                if os.path.exists(cached_file_path):
                    return cached_file_path
                else:
                    self.remove_key(cached_file)

        return None

    def get_file_by_info(self, model_file_info):
        """Check if exist cache file.

        Args:
            model_file_info (ModelFileInfo): The file information of the file.

        Returns:
            str: The file path.
        """
        cache_key = self.__get_cache_key(model_file_info)
        for cached_file in self.cached_files:
            if cached_file == cache_key:
                orig_path = os.path.join(self.cache_root_location, cached_file['Path'])
                if os.path.exists(orig_path):
                    return orig_path
                else:
                    self.remove_key(cached_file)
                    break

        return None

    def __get_cache_key(self, model_file_info):
        cache_key = {
            'Path': model_file_info['Path'],
            'Revision': model_file_info['Revision'],  # commit id
        }
        return cache_key

    def exists(self, model_file_info):
        """Check the file is cached or not.

        Args:
            model_file_info (CachedFileInfo): The cached file info

        Returns:
            bool: If exists return True otherwise False
        """
        key = self.__get_cache_key(model_file_info)
        is_exists = False
        for cached_key in self.cached_files:
            if cached_key['Path'] == key['Path'] and (cached_key['Revision'].startswith(key['Revision'])
                                                      or key['Revision'].startswith(cached_key['Revision'])):
                is_exists = True
                break
        file_path = os.path.join(self.cache_root_location, model_file_info['Path'])
        if is_exists:
            if os.path.exists(file_path):
                return True
            else:
                self.remove_key(model_file_info)  # someone may manual delete the file
        return False

    def remove_if_exists(self, model_file_info):
        """We in cache, remove it.

        Args:
            model_file_info (ModelFileInfo): The model file information from server.
        """
        for cached_file in self.cached_files:
            if cached_file['Path'] == model_file_info['Path']:
                self.remove_key(cached_file)
                file_path = os.path.join(self.cache_root_location, cached_file['Path'])
                if os.path.exists(file_path):
                    os.remove(file_path)
                break

    def put_file(self, model_file_info, model_file_location):
        """Put model on model_file_location to cache, the model first download to /tmp, and move to cache.

        Args:
            model_file_info (str): The file description returned by get_model_files.
            model_file_location (str): The location of the temporary file.

        Returns:
            str: The location of the cached file.
        """
        self.remove_if_exists(model_file_info)  # backup old revision
        cache_key = self.__get_cache_key(model_file_info)
        cache_full_path = os.path.join(self.cache_root_location,
                                       cache_key['Path'])  # Branch and Tag do not have same name.
        cache_file_dir = os.path.dirname(cache_full_path)
        if not os.path.exists(cache_file_dir):
            os.makedirs(cache_file_dir, exist_ok=True)
        # We can't make operation transaction
        move(model_file_location, cache_full_path)
        self.cached_files.append(cache_key)
        self.save_cached_files()
        return cache_full_path
