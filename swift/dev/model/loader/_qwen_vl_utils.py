# Copyright (c) ModelScope Contributors. All rights reserved.
"""Qwen-VL ``vision_process`` setup, internalized from legacy ``swift.model.models.qwen``.

This is a dev-loader-time tweak of the ``qwen_vl_utils.vision_process`` module (env-driven pixel/frame
budgets + a base64/url-tolerant video reader), NOT a reusable model monkey-patch -- so it lives in dev
rather than ``twinkle.patch``. The base64 materialization still routes through ``swift.template.load_file``
because dev's Qwen-VL path is still on the legacy template (that coupling is retired with #3, not here);
what this module DOES remove is the dependency on ``swift.model.models.qwen``.
"""
import os

from swift.dev.utils import get_env_args


def _get_new_read_video_func(read_video_func, read_backend):
    if read_backend == 'torchvision':

        def _new_read_video(ele: dict):
            try:
                return read_video_func(ele)
            except Exception:
                from swift.template import load_file  # base64
                ele['video'] = load_file(ele['video'])
                return read_video_func(ele)
    else:

        def _new_read_video(ele: dict):
            from swift.template import load_file
            ele['video'] = load_file(ele['video'])
            return read_video_func(ele)

    return _new_read_video


def patch_qwen_vl_utils(vision_process):
    if hasattr(vision_process, '_patch'):
        return
    if os.getenv('VIDEO_MAX_PIXELS') and not os.getenv('VIDEO_TOTAL_PIXELS'):
        # https://github.com/QwenLM/Qwen2.5-VL/issues/1120
        os.environ['VIDEO_TOTAL_PIXELS'] = str(int(128000 * 28 * 28 * 0.9))
    res = {}
    for key in [
            'image_factor',  # image_patch_size * SPATIAL_MERGE_SIZE
            'min_pixels',  # IMAGE_MIN_TOKEN_NUM * image_factor ** 2
            'max_pixels',
            'video_min_pixels',
            'video_max_pixels',
            'video_total_pixels',
            #
            'max_ratio',
            'frame_factor',
            'fps',
            'fps_min_frames',
            'fps_max_frames',
            # qwen3_vl
            'image_max_token_num',
            'image_min_token_num',
            'spatial_merge_size',
            'video_max_token_num',
            'video_min_token_num',
    ]:
        type_func = float if key == 'fps' else int
        default_value = getattr(vision_process, key.upper(), None)
        if default_value is None:
            # Skip keys not supported by the specific vision_process implementation
            continue
        val = get_env_args(key, type_func, default_value)
        setattr(vision_process, key.upper(), val)
        res[key] = val
    # Patch video reader if available
    backends = getattr(vision_process, 'VIDEO_READER_BACKENDS', None)
    for read_backend in ['torchvision', 'decord', 'torchcodec']:
        func_key = f'_read_video_{read_backend}'
        _read_video = getattr(vision_process, func_key, None)
        if _read_video is not None:
            _new_read_video = _get_new_read_video_func(_read_video, read_backend)
            if isinstance(backends, dict):
                backends[read_backend] = _new_read_video
            elif backends is None:  # keye_vl
                setattr(vision_process, func_key, _new_read_video)
    vision_process._patch = True
    return res
