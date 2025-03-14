# Copyright (c) Alibaba, Inc. and its affiliates.
import base64
import math
import os
import re
from io import BytesIO
from typing import Any, Callable, List, TypeVar, Union

import numpy as np
import requests
import torch
from PIL import Image

from swift.utils import get_env_args

# >>> internvl
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _build_transform(input_size):
    import torchvision.transforms as T
    from torchvision.transforms.functional import InterpolationMode
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform


def _find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def _dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set((i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1)
                        if min_num <= i * j <= max_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = _find_closest_aspect_ratio(aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = ((i % (target_width // image_size)) * image_size, (i // (target_width // image_size)) * image_size,
               ((i % (target_width // image_size)) + 1) * image_size, ((i //
                                                                        (target_width // image_size)) + 1) * image_size)
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images


# <<< internvl


def rescale_image(img: Image.Image, max_pixels: int) -> Image.Image:
    import torchvision.transforms as T
    width = img.width
    height = img.height
    if max_pixels is None or max_pixels <= 0 or width * height <= max_pixels:
        return img

    ratio = width / height
    height_scaled = math.sqrt(max_pixels / ratio)
    width_scaled = height_scaled * ratio
    return T.Resize((int(height_scaled), int(width_scaled)))(img)


_T = TypeVar('_T')


def load_file(path: Union[str, bytes, _T]) -> Union[BytesIO, _T]:
    res = path
    if isinstance(path, str):
        path = path.strip()
        if path.startswith('http'):
            request_kwargs = {}
            timeout = float(os.getenv('TIMEOUT', '300'))
            if timeout > 0:
                request_kwargs['timeout'] = timeout
            content = requests.get(path, **request_kwargs).content
            res = BytesIO(content)
        elif os.path.exists(path) or (not path.startswith('data:') and len(path) <= 200):
            path = os.path.abspath(os.path.expanduser(path))
            with open(path, 'rb') as f:
                res = BytesIO(f.read())
        else:  # base64_str
            data = path
            if data.startswith('data:'):
                match_ = re.match(r'data:(.+?);base64,(.+)', data)
                assert match_ is not None
                data = match_.group(2)
            data = base64.b64decode(data)
            res = BytesIO(data)
    elif isinstance(path, bytes):
        res = BytesIO(path)
    return res


def load_image(image: Union[str, bytes, Image.Image]) -> Image.Image:
    image = load_file(image)
    if isinstance(image, BytesIO):
        image = Image.open(image)
    if image.mode != 'RGB':
        image = image.convert('RGB')
    return image


def load_batch(path_list: List[Union[str, None, Any, BytesIO]],
               load_func: Callable[[Any], _T] = load_image) -> List[_T]:
    res = []
    assert isinstance(path_list, (list, tuple)), f'path_list: {path_list}'
    for path in path_list:
        if path is None:  # ignore None
            continue
        res.append(load_func(path))
    return res


def _get_index(bound, fps, max_frame, first_idx=0, num_segments=32):
    if bound:
        start, end = bound[0], bound[1]
    else:
        start, end = -100000, 100000
    start_idx = max(first_idx, round(start * fps))
    end_idx = min(round(end * fps), max_frame)
    seg_size = float(end_idx - start_idx) / num_segments
    frame_indices = np.array(
        [int(start_idx + (seg_size / 2) + np.round(seg_size * idx)) for idx in range(num_segments)])
    return frame_indices


def transform_image(image, input_size=448, max_num=12):
    transform = _build_transform(input_size=input_size)
    images = _dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values


def load_video_internvl(video: Union[str, bytes], bound=None, num_segments=32):
    from decord import VideoReader, cpu
    video_io = load_file(video)
    vr = VideoReader(video_io, ctx=cpu(0), num_threads=1)
    max_frame = len(vr) - 1
    fps = float(vr.get_avg_fps())

    images = []
    frame_indices = _get_index(bound, fps, max_frame, first_idx=0, num_segments=num_segments)
    for frame_index in frame_indices:
        images.append(Image.fromarray(vr[frame_index].asnumpy()).convert('RGB'))
    return images


def load_video_cogvlm2(video: Union[str, bytes]) -> np.ndarray:
    from decord import cpu, VideoReader, bridge
    video_io = load_file(video)
    bridge.set_bridge('torch')
    clip_end_sec = 60
    clip_start_sec = 0
    num_frames = get_env_args('num_frames', int, 24)
    decord_vr = VideoReader(video_io, ctx=cpu(0))
    duration = len(decord_vr)  # duration in terms of frames
    start_frame = int(clip_start_sec * decord_vr.get_avg_fps())
    end_frame = min(duration, int(clip_end_sec * decord_vr.get_avg_fps())) if \
        clip_end_sec is not None else duration
    frame_id_list = np.linspace(start_frame, end_frame - 1, num_frames, dtype=int)
    video_data = decord_vr.get_batch(frame_id_list)
    video_data = video_data.permute(3, 0, 1, 2)
    return video_data


def load_video_llava(video: Union[str, bytes]) -> np.ndarray:
    import av
    video_io = load_file(video)
    container = av.open(video_io)
    total_frames = container.streams.video[0].frames
    num_frames = get_env_args('num_frames', int, 16)
    indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    frames = []
    container.seek(0)
    start_index = indices[0]
    end_index = indices[-1]
    for i, frame in enumerate(container.decode(video=0)):
        if i > end_index:
            break
        if i >= start_index and i in indices:
            frames.append(frame)
    return np.stack([x.to_ndarray(format='rgb24') for x in frames])


def load_video_minicpmv_mplug_owl3(video: Union[str, bytes], max_num_frames):

    from decord import VideoReader, cpu  # pip install decord

    def uniform_sample(_l, _n):
        gap = len(_l) / _n
        idxs = [int(i * gap + gap / 2) for i in range(_n)]
        return [_l[i] for i in idxs]

    video_io = load_file(video)
    vr = VideoReader(video_io, ctx=cpu(0))
    sample_fps = round(vr.get_avg_fps() / 1)  # FPS
    frame_idx = [i for i in range(0, len(vr), sample_fps)]

    if len(frame_idx) > max_num_frames:
        frame_idx = uniform_sample(frame_idx, max_num_frames)
    frames = vr.get_batch(frame_idx).asnumpy()
    frames = [Image.fromarray(v.astype('uint8')) for v in frames]
    return frames


def load_audio(audio: Union[str, bytes], sampling_rate: int, return_sr: bool = False):
    import librosa
    audio_io = load_file(audio)
    res = librosa.load(audio_io, sr=sampling_rate)
    return res if return_sr else res[0]


def load_video_valley(video: Union[str, bytes]):
    import decord
    from torchvision import transforms
    video_io = load_file(video)
    video_reader = decord.VideoReader(video_io)
    decord.bridge.set_bridge('torch')
    video = video_reader.get_batch(np.linspace(0, len(video_reader) - 1, 8).astype(np.int_)).byte()
    images = [transforms.ToPILImage()(image.permute(2, 0, 1)).convert('RGB') for image in video]
    return images


def load_video_ovis2(video_path, num_frames):
    from moviepy.editor import VideoFileClip
    with VideoFileClip(video_path) as clip:
        total_frames = int(clip.fps * clip.duration)
        if total_frames <= num_frames:
            sampled_indices = range(total_frames)
        else:
            stride = total_frames / num_frames
            sampled_indices = [
                min(total_frames - 1, int((stride * i + stride * (i + 1)) / 2)) for i in range(num_frames)
            ]
        frames = [clip.get_frame(index / clip.fps) for index in sampled_indices]
        frames = [Image.fromarray(frame, mode='RGB') for frame in frames]
    return frames
