import base64
import math
import os
from io import BytesIO
from typing import Any, Callable, List, TypeVar, Union

import numpy as np
import requests
import torch
from packaging import version

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
                        if i * j <= max_num and i * j >= min_num)
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


def rescale_image(img: 'PIL.Image.Image', rescale_image: int = -1) -> 'PIL.Image.Image':
    import torchvision.transforms as T
    width = img.width
    height = img.height
    if rescale_image <= 0 or width * height <= rescale_image:
        return img

    ratio = width / height
    height_scaled = math.pow(rescale_image / ratio, 0.5)
    width_scaled = height_scaled * ratio
    return T.Resize((int(height_scaled), int(width_scaled)))(img)


_T = TypeVar('_T')


def load_file(path: Union[str, _T]) -> Union[BytesIO, _T]:
    res = path
    if isinstance(path, str):
        path = path.strip()
        if path.startswith('http'):
            request_kwargs = {}
            timeout = float(os.getenv('TIMEOUT', '60'))
            if timeout > 0:
                request_kwargs['timeout'] = timeout
            content = requests.get(path, **request_kwargs).content
            res = BytesIO(content)
        elif os.path.exists(path):
            with open(path, 'rb') as f:
                res = BytesIO(f.read())
        else:  # base64_str
            import binascii
            try:
                data = base64.b64decode(path)
                res = BytesIO(data)
            except (ValueError, binascii.Error) as error:
                if len(path) < 200:
                    raise ValueError(f'invalid image: "{path}"')
                else:
                    raise ValueError(f'invalid image: {error}')
    return res


def load_file_decorator(func):

    def new_func(path, *args, **kwargs):
        path = load_file(path)
        res = func(path, *args, **kwargs)
        return res

    return new_func


@load_file_decorator
def load_image(image: Union['PIL.Image.Image', BytesIO]) -> 'PIL.Image.Image':
    from PIL import Image
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


@load_file_decorator
def load_video_internvl(video_io: BytesIO, bound=None, num_segments=32):
    from decord import VideoReader, cpu
    from PIL import Image
    vr = VideoReader(video_io, ctx=cpu(0), num_threads=1)
    max_frame = len(vr) - 1
    fps = float(vr.get_avg_fps())

    images = []
    frame_indices = _get_index(bound, fps, max_frame, first_idx=0, num_segments=num_segments)
    for frame_index in frame_indices:
        images.append(Image.fromarray(vr[frame_index].asnumpy()).convert('RGB'))
    return images


def draw_plot(img_dir: str, bbox: List[int], bbox_type: str, output_file: str):
    from PIL import Image, ImageDraw
    from .template import Template
    image = Image.open(img_dir)

    objects = [{'bbox': bbox, 'bbox_type': bbox_type, 'image': 0}]
    Template.normalize_bbox(objects, [image], 'real')
    bbox = objects[0]['bbox']
    draw = ImageDraw.Draw(image)
    draw.rectangle(bbox, outline='red', width=2)
    image.save(output_file)


@load_file_decorator
def load_video_cogvlm2(video_io: BytesIO) -> np.ndarray:
    from decord import cpu, VideoReader, bridge
    from .template import get_env_args
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


@load_file_decorator
def load_video_llava(video_io: BytesIO) -> np.ndarray:
    import av
    from .template import get_env_args
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


@load_file_decorator
def load_video_minicpmv_mplug_owl3(video_io: BytesIO, max_num_frames):
    from PIL import Image
    from decord import VideoReader, cpu  # pip install decord

    def uniform_sample(_l, _n):
        gap = len(_l) / _n
        idxs = [int(i * gap + gap / 2) for i in range(_n)]
        return [_l[i] for i in idxs]

    vr = VideoReader(video_io, ctx=cpu(0))
    sample_fps = round(vr.get_avg_fps() / 1)  # FPS
    frame_idx = [i for i in range(0, len(vr), sample_fps)]

    if len(frame_idx) > max_num_frames:
        frame_idx = uniform_sample(frame_idx, max_num_frames)
    frames = vr.get_batch(frame_idx).asnumpy()
    frames = [Image.fromarray(v.astype('uint8')) for v in frames]
    return frames


@load_file_decorator
def load_audio_qwen(audio_io: BytesIO, sampling_rate: int):
    import librosa
    return librosa.load(audio_io, sr=sampling_rate)[0]


def load_video_qwen2(video_path: str):
    from .template import get_env_args
    import torchvision
    from torchvision import io, transforms
    from qwen_vl_utils.vision_process import (round_by_factor, FPS, FRAME_FACTOR, FPS_MIN_FRAMES, FPS_MAX_FRAMES,
                                              VIDEO_MIN_PIXELS, VIDEO_MAX_PIXELS, VIDEO_TOTAL_PIXELS, smart_resize,
                                              ceil_by_factor, floor_by_factor)
    from torchvision.transforms import InterpolationMode

    if version.parse(torchvision.__version__) >= version.parse('0.19'):
        video_path = load_file(video_path)
    video, _, info = io.read_video(
        video_path,
        pts_unit='sec',
        output_format='TCHW',
    )
    nframes = get_env_args('nframes', int, None)
    fps = get_env_args('fps', int, None)
    size_factor = get_env_args('frame_factor', int, FRAME_FACTOR, ['size_factor'])
    assert not (fps and nframes), 'Only accept either `fps` or `nframes`'
    if nframes is not None:
        nframes = round_by_factor(nframes, size_factor)
    else:
        if fps is None:
            fps = FPS
        nframes = video.size(0) / info['video_fps'] * fps
        nframes = round_by_factor(nframes, size_factor)
        min_frames = get_env_args('fps_min_frames', int, FPS_MIN_FRAMES, ['min_frames'])
        max_frames = get_env_args('fps_max_frames', int, FPS_MAX_FRAMES, ['max_frames'])
        if nframes < min_frames:
            nframes = ceil_by_factor(min_frames, size_factor)
        if nframes > max_frames:
            nframes = floor_by_factor(max_frames, size_factor)

    if not (size_factor <= nframes and nframes <= video.size(0)):
        raise ValueError(f'nframes should in interval [{size_factor}, {video.size(0)}], but got {nframes}.')

    idx = torch.linspace(0, video.size(0) - 1, nframes).round().long()
    height, width = video.shape[2:]
    video = video[idx]

    min_pixels = get_env_args('video_min_pixels', int, VIDEO_MIN_PIXELS, ['min_pixels'])
    total_pixels = get_env_args('video_total_pixels', int, VIDEO_TOTAL_PIXELS, ['total_pixels'])
    max_pixels = get_env_args('video_max_pixels', int, None, ['max_pixels'])
    if max_pixels is None:
        max_pixels = VIDEO_MAX_PIXELS
        max_pixels = max(min(max_pixels, total_pixels / nframes * size_factor), min_pixels * 1.05)
    # resize
    resized_height = get_env_args('resized_height', int, None)
    resized_width = get_env_args('resized_width', int, None)
    if resized_height and resized_width:
        resized_height, resized_width = smart_resize(
            resized_height,
            resized_width,
            factor=size_factor,
        )
    else:
        resized_height, resized_width = smart_resize(
            height,
            width,
            factor=size_factor,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )

    video = transforms.functional.resize(
        video,
        [resized_height, resized_width],
        interpolation=InterpolationMode.BICUBIC,
        antialias=True,
    ).float()
    return video


if __name__ == '__main__':
    # A test main to draw bbox
    draw_plot('man.jpg', [354, 462, 580, 738], 'norm_1000', 'man_bbox.jpg')
