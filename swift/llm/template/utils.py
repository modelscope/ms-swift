import base64
import hashlib
import math
import os
import re
from collections.abc import Mapping
from copy import deepcopy
from io import BytesIO
from typing import Any, Callable, List, TypeVar, Union, Tuple, Set, Dict, Type, Optional, Sequence

import numpy as np
import requests
import torch
from packaging import version


History = List[Union[Tuple[str, str], List[str]]]
Prompt = List[Union[str, List[int], List[str]]]
StopWords = Prompt
Context = Union[str, List[int]]
Messages = List[Dict[str, Union[str, List[Dict]]]]


# >>> internvl
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def split_str_parts_by(text: str, delimiters: List[str]):
    """Split the text field into parts.

    Args:
        text: A text to be split.
        delimiters: The delimiters.

    Returns:
        The split text in list of dicts.
    """
    assert isinstance(text, str), f'text: {text}'
    all_start_chars = [d[0] for d in delimiters]
    all_length = [len(d) for d in delimiters]

    text_list = []
    last_words = ''

    while len(text) > 0:
        for char_idx, char in enumerate(text):
            match_index = [idx for idx, start_char in enumerate(all_start_chars) if start_char == char]
            is_delimiter = False
            for index in match_index:
                if text[char_idx:char_idx + all_length[index]] == delimiters[index]:
                    if text_list:
                        text_list[-1]['content'] = last_words
                    elif last_words:
                        text_list.append({'key': '', 'content': last_words})
                    last_words = ''
                    text_list.append({'key': delimiters[index]})
                    text = text[char_idx + all_length[index]:]
                    is_delimiter = True
                    break
            if not is_delimiter:
                last_words += char
            else:
                break
        if last_words == text:
            text = ''

    if len(text_list):
        text_list[-1]['content'] = last_words
    else:
        text_list.append({'key': '', 'content': last_words})
    return text_list


def split_parts_by_regex(text_list: list, regex_delimiters: Dict[str, List[float]]) -> None:
    import re
    compiled_patterns = [(re.compile(pattern), scale) for pattern, scale in regex_delimiters.items()]
    for i in range(len(text_list) - 1, -1, -1):
        item = text_list[i]
        if item.get('key') == '':
            res_text = item['content']
            last_idx = 0
            segments = []

            for pattern, scale in compiled_patterns:
                matches = list(re.finditer(pattern, res_text))
                for match in matches:
                    if match.start() > last_idx:
                        segments.append({'key': '', 'content': res_text[last_idx:match.start()]})
                    segments.append({'key': scale[0], 'content': match.group(0)})
                    last_idx = match.end()

            if last_idx < len(res_text):
                segments.insert(0, {'key': '', 'content': res_text[last_idx:]})

            if segments:
                text_list[i:i + 1] = segments


def _decode_prompt(prompt: str, tmp_dir: str = 'tmp') -> str:
    pattern = r'<(?:img|audio|video)>(.+?)</(?:img|audio|video)>'
    match_iter = re.finditer(pattern, prompt)
    new_content = ''
    idx = 0
    for m in match_iter:
        span = m.span(1)
        img_base64 = m.group(1)
        img_path = _from_base64(img_base64, tmp_dir)
        new_content += prompt[idx:span[0]] + img_path
        idx = span[1]
    new_content += prompt[idx:]
    return new_content


def _to_base64(img_path: Union[str, 'PIL.Image.Image', bytes]) -> str:
    if isinstance(img_path, str) and not os.path.isfile(img_path):
        # base64
        return img_path
    if isinstance(img_path, str):
        # local_path
        with open(img_path, 'rb') as f:
            _bytes = f.read()
    elif not isinstance(img_path, bytes):  # PIL.Image.Image
        bytes_io = BytesIO()
        img_path.save(bytes_io, format='png')
        _bytes = bytes_io.getvalue()
    else:
        _bytes = img_path
    img_base64: str = base64.b64encode(_bytes).decode('utf-8')
    return img_base64


def _from_base64(img_base64: Union[str, 'PIL.Image.Image'], tmp_dir: str = 'tmp') -> str:
    from PIL import Image
    if not isinstance(img_base64, str):  # PIL.Image.Image
        img_base64 = _to_base64(img_base64)
    if os.path.isfile(img_base64) or img_base64.startswith('http'):
        return img_base64
    sha256_hash = hashlib.sha256(img_base64.encode('utf-8')).hexdigest()
    img_path = os.path.join(tmp_dir, f'{sha256_hash}.png')
    image = Image.open(BytesIO(base64.b64decode(img_base64)))
    if not os.path.exists(img_path):
        image.save(img_path)
    return img_path


def decode_base64(*,
                  messages: Optional[Messages] = None,
                  prompt: Optional[str] = None,
                  images: Optional[List[str]] = None,
                  tmp_dir: str = 'tmp') -> Dict[str, Any]:
    # base64 -> local_path
    os.makedirs(tmp_dir, exist_ok=True)
    res = {}
    if messages is not None:
        res_messages = []
        for m in messages:
            m_new = deepcopy(m)
            m_new['content'] = _decode_prompt(m_new['content'], tmp_dir)
            res_messages.append(m_new)
        res['messages'] = res_messages
    if prompt is not None:
        prompt = _decode_prompt(prompt, tmp_dir)
        res['prompt'] = prompt
    if images is not None:
        res_images = []
        for image in images:
            image = _from_base64(image, tmp_dir)
            res_images.append(image)
        res['images'] = res_images
    return res


def to_device(inputs: Any, device: torch.device) -> Any:
    """Move inputs to a device"""
    if callable(getattr(inputs, 'to', None)):
        return inputs.to(device=device)

    if isinstance(inputs, Mapping):
        res = {}
        for k, v in inputs.items():
            res[k] = to_device(v, device)
    elif isinstance(inputs, Sequence) and not isinstance(inputs, str):
        res = []
        for b in inputs:
            res.append(to_device(b, device))
    else:
        res = inputs
    return res


def upper_bound(lo: int, hi: int, cond: Callable[[int], bool]) -> int:
    # The upper bound satisfying the condition "cond".
    while lo < hi:
        mid = (lo + hi + 1) >> 1  # lo + (hi-lo+1)>>1
        if cond(mid):
            lo = mid
        else:
            hi = mid - 1
    return lo


def fetch_one(element: Union[Tuple, List, Set, Dict, Any], type: Type = None) -> Any:
    if isinstance(element, (tuple, set, list)):
        for ele in element:
            out = fetch_one(ele)
            if out and (type is None or isinstance(out, type)):
                return out
    elif isinstance(element, dict):
        return fetch_one(list(element.values()))
    else:
        return element


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
    return T.Resize((int(width_scaled), int(height_scaled)))(img)


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
    from swift.llm.template.template import Template
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
    bridge.set_bridge('torch')
    clip_end_sec = 60
    clip_start_sec = 0
    num_frames = 24
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
    container = av.open(video_io)
    total_frames = container.streams.video[0].frames
    indices = np.arange(0, total_frames, total_frames / 8).astype(int)
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
    from swift.llm.template.template import get_env_args
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
    size_factor = get_env_args('size_factor', int, FRAME_FACTOR)
    assert not (fps and nframes), 'Only accept either `fps` or `nframes`'
    if nframes is not None:
        nframes = round_by_factor(nframes, size_factor)
    else:
        fps = FPS
        nframes = video.size(0) / info['video_fps'] * fps
        nframes = round_by_factor(nframes, size_factor)
        min_frames = get_env_args('min_frames', int, FPS_MIN_FRAMES)
        max_frames = get_env_args('max_frames', int, FPS_MAX_FRAMES)
        if nframes < min_frames:
            nframes = ceil_by_factor(min_frames, size_factor)
        if nframes > max_frames:
            nframes = floor_by_factor(max_frames, size_factor)

    if not (size_factor <= nframes and nframes <= video.size(0)):
        raise ValueError(f'nframes should in interval [{size_factor}, {video.size(0)}], but got {nframes}.')

    idx = torch.linspace(0, video.size(0) - 1, nframes).round().long()
    height, width = video.shape[2:]
    video = video[idx]

    min_pixels = get_env_args('min_pixels', int, VIDEO_MIN_PIXELS)
    total_pixels = get_env_args('total_pixels', int, VIDEO_TOTAL_PIXELS)
    max_pixels = get_env_args('max_pixels', int, None)
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
