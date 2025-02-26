import colorsys
import itertools
import os
from copy import deepcopy
from typing import Any, List, Literal

import requests
from modelscope.hub.utils.utils import get_cache_dir
from PIL import Image, ImageDraw, ImageFont


def _shuffle_colors(nums: List[Any]) -> List[Any]:
    if len(nums) == 1:
        return nums

    mid = len(nums) // 2

    left = nums[:mid]
    right = nums[mid:]
    left = _shuffle_colors(left)
    right = _shuffle_colors(right)
    new_nums = []
    for x, y in zip(left, right):
        new_nums += [x, y]
    new_nums += left[len(right):] or right[len(left):]
    return new_nums


def generate_colors():
    vs_combinations = [(v, s) for v, s in itertools.product([0.7, 0.3, 1], [0.7, 0.3, 1])]
    colors = [colorsys.hsv_to_rgb(i / 16, s, v) for v, s in vs_combinations for i in _shuffle_colors(list(range(16)))]
    colors = [(int(r * 255), int(g * 255), int(b * 255)) for r, g, b in colors]
    return _shuffle_colors(colors)


def download_file(url: str) -> str:
    url = url.rstrip('/')
    file_name = url.rsplit('/', 1)[-1]
    cache_dir = os.path.join(get_cache_dir(), 'files')
    os.makedirs(cache_dir, exist_ok=True)
    req = requests.get(url)
    file_path = os.path.join(cache_dir, file_name)
    with open(file_path, 'wb') as f:
        f.write(req.content)
    return file_path


colors = generate_colors()
color_mapping = {}


def _calculate_brightness(image, region: List[int]):
    cropped_image = image.crop(region)
    grayscale_image = cropped_image.convert('L')
    pixels = list(grayscale_image.getdata())
    average_brightness = sum(pixels) / len(pixels)
    return average_brightness


def draw_bbox(image: Image.Image,
              ref: List[str],
              bbox: List[List[int]],
              norm_bbox: Literal['norm1000', 'none'] = 'norm1000'):
    bbox = deepcopy(bbox)
    font_path = 'https://modelscope.cn/models/Qwen/Qwen-VL-Chat/resolve/master/SimSun.ttf'
    # norm bbox
    for i, box in enumerate(bbox):
        for i in range(len(box)):
            box[i] = int(box[i])
        if norm_bbox == 'norm1000':
            box[0] = box[0] / 1000 * image.width
            box[2] = box[2] / 1000 * image.width
            box[1] = box[1] / 1000 * image.height
            box[3] = box[3] / 1000 * image.height

    draw = ImageDraw.Draw(image)
    # draw bbox
    assert len(ref) == len(bbox), f'len(refs): {len(ref)}, len(bboxes): {len(bbox)}'
    for (left, top, right, bottom), box_ref in zip(bbox, ref):
        if box_ref not in color_mapping:
            color_mapping[box_ref] = colors[len(color_mapping) % len(colors)]
        color = color_mapping[box_ref]
        draw.rectangle([(left, top), (right, bottom)], outline=color, width=3)
    # draw text
    file_path = download_file(font_path)
    font = ImageFont.truetype(file_path, 20)
    for (left, top, _, _), box_ref in zip(bbox, ref):
        brightness = _calculate_brightness(
            image, [left, top, min(left + 100, image.width),
                    min(top + 20, image.height)])
        draw.text((left, top), box_ref, fill='white' if brightness < 128 else 'black', font=font)
