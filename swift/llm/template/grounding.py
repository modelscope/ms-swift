from typing import Any, Dict, List, Literal

from PIL import Image


def normalize_bbox(images: List[Image.Image],
                   objects: Dict[str, List[Any]],
                   norm_bbox: Literal['norm1000', 'none'] = 'norm1000') -> None:
    if not objects or not images or norm_bbox == 'none':
        return
    bbox_list = objects['bbox']
    bbox_type = objects.pop('bbox_type', None) or 'real'
    image_id_list = objects.pop('image_id', None) or []
    image_id_list += [0] * (len(bbox_list) - len(image_id_list))
    for bbox, image_id in zip(bbox_list, image_id_list):
        if norm_bbox == 'norm1000':
            if bbox_type == 'norm1':
                width, height = 1, 1
            else:
                image = images[image_id]
                width, height = image.width, image.height
            for i, (x, y) in enumerate(zip(bbox[::2], bbox[1::2])):
                bbox[2 * i] = int(x / width * 1000)
                bbox[2 * i + 1] = int(y / height * 1000)
