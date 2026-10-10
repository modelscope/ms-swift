# Copyright (c) ModelScope Contributors. All rights reserved.
"""OmniJev media composition (decision H): a faithful port of the official `mso/panels.py` +
`mso/video.py` plus `mso/records.py::video_content`.

OmniJev encodes exactly ONE image per record, so every multi-media state is folded into one image:
  * several stills  -> a single numbered panel (`panels.compose`): reading order, thin border, a small
    numeral per tile so "the second picture" has something to point at;
  * a video clip    -> a 4x4 mosaic of 16 uniformly-spaced, letterboxed, timestamp-stamped frames
    (`video.video_state`), which `records.video_content` then crops BACK into those 16 frames and
    interleaves with "t=<s>:" text (the mosaic is only an intermediate/cache artifact);
  * a single still  -> passed through untouched (bit-for-bit unaffected).

These are model-local helpers (pure PIL / ffmpeg); `OmniJevTemplate` calls `compose_state` +
`video_content` + `content_to_text`. Nothing here touches shared swift framework code.

`video_state` is the OFFLINE tool that turns a raw clip into the `{"images": [mosaic], "video": {...}}`
state OmniJev's `system_one` consumes. The training/inference Template expects that already-composed
state -- a raw video file is NOT decoded inline (that would run ffmpeg per sample per epoch); callers
with raw clips run `video_state` first, exactly as the official demo does.

VERBATIM: `_grid` / `compose` / `compose_path` (panels.py), `font` / `letterbox` / `probe_duration` /
`grab` / `video_mosaic` / `video_state` (video.py) and `video_content` (records.py) are byte-faithful
ports so the composed pixels match the checkpoint's training distribution. The swift-facing adapters
(`_as_pil` / `compose_state` / `content_to_text`) are thin and marked as such.
"""
import hashlib
import io
import os
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from PIL import Image, ImageDraw, ImageFont

# ---- mso/panels.py (verbatim) ----
# MSO_NO_PANELS=1 restores the old behaviour (encode only the first image). It exists so the same
# weights can be scored both ways, which is the only way to attribute a change to this fix.
PANELS_OFF = os.environ.get('MSO_NO_PANELS', '').strip().lower() in ('1', 'true', 'yes', 'on')
_BORDER = 6
_LABEL = 22

# ---- mso/video.py (verbatim) ----
K, COLS, TILE = 16, 4, 384

# OPT markers (mso/records.py); duplicated here so content_to_text is self-contained. The Template /
# branch module import them from their own definitions -- these are only for the body assembler.
OPT_OPEN, OPT_CLOSE = '<|opt|>', '<|/opt|>'

ImageLike = Union[str, 'Image.Image']


def _grid(n):
    return (1, 1) if n <= 1 else (2, 1) if n == 2 else (2, 2) if n <= 4 else (3, (n + 2) // 3)


def compose(paths, max_side=448):
    """tile the images into a single PIL image, in reading order, each panel numbered (panels.compose).
    Accepts file paths OR already-open PIL images (swift may hand either); the port opens paths."""
    ims = []
    for p in paths:
        im = _as_pil(p)
        im = im.convert('RGB')
        if max(im.size) > max_side:
            im.thumbnail((max_side, max_side))
        ims.append(im)
    cols, rows = _grid(len(ims))
    cw = max(i.width for i in ims)
    ch = max(i.height for i in ims)
    W = cols * cw + (cols + 1) * _BORDER
    H = rows * (ch + _LABEL) + (rows + 1) * _BORDER
    canvas = Image.new('RGB', (W, H), (32, 32, 32))
    draw = ImageDraw.Draw(canvas)
    for k, im in enumerate(ims):
        c, r = k % cols, k // cols
        x = _BORDER + c * (cw + _BORDER)
        y = _BORDER + r * (ch + _LABEL + _BORDER)
        draw.text((x + 2, y + 4), '%d' % (k + 1), fill=(255, 255, 255))
        canvas.paste(im, (x + (cw - im.width) // 2, y + _LABEL + (ch - im.height) // 2))
    return canvas


def compose_path(paths, cache_dir, max_side=448):
    """same, written to `cache_dir` and reused on later epochs; returns a path to a single image. A
    single path is returned unchanged, so nothing is copied or re-encoded for the common case.
    Falls back to the in-memory `compose` when any input is not a statable file path (a PIL image)."""
    if not paths:
        return None
    if len(paths) == 1 or PANELS_OFF:
        return paths[0]
    if not all(isinstance(p, str) for p in paths):
        return compose(paths, max_side)  # a PIL image cannot be os.stat'd for the cache key
    stamps = [(str(p), os.stat(p).st_size, os.stat(p).st_mtime_ns) for p in paths]
    key = hashlib.sha1((repr(stamps) + '|%d' % max_side).encode()).hexdigest()[:20]
    out = os.path.join(cache_dir, key + '.jpg')
    if os.path.exists(out):
        return out
    os.makedirs(cache_dir, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=key + '.', suffix='.part', dir=cache_dir)
    os.close(fd)
    try:
        compose(paths, max_side).save(tmp, format='JPEG', quality=88)  # the .part name carries no format
        os.replace(tmp, out)
        return out
    except Exception:
        # Reject incomplete states instead of answering a different question.
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def font(sz):
    for f in ('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', '/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf',
              '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', '/usr/share/fonts/dejavu/DejaVuSans.ttf'):
        try:
            return ImageFont.truetype(f, sz)
        except Exception:
            pass
    return ImageFont.load_default()


FT = font(26)


def letterbox(img, t):
    img = img.convert('RGB')
    w, h = img.size
    s = min(TILE / w, TILE / h)
    img = img.resize((max(1, int(w * s)), max(1, int(h * s))), Image.BILINEAR)
    tile = Image.new('RGB', (TILE, TILE), (0, 0, 0))
    tile.paste(img, ((TILE - img.size[0]) // 2, (TILE - img.size[1]) // 2))
    d = ImageDraw.Draw(tile)
    label = 't=%ds' % t
    d.rectangle([0, 0, 16 + 15 * len(label), 36], fill=(0, 0, 0))
    d.text((6, 4), label, fill=(255, 255, 255), font=FT)
    return tile


def probe_duration(path):
    out = subprocess.run(['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'csv=p=0', path],
                         capture_output=True, text=True, timeout=60).stdout.strip()
    try:
        return float(out)
    except ValueError:
        raise ValueError('could not read the video (ffprobe gave no duration)')


def grab(path, t):
    out = subprocess.run(['ffmpeg', '-v', 'error', '-ss', '%.3f' % t, '-i', path, '-frames:v', '1', '-f', 'image2pipe',
                          '-vcodec', 'mjpeg', '-q:v', '3', 'pipe:1'], capture_output=True, timeout=120).stdout
    return Image.open(io.BytesIO(out)).convert('RGB') if out else None


def video_mosaic(path, out_jpg):
    """16 uniformly spaced frames, letterboxed to 384 px, timestamp stamped, one 4x4 mosaic (video.py)."""
    dur = probe_duration(path)
    if dur < 1.0:
        raise ValueError('video shorter than one second')
    ts = [(i + 0.5) / K * dur for i in range(K)]
    with ThreadPoolExecutor(max_workers=4) as ex:
        imgs = list(ex.map(lambda t: grab(path, min(t, max(0.0, dur - 0.05))), ts))
    last = None
    tiles = []
    for t, im in zip(ts, imgs):
        if im is None:
            im = last if last is not None else Image.new('RGB', (TILE, TILE), (0, 0, 0))
        last = im
        tiles.append(letterbox(im, int(round(t))))
    mosaic = Image.new('RGB', (COLS * TILE, (K // COLS) * TILE))
    for k, tl in enumerate(tiles):
        r, c = divmod(k, COLS)
        mosaic.paste(tl, (c * TILE, r * TILE))
    mosaic.save(out_jpg, quality=85)
    return dur, [int(round(t)) for t in ts]


def video_state(path, out_jpg=None):
    """OFFLINE tool: a raw clip -> `{"images": [mosaic path], "video": {...}}`, the state to hand
    `OmniJevTemplate` (mirrors `mso/video.py::video_state`, the input to official `system_one`)."""
    out_jpg = out_jpg or (os.path.splitext(path)[0] + '.mosaic.jpg')
    dur, ts = video_mosaic(path, out_jpg)
    return {
        'images': [out_jpg],
        'video': {
            'n_frames': K,
            'cols': COLS,
            'tile': TILE,
            'timestamps': ts,
            'duration': round(dur, 1)
        }
    }


def video_content(img, video, instr) -> Tuple[List['Image.Image'], List[Dict[str, Any]]]:
    """One image, or a 4x4 mosaic split back into its frames, each preceded by its timestamp (verbatim
    `mso/records.py::video_content`, adapted to accept a PIL image OR a path). Returns (list of PIL
    images in prompt order, official chat content list of {'type':'image'|'text', ...})."""
    im = _as_pil(img).convert('RGB')
    if not video:
        return [im], [{'type': 'image'}, {'type': 'text', 'text': instr}]
    n_frames, cols, tile = video['n_frames'], video['cols'], video['tile']
    frames: List[Image.Image] = []
    content: List[Dict[str, Any]] = [{
        'type': 'text',
        'text': 'A video of {:.0f} seconds; {} frames sampled uniformly in order.'.format(
            video.get('duration', 0), n_frames)
    }]
    for k in range(n_frames):
        r, c = divmod(k, cols)
        frames.append(im.crop((c * tile, r * tile, (c + 1) * tile, (r + 1) * tile)))
        content.append({'type': 'text', 'text': 't={}s:'.format(video['timestamps'][k])})
        content.append({'type': 'image'})
    content.append({'type': 'text', 'text': instr})
    return frames, content


# ---- swift-facing adapters (thin; NOT part of the official port) ----
def _as_pil(img) -> 'Image.Image':
    """Accept an already-open PIL image, a file path, an http(s) URL, a base64 string, raw bytes, OR an
    HF-datasets Image-feature dict (`{'bytes': ..., 'path': ...}`, what swift's dataset pipeline hands the
    `images` column). The official helpers only ever saw local paths, so anything else is resolved through
    swift's own `load_image` (which fetches URLs / decodes base64); a PIL image is returned untouched."""
    if isinstance(img, Image.Image):
        return img
    if isinstance(img, dict) and 'bytes' in img:
        img = img['bytes'] or img['path']
    from swift.template.vision_utils import load_image
    return load_image(img)


def compose_state(images: Sequence[ImageLike], video: Optional[Dict[str, Any]], cache_dir: Optional[str] = None,
                  max_side: int = 448) -> ImageLike:
    """Mirror `mso/infer.py::system_one`'s media choice: several stills and NO video -> one numbered
    panel (`compose_path`); otherwise `images[0]` (the pre-composed video mosaic, or the single still).
    Returns the ONE image (path or PIL) that `video_content` then reads."""
    if not images:
        raise ValueError('OmniJev state carries no images; `system_one` encodes exactly one image per record.')
    if len(images) > 1 and not video:
        cache_dir = cache_dir or os.environ.get('MSO_PANELS') or os.path.join(tempfile.gettempdir(), 'mso_panels')
        return compose_path(list(images), cache_dir, max_side)
    return images[0]


def content_to_text(content: Sequence[Dict[str, Any]]) -> str:
    """Official chat content list -> the swift message string: each `{'type':'image'}` becomes one
    `<image>` tag (which `Qwen2VLTemplate.replace_tag` expands to `<|vision_start|><|image_pad|><|vision_end|>`
    and `_encode` grows to the grid_thw token count), each `{'type':'text'}` its verbatim text. The tag
    count equals the frame count `video_content` returned, so `_add_default_tags` adds nothing."""
    parts: List[str] = []
    for c in content:
        if c.get('type') == 'image':
            parts.append('<image>')
        else:
            parts.append(c.get('text', ''))
    return ''.join(parts)


def option_body(opt_texts: Sequence[str]) -> str:
    """The assistant body appended after the chat prompt: every option block wrapped in the markers,
    verbatim `mso/records.py::Collator` (`"".join("{}{}{}".format(OPT_OPEN, t, OPT_CLOSE) ...)`)."""
    return ''.join('{}{}{}'.format(OPT_OPEN, t, OPT_CLOSE) for t in opt_texts)
