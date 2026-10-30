# Copyright (c) ModelScope Contributors. All rights reserved.
import io
import unittest
from PIL import Image, ImageDraw

from swift.template.vision_utils import load_image


def _png_bytes(image: Image.Image, **save_kwargs) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, 'PNG', **save_kwargs)
    return buffer.getvalue()


def _black_outline_on_transparent(mode: str) -> Image.Image:
    """A black square outline drawn on a fully transparent background."""
    base = Image.new('RGBA', (224, 224), (0, 0, 0, 0))
    ImageDraw.Draw(base).rectangle([56, 56, 168, 168], outline=(0, 0, 0, 255), width=8)
    if mode == 'RGBA':
        return base
    if mode == 'LA':
        return base.convert('LA')
    palette = Image.new('P', base.size, 0)
    palette.putpalette([0, 0, 0, 0, 0, 0] + [0] * 762)  # colour 0 transparent, colour 1 black
    palette.paste(1, mask=base.getchannel('A'))
    return palette


class TestLoadImageTransparency(unittest.TestCase):

    def test_transparent_images_are_composited_onto_white(self):
        for mode, save_kwargs in (('RGBA', {}), ('LA', {}), ('P', {'transparency': 0})):
            with self.subTest(mode=mode):
                loaded = load_image(_png_bytes(_black_outline_on_transparent(mode), **save_kwargs))
                self.assertEqual(loaded.mode, 'RGB')
                # The transparent background must become white, not black,
                # so the black outline stays visible.
                self.assertEqual(loaded.getpixel((10, 10)), (255, 255, 255))
                self.assertEqual(loaded.getpixel((112, 60)), (0, 0, 0))

    def test_fully_opaque_rgba_is_unchanged(self):
        source = Image.new('RGBA', (32, 32), (10, 20, 30, 255))
        loaded = load_image(_png_bytes(source))
        self.assertEqual(loaded.mode, 'RGB')
        self.assertEqual(loaded.getpixel((5, 5)), (10, 20, 30))

    def test_semi_transparent_pixels_blend_with_white(self):
        source = Image.new('RGBA', (16, 16), (0, 0, 0, 128))
        loaded = load_image(_png_bytes(source))
        # 50% black over white -> mid gray (PIL integer blending truncates to
        # 127), not black and not white.
        self.assertEqual(loaded.getpixel((5, 5)), (127, 127, 127))

    def test_images_without_transparency_are_unchanged(self):
        rgb = Image.new('RGB', (16, 16), (1, 2, 3))
        gray = Image.new('L', (16, 16), 77)
        palette = Image.new('P', (16, 16), 0)
        palette.putpalette([255, 0, 0] + [0] * 765)
        for source in (rgb, gray, palette):
            with self.subTest(mode=source.mode):
                loaded = load_image(_png_bytes(source))
                self.assertEqual(list(loaded.getdata()), list(source.convert('RGB').getdata()))

    def test_image_objects_are_composited_too(self):
        loaded = load_image(_black_outline_on_transparent('RGBA'))
        self.assertEqual(loaded.mode, 'RGB')
        self.assertEqual(loaded.getpixel((10, 10)), (255, 255, 255))
        self.assertEqual(loaded.getpixel((112, 60)), (0, 0, 0))


if __name__ == '__main__':
    unittest.main()
