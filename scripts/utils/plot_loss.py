import os

from swift.utils import plot_images

ckpt_dir = 'output/xxx/vx-xxx'
if __name__ == '__main__':
    images_dir = os.path.join(ckpt_dir, 'images')
    tb_dir = os.path.join(ckpt_dir, 'runs')
    plot_images(images_dir, tb_dir, ['train/loss'], 0.9)
