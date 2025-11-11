"""
模块功能：
    该模块为视觉定位（Grounding）任务提供可视化工具，主要用于在图像上绘制带标签的边界框（bounding box）。
    视觉定位任务是指根据文本描述（ref）在图像中标注出对应物体的位置，常用于目标检测、视觉问答等多模态任务。

核心功能：
    1. 生成多样化的颜色方案，用于区分不同的物体标签
    2. 在图像上绘制归一化或像素坐标的边界框
    3. 根据背景亮度自适应调整标签文字颜色（黑/白），确保清晰可读
    4. 自动下载和缓存中文字体文件（SimSun.ttf），支持中文标签显示

应用场景：
    - Grounding 任务的结果可视化：将模型预测的边界框绘制在图像上
    - 数据集标注展示：可视化训练数据中的物体位置和对应文本
    - 多模态推理调试：检查模型是否正确理解了文本与图像区域的对应关系
    - 演示和报告：生成带标注的图像用于展示

使用示例：
    >>> from PIL import Image
    >>> from swift.llm.template import draw_bbox
    >>> 
    >>> # 加载图像
    >>> image = Image.open('example.jpg')
    >>> 
    >>> # 定义物体标签和边界框（归一化坐标 0-1000）
    >>> refs = ['cat', 'dog', 'person']
    >>> bboxes = [
    ...     [100, 200, 300, 400],  # cat: 左上(100, 200), 右下(300, 400)
    ...     [500, 100, 700, 300],  # dog
    ...     [200, 500, 400, 900]   # person
    ... ]
    >>> 
    >>> # 绘制边界框（默认使用归一化坐标）
    >>> draw_bbox(image, refs, bboxes, norm_bbox='norm1000')
    >>> image.show()  # 显示标注后的图像
    >>> 
    >>> # 使用像素坐标绘制
    >>> draw_bbox(image, refs, bboxes, norm_bbox='none')
"""
import colorsys  # 颜色空间转换：用于生成HSV到RGB的颜色映射
import itertools  # 迭代工具：用于生成颜色组合的笛卡尔积
import os  # 文件系统操作：创建缓存目录
from copy import deepcopy  # 深拷贝：避免修改原始边界框数据
from typing import Any, List, Literal  # 类型注解：Any通用类型，List列表，Literal字面量类型

import requests  # HTTP请求：用于下载字体文件
from modelscope.hub.utils.utils import get_cache_dir  # 获取modelscope缓存目录
from PIL import Image, ImageDraw, ImageFont  # PIL图像处理：Image图像对象，ImageDraw绘制工具，ImageFont字体加载


def _shuffle_colors(nums: List[Any]) -> List[Any]:
    """
    功能：
        递归打乱列表元素顺序，使用分治法将相邻元素分散排列，增加颜色多样性。
        通过将列表分成左右两半，递归打乱后交替组合，确保相邻颜色差异尽可能大。

    参数：
        nums (List[Any]): 待打乱的列表，可以是颜色值、索引或任意类型元素。

    返回：
        List[Any]: 打乱后的列表，元素顺序与原列表不同。

    示例：
        >>> colors = [1, 2, 3, 4, 5, 6, 7, 8]
        >>> _shuffle_colors(colors)
        [1, 5, 3, 7, 2, 6, 4, 8]  # 相邻元素来自左右两半，分散排列
    """
    if len(nums) == 1:  # 递归终止条件：只有1个元素无需打乱
        return nums

    mid = len(nums) // 2  # 计算中点，分割列表为左右两半

    # 递归打乱左半部分和右半部分
    left = nums[:mid]
    right = nums[mid:]
    left = _shuffle_colors(left)
    right = _shuffle_colors(right)
    
    # 交替组合：从左右列表各取一个元素，增加相邻元素差异
    new_nums = []
    for x, y in zip(left, right):
        new_nums += [x, y]
    
    # 处理长度不等的情况：将剩余元素追加到末尾
    # 例如：left=[1,2,3], right=[4,5] → new_nums=[1,4,2,5,3]
    new_nums += left[len(right):] or right[len(left):]
    return new_nums


def generate_colors():
    """
    功能：
        生成一组多样化的RGB颜色列表，用于为不同的物体标签分配不同颜色。
        通过HSV颜色空间（色调H、饱和度S、明度V）生成16种色调，并与不同的饱和度和明度组合，
        总共生成144种颜色（16色调 × 9种V-S组合），最后打乱顺序以增加相邻颜色差异。

    返回：
        List[Tuple[int, int, int]]: RGB颜色列表，每个元素是(R, G, B)元组，取值范围0-255。

    示例：
        >>> colors = generate_colors()
        >>> len(colors)
        144
        >>> colors[0]
        (255, 76, 76)  # 示例RGB值
    """
    # 生成9种V-S组合：V和S各取3个值(0.7, 0.3, 1)的笛卡尔积
    vs_combinations = [(v, s) for v, s in itertools.product([0.7, 0.3, 1], [0.7, 0.3, 1])]
    
    # 为每种V-S组合生成16种色调（H=0/16到15/16）
    # _shuffle_colors打乱色调顺序，使相邻颜色差异更大
    colors = [colorsys.hsv_to_rgb(i / 16, s, v) for v, s in vs_combinations for i in _shuffle_colors(list(range(16)))]
    
    # 将HSV颜色(0-1范围)转换为RGB整数(0-255范围)
    colors = [(int(r * 255), int(g * 255), int(b * 255)) for r, g, b in colors]
    
    # 再次打乱所有颜色，确保不同标签获得差异较大的颜色
    return _shuffle_colors(colors)


def download_file(url: str) -> str:
    """
    功能：
        从URL下载文件并缓存到本地，避免重复下载。
        文件缓存在modelscope的缓存目录下的files子目录中。

    参数：
        url (str): 文件下载URL，支持HTTP/HTTPS协议。

    返回：
        str: 下载文件的本地路径。

    示例：
        >>> font_path = download_file('https://modelscope.cn/models/Qwen/Qwen-VL-Chat/resolve/master/SimSun.ttf')
        >>> print(font_path)
        '/home/user/.cache/modelscope/files/SimSun.ttf'
    """
    url = url.rstrip('/')  # 去除URL末尾的斜杠
    file_name = url.rsplit('/', 1)[-1]  # 提取文件名：从最后一个'/'后获取
    cache_dir = os.path.join(get_cache_dir(), 'files')  # 构造缓存目录路径
    os.makedirs(cache_dir, exist_ok=True)  # 创建缓存目录（如已存在则跳过）
    req = requests.get(url)  # 发送HTTP GET请求下载文件
    file_path = os.path.join(cache_dir, file_name)  # 构造本地文件路径
    with open(file_path, 'wb') as f:  # 以二进制写模式打开文件
        f.write(req.content)  # 写入下载的内容
    return file_path


# 全局变量：预生成144种颜色用于标签分配
colors = generate_colors()

# 全局字典：记录每个标签已分配的颜色，确保相同标签使用相同颜色
color_mapping = {}


def _calculate_brightness(image, region: List[int]):
    """
    功能：
        计算图像指定区域的平均亮度（0-255），用于决定标签文字颜色（黑/白）。
        将区域转换为灰度图后，计算所有像素的平均灰度值。

    参数：
        image: PIL.Image对象，完整的原始图像。
        region (List[int]): 区域坐标[left, top, right, bottom]，像素坐标。

    返回：
        float: 平均亮度值，范围0-255（0为黑色，255为白色）。

    示例：
        >>> brightness = _calculate_brightness(image, [100, 100, 200, 150])
        >>> print(brightness)
        128.5  # 中等亮度
    """
    cropped_image = image.crop(region)  # 裁剪出指定区域
    grayscale_image = cropped_image.convert('L')  # 转换为灰度图（L模式，单通道0-255）
    pixels = list(grayscale_image.getdata())  # 获取所有像素值列表
    average_brightness = sum(pixels) / len(pixels)  # 计算平均亮度
    return average_brightness


def draw_bbox(image: Image.Image,
              ref: List[str],
              bbox: List[List[int]],
              norm_bbox: Literal['norm1000', 'none'] = 'norm1000'):
    """
    功能：
        在图像上绘制带标签的边界框，支持归一化坐标和像素坐标两种模式。
        每个物体标签自动分配不同颜色，标签文字颜色根据背景亮度自适应选择黑色或白色。

    参数：
        image (Image.Image): PIL图像对象，将在此图像上绘制边界框（原地修改）。
        ref (List[str]): 物体标签列表，每个元素对应一个边界框的文本描述。
        bbox (List[List[int]]): 边界框列表，每个元素为[left, top, right, bottom]坐标。
        norm_bbox (Literal['norm1000', 'none']): 坐标归一化模式：
            - 'norm1000': 坐标范围0-1000，需要反归一化到图像尺寸
            - 'none': 已是像素坐标，直接使用

    返回：
        无（原地修改image对象）

    示例：
        >>> from PIL import Image
        >>> image = Image.open('cat.jpg')  # 假设尺寸800x600
        >>> refs = ['小猫', '玩具球']
        >>> bboxes = [[100, 200, 300, 400], [500, 100, 700, 300]]  # 归一化坐标
        >>> draw_bbox(image, refs, bboxes, norm_bbox='norm1000')
        >>> # 边界框已绘制在image上，可保存或显示
        >>> image.save('result.jpg')
    """
    bbox = deepcopy(bbox)  # 深拷贝边界框列表，避免修改原始数据
    font_path = 'https://modelscope.cn/models/Qwen/Qwen-VL-Chat/resolve/master/SimSun.ttf'  # 中文字体URL
    
    # 坐标归一化处理：将归一化坐标转换为像素坐标
    for i, box in enumerate(bbox):
        # 将坐标转换为整数
        for i in range(len(box)):
            box[i] = int(box[i])
        # 如果是归一化坐标，反归一化到图像尺寸
        if norm_bbox == 'norm1000':
            box[0] = box[0] / 1000 * image.width   # left: 0-1000 → 0-width
            box[2] = box[2] / 1000 * image.width   # right
            box[1] = box[1] / 1000 * image.height  # top: 0-1000 → 0-height
            box[3] = box[3] / 1000 * image.height  # bottom

    draw = ImageDraw.Draw(image)  # 创建绘图对象
    
    # 绘制矩形边界框
    assert len(ref) == len(bbox), f'len(refs): {len(ref)}, len(bboxes): {len(bbox)}'  # 确保标签与边界框数量一致
    for (left, top, right, bottom), box_ref in zip(bbox, ref):
        # 为标签分配颜色：首次出现时分配新颜色，之后复用
        if box_ref not in color_mapping:
            color_mapping[box_ref] = colors[len(color_mapping) % len(colors)]
        color = color_mapping[box_ref]
        # 绘制矩形：outline指定边框颜色，width指定线宽
        draw.rectangle([(left, top), (right, bottom)], outline=color, width=3)

    # 绘制标签文字
    file_path = download_file(font_path)  # 下载中文字体文件
    font = ImageFont.truetype(file_path, 20)  # 加载字体，大小20
    for (left, top, _, _), box_ref in zip(bbox, ref):
        # 计算文字背景区域的亮度（左上角100x20区域）
        brightness = _calculate_brightness(
            image, [left, top, min(left + 100, image.width),
                    min(top + 20, image.height)])
        # 根据背景亮度选择文字颜色：暗背景用白色，亮背景用黑色
        draw.text((left, top), box_ref, fill='white' if brightness < 128 else 'black', font=font)
