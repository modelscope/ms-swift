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
    """
    功能：
        构建用于 InternVL 等多模态模型的图像预处理变换管道。
        该管道依次执行：RGB 转换 → 双三次插值缩放 → 张量化 → ImageNet 标准化。
        使用 ImageNet 预训练的均值和标准差进行归一化，确保与预训练模型对齐。

    参数：
        input_size (int): 目标图像尺寸（正方形边长）。
            - 示例：448, 224

    返回：
        torchvision.transforms.Compose: 组合变换对象，可直接应用于 PIL 图像。

    示例：
        >>> from PIL import Image
        >>> transform = _build_transform(448)
        >>> img = Image.open('cat.jpg')
        >>> tensor = transform(img)  # 输出 shape=(3, 448, 448)
    """
    import torchvision.transforms as T
    from torchvision.transforms.functional import InterpolationMode
    
    # 使用 ImageNet 数据集的均值和标准差
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    
    # 构建变换管道（按顺序执行）
    transform = T.Compose([
        # 1> RGB 转换：确保图像为 3 通道 RGB 模式
        # 如果已是 RGB 则不变，否则转换（如 RGBA、L 等模式）
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        
        # 2> 缩放：调整图像到目标尺寸（正方形）
        # 使用双三次插值（BICUBIC）保证高质量缩放
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        
        # 3> 张量化：PIL.Image → torch.Tensor
        # 像素值从 [0, 255] 归一化到 [0.0, 1.0]
        # ToTensor 方法实际上做了3件事：
        # 1. 改变数据布局：HWC → CHW
        #    PIL.Image: (height, width, channels)
        #    Tensor:    (channels, height, width)
        
        # 2. 改变数据类型：uint8 → float32
        #    PIL:    uint8  [0, 255]
        #    Tensor: float32
        
        # 3. 归一化数值：除以 255
        #    [0, 255] → [0.0, 1.0]
        T.ToTensor(),
        
        # 4> 标准化：使用 ImageNet 均值和标准差
        # 公式：(x - mean) / std
        # 输出范围约 [-2, 2]（取决于原始像素值）
        T.Normalize(mean=MEAN, std=STD)
    ])
    
    return transform


def _find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    """
    功能：
        从候选宽高比列表中找到与输入图像宽高比最接近的目标宽高比。
        用于动态分辨率处理（如 InternVL 的 dynamic preprocess），选择最合适的切分策略。
        当多个候选宽高比差异相同时，优先选择面积更大的（更接近原始图像面积）。

    参数：
        aspect_ratio (float): 输入图像的宽高比（width / height）。
            - 示例：1.5 (宽1.5倍于高)
        target_ratios (List[Tuple[int, int]]): 候选宽高比列表，每个元素为 (width_ratio, height_ratio)。
            - 示例：[(1,1), (1,2), (2,1), (2,2), ...]
        width (int): 输入图像的宽度（像素）。
        height (int): 输入图像的高度（像素）。
        image_size (int): 单个块的基准尺寸（如 448）。

    返回：
        Tuple[int, int]: 最接近的目标宽高比 (width_ratio, height_ratio)。

    示例：
        >>> # 示例1：横向图像（宽高比 2.0）
        >>> target_ratios = [(1,1), (2,1), (1,2), (2,2)]
        >>> _find_closest_aspect_ratio(2.0, target_ratios, 800, 400, 448)
        (2, 1)  # 宽高比 2:1 最接近输入的 2.0
        
        >>> # 示例2：纵向图像（宽高比 0.5）
        >>> _find_closest_aspect_ratio(0.5, target_ratios, 400, 800, 448)
        (1, 2)  # 宽高比 1:2 最接近输入的 0.5
    """
    # 初始化最佳匹配：差异为无穷大，默认宽高比 1:1
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    
    # 计算输入图像的总像素面积
    area = width * height
    
    # 遍历所有候选宽高比，寻找最接近的
    for ratio in target_ratios:
        # 计算候选宽高比的数值（如 (2,1) → 2.0）
        target_aspect_ratio = ratio[0] / ratio[1]
        
        # 计算与输入宽高比的绝对差异
        # 例如：输入 2.0，候选 2.0 → diff=0.0
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        
        # 情况1：找到更接近的宽高比
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff  # 更新最小差异
            best_ratio = ratio  # 更新最佳宽高比
        
        # 情况2：差异相同，选择面积更大的
        # 条件：area > 0.5 * (目标总面积)
        # 目标总面积 = image_size² * ratio[0] * ratio[1]
        # 例如：image_size=448, ratio=(2,2) → 目标面积=448²*4=802816
        # 如果输入面积 > 0.5*802816=401408，则选择这个更大的宽高比
        elif ratio_diff == best_ratio_diff:
            # 核心思想：只有当输入图像足够大时，才使用更多的块
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio  # 更新为面积更大的候选
    
    return best_ratio


def _dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    """
    thumbnail: 缩略图
    功能：
        动态分辨率预处理图像，根据宽高比自适应切分为多个块。
        用于 InternVL 等多模态模型，保留高分辨率图像细节的同时控制计算量。
        核心思路：找到最接近输入宽高比的切分策略，将图像切分为 N×M 个块，
        每个块缩放到 image_size×image_size，可选添加缩略图用于全局信息。

    参数：
        image (PIL.Image.Image): 输入图像。
        min_num (int): 最小块数（默认 1）。
        max_num (int): 最大块数（默认 12）。
        image_size (int): 单个块的尺寸（默认 448）。
        use_thumbnail (bool): 是否添加缩略图（默认 False）。

    返回：
        List[PIL.Image.Image]: 处理后的图像块列表。
            - 长度 = blocks（或 blocks+1，若启用缩略图）
            - 每个块尺寸 = image_size × image_size

    示例：
        >>> from PIL import Image
        >>> img = Image.open('photo.jpg')  # 假设 1600×800（宽高比 2:1）
        >>> blocks = _dynamic_preprocess(img, max_num=6, image_size=448)
        >>> len(blocks)
        2  # 选择 2×1 切分策略，共 2 个块
        >>> blocks[0].size
        (448, 448)
    """
    # 1> 获取原始图像尺寸和宽高比
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height  # 例如：1600/800 = 2.0

    # 2> 生成候选宽高比列表
    # 目标：找到所有可能的 (i, j) 组合，满足 min_num ≤ i×j ≤ max_num
    # 例如：max_num=6 → 候选包括 (1,1), (2,1), (1,2), (2,2), (3,2), (2,3), (3,3) 等
    # 生成逻辑：
    # - 遍历总块数 n (从 min_num 到 max_num)
    # - 对于每个 n，枚举所有 i, j 使得 i×j ≤ max_num
    # - 使用 set 去重（如 (2,3) 和 (3,2) 是不同的宽高比）
    target_ratios = set((i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1)
                        if min_num <= i * j <= max_num)
    
    # 按块数排序（从小到大），优先考虑块数少的（计算量小）
    # 例如：[(1,1), (2,1), (1,2), (2,2), (3,1), (1,3), ...] 按 i×j 排序
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # 3> 找到最接近的宽高比
    # 调用 _find_closest_aspect_ratio 从候选中选择最佳匹配
    # 例如：输入 2.0 → 可能选择 (2,1) 表示 2×1 切分
    target_aspect_ratio = _find_closest_aspect_ratio(aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # 4> 计算目标尺寸和块数
    # 如果选择 (2,1)，image_size=448 → target_width=896, target_height=448
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]  # 总块数 = 2×1 = 2

    # 5> 缩放图像到目标尺寸
    # 例如：1600×800 → 896×448
    resized_img = image.resize((target_width, target_height))
    
    # 6> 切分为多个块
    processed_images = []
    for i in range(blocks):
        # 计算当前块的裁剪坐标 (left, top, right, bottom)
        # 布局：按行优先顺序（从左到右，从上到下）
        # 
        # 关键计算：
        # - 每行块数 = target_width // image_size
        # - 当前块的列索引 = i % (每行块数)
        # - 当前块的行索引 = i // (每行块数)
        # 
        # 例如：2×1 切分 (target_width=896, target_height=448, image_size=448)
        # 每行块数 = 896 // 448 = 2
        # i=0: 列=0%2=0, 行=0//2=0 → box=(0, 0, 448, 448)     # 左块
        # i=1: 列=1%2=1, 行=1//2=0 → box=(448, 0, 896, 448)   # 右块
        box = ((i % (target_width // image_size)) * image_size, 
               (i // (target_width // image_size)) * image_size,
               ((i % (target_width // image_size)) + 1) * image_size, 
               ((i // (target_width // image_size)) + 1) * image_size)
        
        # 裁剪出当前块
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    
    # 验证块数正确
    assert len(processed_images) == blocks
    
    # 7> 可选添加缩略图（用于保留全局信息）
    # 仅当启用 use_thumbnail 且块数 > 1 时添加
    # 缩略图：将原图缩放到 image_size×image_size（不保持宽高比）
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    
    return processed_images


# <<< internvl


def rescale_image(img: Image.Image, max_pixels: int) -> Image.Image:
    """
    功能：
        按比例缩放图像以满足最大像素数限制，同时保持原始宽高比。
        如果图像的总像素数超过指定的最大像素数，则等比例缩放至满足限制；
        否则返回原图不做修改。此方法常用于减少显存占用和加速模型推理。
    
    参数：
        img (Image.Image): 输入的 PIL 图像对象
        max_pixels (int): 允许的最大像素数（宽度 × 高度）
            - 如果为 None 或 <= 0，不进行缩放
            - 如果图像像素数 <= max_pixels，不进行缩放
    
    返回：
        Image.Image: 缩放后的 PIL 图像对象（如果需要缩放）或原图（如果不需要缩放）
    
    示例：
        # 示例1：缩放超大图像
        from PIL import Image
        img = Image.open('large_image.jpg')  # 假设尺寸为 2000x1500 (3,000,000 像素)
        scaled_img = rescale_image(img, max_pixels=1000000)  # 限制为 1M 像素
        # 结果：scaled_img 尺寸约为 1155x866 (1,000,230 像素)
        # 保持原始宽高比 2000:1500 = 1155:866 ≈ 1.333
        
        # 示例2：图像已满足要求，不缩放
        small_img = Image.open('small_image.jpg')  # 假设尺寸为 800x600 (480,000 像素)
        scaled_img = rescale_image(small_img, max_pixels=1000000)
        # 结果：scaled_img == small_img（返回原图，未修改）
        
        # 示例3：max_pixels 为 None，不缩放
        img = Image.open('any_image.jpg')
        scaled_img = rescale_image(img, max_pixels=None)
        # 结果：scaled_img == img（返回原图）
        
        # 示例4：横向图像缩放
        wide_img = Image.new('RGB', (3000, 1000))  # 宽图：3,000,000 像素
        scaled_img = rescale_image(wide_img, max_pixels=600000)
        # 结果：scaled_img 尺寸约为 1500x500 (750,000 像素接近目标)
        
        # 示例5：纵向图像缩放
        tall_img = Image.new('RGB', (1000, 4000))  # 高图：4,000,000 像素
        scaled_img = rescale_image(tall_img, max_pixels=500000)
        # 结果：scaled_img 尺寸约为 354x1414 (500,556 像素接近目标)
    """
    # 导入 torchvision 的变换模块，用于图像缩放
    import torchvision.transforms as T
    
    # 获取图像的原始宽度和原始高度
    width = img.width
    height = img.height
    
    # 判断是否需要缩放：
    # 1) max_pixels 为 None 或 <= 0：不限制像素数
    # 2) 图像总像素数 <= max_pixels：已满足要求
    # 如果满足任一条件，直接返回原图
    if max_pixels is None or max_pixels <= 0 or width * height <= max_pixels:
        return img

    # 计算图像的宽高比（aspect ratio）
    ratio = width / height
    
    # 计算缩放后的高度
    # 推导：设缩放后宽高为 w', h'，则：
    #   w' * h' = max_pixels  (目标像素数)
    #   w' / h' = ratio       (保持宽高比)
    # => w' = ratio * h'
    # => (ratio * h') * h' = max_pixels
    # => h' = sqrt(max_pixels / ratio)
    height_scaled = math.sqrt(max_pixels / ratio)
    
    # 计算缩放后的宽度：宽度 = 高度 × 宽高比
    width_scaled = height_scaled * ratio
    
    # 使用 torchvision 的 Resize 变换进行图像缩放
    # 注意：Resize 接受 (height, width) 顺序的元组
    # 将浮点数转换为整数，然后应用到图像
    return T.Resize((int(height_scaled), int(width_scaled)))(img)


_T = TypeVar('_T')


def load_file(path: Union[str, bytes, _T]) -> Union[BytesIO, _T]:
    """
    功能：
        统一处理多种文件输入格式，将文件内容转换为 BytesIO 对象以便后续处理
        该函数是一个通用的文件加载工具，支持从 HTTP/HTTPS URL、本地文件路径、Base64 编码字符串
        和原始字节数据中加载文件内容。对于字符串和字节类型的输入，函数会将其转换为 BytesIO 对象；
        对于其他类型的输入（泛型 _T），函数会直接返回原始对象。
    
    参数：
        path (Union[str, bytes, _T]): 文件输入，支持以下格式：
            - str: 字符串类型，可以是以下任意一种：
                * HTTP/HTTPS URL（如 'https://example.com/file.dat'）
                  会通过网络请求下载文件内容，支持通过环境变量 TIMEOUT 设置超时时间（默认 300 秒）
                * 本地文件路径（如 '/path/to/file.txt' 或 '~/file.txt'）
                  支持相对路径和绝对路径，会自动展开用户目录（~）
                * Base64 编码字符串（可带前缀 'data:image/png;base64,' 或不带前缀）
                  会自动解码 Base64 字符串为二进制数据
            - bytes: 文件的原始字节数据，会直接转换为 BytesIO 对象
            - _T: 其他任意类型的对象，会原样返回（泛型支持）
    
    返回值：
        Union[BytesIO, _T]: 返回值类型取决于输入：
            - 如果输入是 str 或 bytes，返回包含文件内容的 BytesIO 对象
            - 如果输入是其他类型（_T），直接返回原始输入对象
    
    环境变量：
        TIMEOUT (str): 网络请求的超时时间（秒），默认为 '300'。
                       设置为 0 或负数则不设置超时限制
    
    使用示例：
        # 从 URL 加载文件
        file_io = load_file('https://example.com/data.bin')
        
        # 从本地文件加载
        file_io = load_file('/path/to/local/file.txt')
        
        # 从用户目录加载
        file_io = load_file('~/documents/file.pdf')
        
        # 从 Base64 字符串加载（带前缀）
        file_io = load_file('data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAUA...')
        
        # 从 Base64 字符串加载（不带前缀）
        file_io = load_file('SGVsbG8gV29ybGQh')  # 解码后为 "Hello World!"
        
        # 从字节数据加载
        raw_bytes = b'\\x89PNG\\r\\n\\x1a\\n...'
        file_io = load_file(raw_bytes)
        
        # 传入其他类型对象（原样返回）
        from PIL import Image
        img = Image.open('test.png')
        result = load_file(img)  # 返回 img 本身
        
        # 设置网络请求超时（通过环境变量）
        import os
        os.environ['TIMEOUT'] = '60'  # 设置 60 秒超时
        file_io = load_file('https://example.com/large_file.zip')
    """
    # 初始化返回结果为输入参数本身
    res = path
    
    # 处理字符串类型的输入
    if isinstance(path, str):
        # 去除字符串首尾的空白字符
        path = path.strip()
        
        # 情况1：处理 HTTP/HTTPS URL
        if path.startswith('http'):
            # 初始化请求参数字典
            request_kwargs = {}
            
            # 从环境变量获取超时设置，默认 300 秒
            timeout = float(os.getenv('TIMEOUT', '300'))
            
            # 如果超时时间大于 0，则设置到请求参数中
            if timeout > 0:
                request_kwargs['timeout'] = timeout
            
            # 发送 GET 请求下载文件内容
            content = requests.get(path, **request_kwargs).content
            
            # 将下载的内容包装为 BytesIO 对象
            res = BytesIO(content)
        
        # 情况2：处理本地文件路径
        # 判断条件：文件路径存在，或者（不是 data: 前缀且长度小于等于 200 字符）
        # 后者用于处理可能的短路径字符串
        elif os.path.exists(path) or (not path.startswith('data:') and len(path) <= 200):
            # 将路径转换为绝对路径，并展开用户目录符号（~）
            path = os.path.abspath(os.path.expanduser(path))
            
            # 以二进制只读模式打开文件
            with open(path, 'rb') as f:
                # 读取文件全部内容并包装为 BytesIO 对象
                res = BytesIO(f.read())

        # 情况3：处理 Base64 编码字符串
        else:  # base64_str
            # 将路径字符串作为 Base64 数据
            data = path
            
            # 如果是带前缀的 Data URI 格式（如 'data:image/png;base64,...'）
            if data.startswith('data:'):
                # 使用正则表达式提取 Base64 数据部分
                # 格式：data:<MIME类型>;base64,<Base64数据>
                match_ = re.match(r'data:(.+?);base64,(.+)', data)
                
                # 断言匹配成功，否则格式不正确
                assert match_ is not None
                
                # 提取第二个捕获组（Base64 数据部分）
                data = match_.group(2)
            
            # 解码 Base64 字符串为二进制数据
            data = base64.b64decode(data)
            
            # 将解码后的二进制数据包装为 BytesIO 对象
            res = BytesIO(data)
    
    # 处理字节类型的输入
    elif isinstance(path, bytes):
        # 直接将字节数据包装为 BytesIO 对象
        res = BytesIO(path)
    
    # 返回处理结果（BytesIO 对象或原始输入）
    return res


def load_image(image: Union[str, bytes, Image.Image]) -> Image.Image:
    """
    加载图像文件并将其转换为 RGB 模式的 PIL Image 对象。
    
    该函数支持多种图像输入格式，包括本地文件路径、HTTP/HTTPS URL、Base64 编码字符串、
    字节数据以及已有的 PIL Image 对象。无论输入格式如何，最终都会返回一个 RGB 模式的图像对象。
    
    功能：
        从多种来源加载图像数据并统一转换为 RGB 模式的 PIL.Image.Image 对象
    
    参数：
        image (Union[str, bytes, Image.Image]): 图像输入，支持以下格式：
            - str: 可以是以下任意一种：
                * HTTP/HTTPS URL（如 'https://example.com/image.jpg'）
                * 本地文件路径（如 '/path/to/image.png' 或相对路径）
                * Base64 编码字符串（可带前缀 'data:image/png;base64,' 或不带前缀）
            - bytes: 图像的原始字节数据
            - Image.Image: 已经加载的 PIL Image 对象
    
    返回值：
        Image.Image: RGB 模式的 PIL Image 对象。如果输入图像不是 RGB 模式（如 RGBA、L 等），
                     会自动转换为 RGB 模式
    
    使用示例：
        # 从本地文件加载
        img = load_image('/path/to/image.png')
        
        # 从 URL 加载
        img = load_image('https://example.com/image.jpg')
        
        # 从 Base64 字符串加载
        img = load_image('data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAUA...')
        
        # 从字节数据加载
        with open('image.jpg', 'rb') as f:
            img_bytes = f.read()
        img = load_image(img_bytes)
        
        # 传入已有的 PIL Image 对象（会确保转换为 RGB 模式）
        from PIL import Image
        pil_img = Image.open('image.png')
        img = load_image(pil_img)
    """
    # 使用 load_file 函数加载图像数据，返回 BytesIO 对象或原始的 Image.Image 对象
    image = load_file(image)
    
    # 如果返回的是 BytesIO 对象（即从文件/URL/Base64/字节加载的数据），则打开为 PIL Image
    if isinstance(image, BytesIO):
        image = Image.open(image)
    
    # 检查图像的颜色模式，如果不是 RGB 模式，则转换为 RGB 模式
    # 这确保了所有输出图像都使用统一的 RGB 格式（3 通道），方便后续处理
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    return image


def load_batch(path_list: List[Union[str, None, Any, BytesIO]],
               load_func: Callable[[Any], _T] = load_image) -> List[_T]:
    """
    功能：
        批量加载文件列表，使用指定的加载函数处理每个文件。具体地，
        遍历路径列表，对每个非 None 的路径调用加载函数，并返回加载结果的列表。
        此方法支持泛型类型，可用于批量加载图像、音频、视频等多种文件类型。
    
    参数：
        path_list (List[Union[str, None, Any, BytesIO]]): 文件路径列表，支持以下类型：
            - str: 文件路径（URL、本地路径、Base64 编码等）
            - None: 会被忽略，不加载
            - BytesIO: 字节流对象
            - Any: 其他任意类型（如 PIL.Image 对象）
        load_func (Callable[[Any], _T]): 加载函数，默认为 load_image
            - 接受单个路径参数，返回加载后的对象
            - 可以是 load_image、load_audio、load_video 等函数
            - 支持自定义加载函数
    
    返回值：
        List[_T]: 加载结果的列表，类型由 load_func 的返回值决定
            - 如果 load_func 是 load_image，返回 List[Image.Image]
            - 如果 load_func 是 load_audio，返回 List[np.ndarray]
            - None 值会被跳过，不包含在返回列表中
    
    使用示例：
        # 示例1：批量加载图像（默认）
        image_paths = ['cat.jpg', 'dog.jpg', 'bird.jpg']
        images = load_batch(image_paths)
        # 返回：[Image.Image, Image.Image, Image.Image]
        
        # 示例2：批量加载图像（包含 None）
        image_paths = ['cat.jpg', None, 'dog.jpg']
        images = load_batch(image_paths)
        # 返回：[Image.Image, Image.Image]  # None 被跳过
        
        # 示例3：批量加载音频
        from functools import partial
        audio_paths = ['sound1.mp3', 'sound2.wav']
        audios = load_batch(
            audio_paths,
            load_func=partial(load_audio, sampling_rate=16000, return_sr=True)
        )
        # 返回：[(audio_array1, sr1), (audio_array2, sr2)]
        
        # 示例4：批量加载视频
        video_paths = ['video1.mp4', 'video2.avi']
        videos = load_batch(video_paths, load_func=load_video_llava)
        # 返回：[video_frames1, video_frames2]
        
        # 示例5：自定义加载函数
        def custom_loader(path):
            # 自定义加载逻辑
            return f"Loaded: {path}"
        
        paths = ['file1.txt', 'file2.txt']
        results = load_batch(paths, load_func=custom_loader)
        # 返回：['Loaded: file1.txt', 'Loaded: file2.txt']
        
        # 示例6：混合类型输入
        from PIL import Image
        mixed_inputs = [
            'image.jpg',           # 路径
            Image.open('cat.jpg'), # PIL.Image 对象
            None,                  # 跳过
            BytesIO(b'...')       # 字节流
        ]
        images = load_batch(mixed_inputs)
        # 返回：[Image.Image, Image.Image, Image.Image]
    """
    res = []
    # 验证 path_list 必须是列表或元组类型
    assert isinstance(path_list, (list, tuple)), f'path_list: {path_list}'
    # 遍历路径列表中的每个元素
    for path in path_list:
        # 跳过 None 值（忽略空路径）
        if path is None:  # ignore None
            continue
        # 使用指定的加载函数加载当前路径，并将结果添加到列表
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
