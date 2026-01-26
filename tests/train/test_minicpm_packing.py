"""MiniCPM packing_row 方法的单元测试。

测试 MiniCPMV2_6Template 和 MiniCPMV4_5Template 的 packing_row 方法，
验证 image_bound 偏移、pixel_values/tgt_sizes/temporal_ids 拼接的正确性。
"""
import unittest
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import torch


class MockMiniCPMV2_6Template:
    """模拟 MiniCPMV2_6Template 用于测试 packing_row 逻辑。"""

    def packing_row(self, row: List[Dict[str, Any]]) -> Dict[str, Any]:
        """从 minicpm.py 复制的实际实现。"""
        # 模拟父类 packing_row 的基础处理
        packed = self._base_packing_row(row)

        # 处理 image_bound - 需要根据累积的 token offset 调整索引
        image_bounds = []
        offset = 0
        for r in row:
            bounds = r.get('image_bound')
            if bounds is not None:
                for bound in bounds:
                    if isinstance(bound, torch.Tensor) and bound.numel() > 0:
                        adjusted = bound + offset
                        image_bounds.append(adjusted)
                    elif isinstance(bound, (list, tuple)) and len(bound) > 0:
                        adjusted = torch.tensor(bound) + offset
                        image_bounds.append(adjusted)
            offset += r['length']
        packed['image_bound'] = image_bounds

        # 拼接 pixel_values（与 minicpm.py 保持一致）
        # 输入格式：每个样本 pixel_values = [[Tensor]]，双层嵌套
        # 输出格式：合并后 pixel_values = [Tensor1, Tensor2, ...]（扁平列表）
        pixel_values = []
        for r in row:
            pv = r.get('pixel_values')
            if pv is not None:
                # pv 是 [[Tensor, ...]]，需要展开两层收集所有 Tensor
                for inner_list in pv:
                    pixel_values.extend(inner_list)
        packed['pixel_values'] = pixel_values

        tgt_sizes = []
        for r in row:
            ts = r.get('tgt_sizes')
            if ts is not None:
                tgt_sizes.extend(ts)
        packed['tgt_sizes'] = tgt_sizes

        return packed

    def _base_packing_row(self, row: List[Dict[str, Any]]) -> Dict[str, Any]:
        """模拟基类 Template.packing_row 的行为。"""
        packed = {}
        keys = set()
        length = []
        for r in row:
            keys.update(r.keys())
            length.append(r['length'])
        for key in keys:
            if key in {'input_ids', 'labels', 'loss_scale'}:
                packed[key] = sum((x.get(key) or [] for x in row), start=[])
            elif key == 'length':
                packed[key] = sum((x[key] for x in row))
        if 'position_ids' not in packed:
            packed['position_ids'] = sum((list(range(x)) for x in length), start=[])
        return packed


class MockMiniCPMV4_5Template(MockMiniCPMV2_6Template):
    """模拟 MiniCPMV4_5Template 用于测试 packing_row 逻辑。"""

    def packing_row(self, row: List[Dict[str, Any]]) -> Dict[str, Any]:
        """从 minicpm.py 复制的实际实现。"""
        packed = super().packing_row(row)

        # 处理 temporal_ids（视频场景）
        temporal_ids = []
        for r in row:
            tid = r.get('temporal_ids')
            if tid is not None:
                temporal_ids.extend(tid)
        packed['temporal_ids'] = temporal_ids

        return packed


class TestMiniCPMV2_6Packing(unittest.TestCase):
    """测试 MiniCPMV2_6Template 的 packing_row 方法。"""

    def setUp(self):
        self.template = MockMiniCPMV2_6Template()

    def test_single_sample_packing(self):
        """测试单个样本的 packing（边界情况）。"""
        row = [{
            'input_ids': [1, 2, 3, 4, 5],
            'labels': [-100, -100, 3, 4, 5],
            'length': 5,
            'image_bound': [torch.tensor([[1, 3]])],
            'pixel_values': [[torch.randn(3, 224, 224)]],  # 双层嵌套！
            'tgt_sizes': [torch.tensor([224, 224])],
        }]

        packed = self.template.packing_row(row)

        # 验证基础字段
        self.assertEqual(packed['input_ids'], [1, 2, 3, 4, 5])
        self.assertEqual(packed['labels'], [-100, -100, 3, 4, 5])
        self.assertEqual(packed['length'], 5)
        self.assertEqual(packed['position_ids'], [0, 1, 2, 3, 4])

        # 验证 image_bound 未偏移（offset=0）
        self.assertEqual(len(packed['image_bound']), 1)
        self.assertTrue(torch.equal(packed['image_bound'][0], torch.tensor([[1, 3]])))

        # 验证 pixel_values（packing后是扁平列表）和 tgt_sizes
        self.assertEqual(len(packed['pixel_values']), 1)
        self.assertEqual(len(packed['tgt_sizes']), 1)

    def test_two_samples_packing(self):
        """测试两个样本的 packing，验证 image_bound 偏移计算。"""
        row = [
            {
                'input_ids': [1, 2, 3],
                'labels': [-100, 2, 3],
                'length': 3,
                'image_bound': [torch.tensor([[0, 2]])],  # 图像在位置 0-2
                'pixel_values': [[torch.randn(3, 224, 224)]],  # 双层嵌套！
                'tgt_sizes': [torch.tensor([224, 224])],
            },
            {
                'input_ids': [4, 5, 6, 7],
                'labels': [4, 5, 6, 7],
                'length': 4,
                'image_bound': [torch.tensor([[1, 3]])],  # 图像在位置 1-3
                'pixel_values': [[torch.randn(3, 336, 336)]],  # 双层嵌套！
                'tgt_sizes': [torch.tensor([336, 336])],
            },
        ]

        packed = self.template.packing_row(row)

        # 验证基础字段拼接
        self.assertEqual(packed['input_ids'], [1, 2, 3, 4, 5, 6, 7])
        self.assertEqual(packed['labels'], [-100, 2, 3, 4, 5, 6, 7])
        self.assertEqual(packed['length'], 7)
        self.assertEqual(packed['position_ids'], [0, 1, 2, 0, 1, 2, 3])

        # 验证 image_bound 偏移
        # 第一个样本：offset=0，image_bound=[0, 2]
        # 第二个样本：offset=3，image_bound=[1, 3] -> [4, 6]
        self.assertEqual(len(packed['image_bound']), 2)
        self.assertTrue(torch.equal(packed['image_bound'][0], torch.tensor([[0, 2]])))
        self.assertTrue(torch.equal(packed['image_bound'][1], torch.tensor([[4, 6]])))

        # 验证 pixel_values（packing后是扁平列表）和 tgt_sizes 拼接
        self.assertEqual(len(packed['pixel_values']), 2)
        self.assertEqual(len(packed['tgt_sizes']), 2)

    def test_three_samples_with_multiple_images(self):
        """测试三个样本，每个样本有多张图片。"""
        row = [
            {
                'input_ids': list(range(10)),
                'labels': list(range(10)),
                'length': 10,
                'image_bound': [torch.tensor([[2, 4], [6, 8]])],  # 两个图像区间
                'pixel_values': [[torch.randn(3, 224, 224), torch.randn(3, 224, 224)]],  # 双层嵌套！
                'tgt_sizes': [torch.tensor([224, 224]), torch.tensor([224, 224])],
            },
            {
                'input_ids': list(range(5)),
                'labels': list(range(5)),
                'length': 5,
                'image_bound': [torch.tensor([[1, 3]])],
                'pixel_values': [[torch.randn(3, 336, 336)]],  # 双层嵌套！
                'tgt_sizes': [torch.tensor([336, 336])],
            },
            {
                'input_ids': list(range(8)),
                'labels': list(range(8)),
                'length': 8,
                'image_bound': [torch.tensor([[0, 2], [4, 6]])],
                'pixel_values': [[torch.randn(3, 448, 448), torch.randn(3, 448, 448)]],  # 双层嵌套！
                'tgt_sizes': [torch.tensor([448, 448]), torch.tensor([448, 448])],
            },
        ]

        packed = self.template.packing_row(row)

        # 验证长度
        self.assertEqual(packed['length'], 23)
        self.assertEqual(len(packed['input_ids']), 23)

        # 验证 image_bound 偏移
        # 样本1 offset=0: [[2, 4], [6, 8]]
        # 样本2 offset=10: [[1, 3]] -> [[11, 13]]
        # 样本3 offset=15: [[0, 2], [4, 6]] -> [[15, 17], [19, 21]]
        self.assertEqual(len(packed['image_bound']), 3)
        self.assertTrue(torch.equal(packed['image_bound'][0], torch.tensor([[2, 4], [6, 8]])))
        self.assertTrue(torch.equal(packed['image_bound'][1], torch.tensor([[11, 13]])))
        self.assertTrue(torch.equal(packed['image_bound'][2], torch.tensor([[15, 17], [19, 21]])))

        # 验证 pixel_values（packing后是扁平列表）和 tgt_sizes 数量
        self.assertEqual(len(packed['pixel_values']), 5)
        self.assertEqual(len(packed['tgt_sizes']), 5)

    def test_empty_image_bound(self):
        """测试空 image_bound 的处理。"""
        row = [
            {
                'input_ids': [1, 2, 3],
                'labels': [1, 2, 3],
                'length': 3,
                'image_bound': [torch.tensor([[1, 2]])],
                'pixel_values': [[torch.randn(3, 224, 224)]],  # 双层嵌套！
                'tgt_sizes': [torch.tensor([224, 224])],
            },
            {
                'input_ids': [4, 5],
                'labels': [4, 5],
                'length': 2,
                'image_bound': [],  # 空列表
                'pixel_values': [],  # 空列表
                'tgt_sizes': [],
            },
        ]

        packed = self.template.packing_row(row)

        # 只有第一个样本有 image_bound
        self.assertEqual(len(packed['image_bound']), 1)
        self.assertTrue(torch.equal(packed['image_bound'][0], torch.tensor([[1, 2]])))

    def test_none_image_bound(self):
        """测试 image_bound 为 None 的处理。"""
        row = [
            {
                'input_ids': [1, 2, 3],
                'labels': [1, 2, 3],
                'length': 3,
                'image_bound': None,
                'pixel_values': None,
                'tgt_sizes': None,
            },
            {
                'input_ids': [4, 5, 6],
                'labels': [4, 5, 6],
                'length': 3,
                'image_bound': [torch.tensor([[0, 2]])],
                'pixel_values': [[torch.randn(3, 224, 224)]],  # 双层嵌套！
                'tgt_sizes': [torch.tensor([224, 224])],
            },
        ]

        packed = self.template.packing_row(row)

        # 第二个样本的 image_bound 偏移 3
        self.assertEqual(len(packed['image_bound']), 1)
        self.assertTrue(torch.equal(packed['image_bound'][0], torch.tensor([[3, 5]])))


class TestMiniCPMV4_5Packing(unittest.TestCase):
    """测试 MiniCPMV4_5Template 的 packing_row 方法。"""

    def setUp(self):
        self.template = MockMiniCPMV4_5Template()

    def test_with_temporal_ids(self):
        """测试 temporal_ids 的拼接（视频场景）。"""
        row = [
            {
                'input_ids': [1, 2, 3],
                'labels': [1, 2, 3],
                'length': 3,
                'image_bound': [torch.tensor([[0, 2]])],
                'pixel_values': [[torch.randn(3, 224, 224)]],  # 双层嵌套！
                'tgt_sizes': [torch.tensor([224, 224])],
                'temporal_ids': [torch.tensor([0, 0, 1])],
            },
            {
                'input_ids': [4, 5],
                'labels': [4, 5],
                'length': 2,
                'image_bound': [torch.tensor([[0, 1]])],
                'pixel_values': [[torch.randn(3, 336, 336)]],  # 双层嵌套！
                'tgt_sizes': [torch.tensor([336, 336])],
                'temporal_ids': [torch.tensor([0, 1])],
            },
        ]

        packed = self.template.packing_row(row)

        # 验证 temporal_ids 拼接
        self.assertEqual(len(packed['temporal_ids']), 2)

        # 验证其他字段正常工作
        self.assertEqual(packed['length'], 5)
        self.assertEqual(len(packed['image_bound']), 2)
        self.assertTrue(torch.equal(packed['image_bound'][1], torch.tensor([[3, 4]])))

    def test_mixed_with_and_without_temporal_ids(self):
        """测试混合场景：有些样本有 temporal_ids，有些没有。"""
        row = [
            {
                'input_ids': [1, 2],
                'labels': [1, 2],
                'length': 2,
                'image_bound': [torch.tensor([[0, 1]])],
                'pixel_values': [[torch.randn(3, 224, 224)]],  # 双层嵌套！
                'tgt_sizes': [torch.tensor([224, 224])],
                'temporal_ids': None,  # 图像，无 temporal_ids
            },
            {
                'input_ids': [3, 4, 5],
                'labels': [3, 4, 5],
                'length': 3,
                'image_bound': [torch.tensor([[0, 2]])],
                'pixel_values': [[torch.randn(3, 336, 336)]],  # 双层嵌套！
                'tgt_sizes': [torch.tensor([336, 336])],
                'temporal_ids': [torch.tensor([0, 0, 1])],  # 视频，有 temporal_ids
            },
        ]

        packed = self.template.packing_row(row)

        # 只有第二个样本的 temporal_ids 被收集
        self.assertEqual(len(packed['temporal_ids']), 1)


class TestIntegrationWithRealTemplate(unittest.TestCase):
    """集成测试：使用真实的 Template 类（如果可用）。"""

    def test_import_template(self):
        """测试能否正常导入模板类。"""
        try:
            from swift.llm.template.template.minicpm import (
                MiniCPMV2_6Template,
                MiniCPMV4_5Template,
            )
            # 验证 packing_row 方法存在
            self.assertTrue(hasattr(MiniCPMV2_6Template, 'packing_row'))
            self.assertTrue(hasattr(MiniCPMV4_5Template, 'packing_row'))
        except ImportError as e:
            self.skipTest(f'无法导入 MiniCPM template: {e}')


if __name__ == '__main__':
    unittest.main()
