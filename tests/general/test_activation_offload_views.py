# Copyright (c) ModelScope Contributors. All rights reserved.
import torch
import unittest

from swift.callbacks.activation_cpu_offload import _get_unique_tensor_key, get_activation_offload_context


class TestActivationOffloadViews(unittest.TestCase):

    def test_different_views_have_different_keys(self):
        tensor = torch.arange(12).reshape(3, 4)
        for view in [tensor.T, tensor[:2], tensor[:, :2], tensor.reshape(4, 3), tensor[1:]]:
            with self.subTest(shape=view.shape, stride=view.stride()):
                self.assertNotEqual(_get_unique_tensor_key(tensor), _get_unique_tensor_key(view))
        square = torch.arange(9).reshape(3, 3)
        self.assertNotEqual(_get_unique_tensor_key(square), _get_unique_tensor_key(square.T))

    def test_identical_views_can_still_share_a_copy(self):
        tensor = torch.arange(12).reshape(3, 4)
        self.assertEqual(_get_unique_tensor_key(tensor), _get_unique_tensor_key(tensor.view_as(tensor)))
        self.assertEqual(_get_unique_tensor_key(tensor[:, 1:]), _get_unique_tensor_key(tensor[:, 1:]))

    @unittest.skipUnless(torch.cuda.is_available(), 'requires CUDA activation transfers')
    def test_view_values_survive_offload(self):
        tensor = torch.arange(12, device='cuda').reshape(3, 4)
        context, commit = get_activation_offload_context()
        handler = context.offload_handler
        views = [tensor, tensor.T, tensor[:2], tensor[:, :2], tensor.reshape(4, 3), tensor.view_as(tensor)]
        tags = [handler.tensor_push(view) for view in views]
        commit(torch.ones((), device='cuda', requires_grad=True))
        self.assertEqual(len(handler.group_offload_mapping[0]), len(views) - 1)
        handler.on_group_commit_backward()
        for tag, expected in zip(tags, views):
            torch.testing.assert_close(handler.tensor_pop(tag), expected, check_stride=False)
        self.assertFalse(handler.tensor_tag_to_state)
        self.assertFalse(handler.group_offload_mapping)

    @unittest.skipUnless(torch.cuda.is_available(), 'requires CUDA activation transfers')
    def test_transposed_activations_preserve_backward(self):
        for dtype in [torch.float32, torch.bfloat16]:
            with self.subTest(dtype=dtype):
                context, commit = get_activation_offload_context()
                for _ in range(2):
                    tensor = torch.arange(1, 10, device='cuda', dtype=dtype).reshape(3, 3).requires_grad_()
                    reference = tensor.detach().clone().requires_grad_()
                    with context:
                        loss = tensor.square().sum() + tensor.T.square().sum()
                    loss = commit(loss)
                    loss.backward()
                    expected = reference.square().sum() + reference.T.square().sum()
                    expected.backward()
                    torch.testing.assert_close(loss, expected)
                    torch.testing.assert_close(tensor.grad, reference.grad)
                    self.assertFalse(context.offload_handler.tensor_tag_to_state)


if __name__ == '__main__':
    unittest.main()
