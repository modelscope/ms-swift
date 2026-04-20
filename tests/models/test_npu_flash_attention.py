#!/usr/bin/env python
# Copyright (c) ModelScope Contributors. All rights reserved.
"""
Core tests for NPU Flash Attention native integration in ms-swift

Key scenarios:
1. NPU available: Auto-register npu_flash_attention
2. Non-NPU: No error, graceful skip
3. Module import: Works on any machine
4. End-to-end: Full workflow with NPU FA
5. Missing torch_npu: No crash
"""

import os
import sys
import unittest
from unittest.mock import MagicMock, patch

# Disable auto-registration for clean testing
os.environ['SWIFT_DISABLE_NPU_FA'] = '1'


class TestNPUFlashAttentionCore(unittest.TestCase):
    """Core functionality tests"""

    def test_01_npu_available_auto_registers(self):
        """
        SCENARIO 1: NPU available → Auto-register npu_flash_attention
        
        When NPU is detected, register_npu_flash_attention() should:
        - Return True
        - Add 'npu_flash_attention' to ALL_ATTENTION_FUNCTIONS
        """
        from swift.model import npu_flash_attention as npu_fa
        
        # Reset state
        npu_fa._NPU_FA_REGISTERED = False
        
        # Mock NPU available
        with patch.object(npu_fa, 'is_torch_npu_available', return_value=True):
            with patch('transformers.integrations.npu_flash_attention.is_torch_npu_available', return_value=True):
                mock_attn_funcs = {}
                with patch('transformers.modeling_utils.ALL_ATTENTION_FUNCTIONS', mock_attn_funcs):
                    result = npu_fa.register_npu_flash_attention()
                    
                    self.assertTrue(result)
                    self.assertIn('npu_flash_attention', mock_attn_funcs)

    def test_02_non_npu_no_error(self):
        """
        SCENARIO 2: Non-NPU environment → No error, returns False
        
        On CUDA-only or CPU-only machines, registration should:
        - Return False
        - NOT raise any exception
        - NOT affect other functionality
        """
        from swift.model import npu_flash_attention as npu_fa
        
        npu_fa._NPU_FA_REGISTERED = False
        
        with patch.object(npu_fa, 'is_torch_npu_available', return_value=False):
            try:
                result = npu_fa.register_npu_flash_attention()
                self.assertFalse(result)
            except Exception as e:
                self.fail(f"Should not raise on non-NPU machine, got: {e}")

    def test_03_import_no_crash(self):
        """
        SCENARIO 3: Import module on any machine → No crash
        
        The module should import successfully regardless of hardware.
        """
        # Force re-import
        if 'swift.model.npu_flash_attention' in sys.modules:
            del sys.modules['swift.model.npu_flash_attention']
        
        try:
            from swift.model import npu_flash_attention
            self.assertTrue(hasattr(npu_flash_attention, 'register_npu_flash_attention'))
            self.assertTrue(hasattr(npu_flash_attention, 'npu_flash_attention_forward'))
        except (ImportError, AttributeError) as e:
            self.fail(f"Import should work on any machine, got: {e}")

    def test_04_npu_end_to_end_workflow(self):
        """
        SCENARIO 4: NPU environment complete workflow
        
        Simulate: import swift → auto-register → load model with npu_flash_attention
        """
        from swift.model import npu_flash_attention as npu_fa
        from swift.model.utils import AttnImpl
        
        # Clear disable flag for this test
        original_env = os.environ.pop('SWIFT_DISABLE_NPU_FA', None)
        
        try:
            npu_fa._NPU_FA_REGISTERED = False
            
            with patch.object(npu_fa, 'is_torch_npu_available', return_value=True):
                with patch('transformers.integrations.npu_flash_attention.is_torch_npu_available', return_value=True):
                    mock_attn_funcs = {}
                    with patch('transformers.modeling_utils.ALL_ATTENTION_FUNCTIONS', mock_attn_funcs):
                        # Step 1: Auto-register on import
                        result = npu_fa.auto_register_npu_flash_attention()
                        self.assertTrue(result)
                        
                        # Step 2: Verify available for model loading
                        self.assertIn('npu_flash_attention', mock_attn_funcs)
                        
                        # Step 3: AttnImpl recognizes it
                        self.assertTrue(AttnImpl.to_use_flash_attn('npu_flash_attention'))
                        
        finally:
            if original_env is not None:
                os.environ['SWIFT_DISABLE_NPU_FA'] = original_env

    def test_05_missing_torch_npu_no_crash(self):
        """
        SCENARIO 5: Missing torch_npu package → No crash
        
        When torch doesn't have npu attribute (no torch_npu installed),
        is_torch_npu_available should return False without ImportError.
        """
        from swift.model import npu_flash_attention as npu_fa
        
        # Mock torch without npu
        mock_torch = MagicMock()
        del mock_torch.npu  # Remove npu attribute
        
        with patch('swift.model.npu_flash_attention.torch', mock_torch):
            try:
                result = npu_fa.is_torch_npu_available()
                self.assertFalse(result)
            except (ImportError, AttributeError) as e:
                self.fail(f"Should handle missing torch_npu gracefully, got: {e}")


if __name__ == '__main__':
    unittest.main(verbosity=2)
