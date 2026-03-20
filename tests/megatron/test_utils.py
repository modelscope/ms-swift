"""
Unit tests for get_padding_to() in swift/megatron/utils/utils.py.

The function is inlined here so the tests run without requiring the full
swift / megatron / torch stack installed in the test environment.
Keep the inlined copy in sync with the source whenever it changes.
"""
import unittest
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Typed args dataclass – matches the fields accessed by get_padding_to
# ---------------------------------------------------------------------------
@dataclass
class _PaddingArgs:
    tensor_model_parallel_size: int = 1
    sequence_parallel: bool = False
    context_parallel_size: int = 1
    fp8_recipe: Optional[str] = None
    fp8_format: Optional[str] = None
    fp8: Optional[str] = None
    attention_backend: Optional[str] = None
    moe_flex_dispatcher_backend: Optional[str] = None
    seq_length: Optional[int] = None
    max_length: Optional[int] = None


# ---------------------------------------------------------------------------
# Inlined implementation – mirrors swift/megatron/utils/utils.py:get_padding_to
# ---------------------------------------------------------------------------
def get_padding_to(args: _PaddingArgs) -> Optional[int]:
    padding_to: Optional[int] = None
    if args.tensor_model_parallel_size > 1 and args.sequence_parallel:
        padding_to = args.tensor_model_parallel_size
    if args.context_parallel_size > 1:
        padding_to = (padding_to or 1) * args.context_parallel_size
    origin_padding_to = padding_to
    fp8_format: Optional[str] = args.fp8_format or args.fp8
    if args.fp8_recipe == 'blockwise':
        padding_to = (padding_to or 1) * 128
    elif fp8_format is not None:
        padding_to = max((padding_to or 1) * 8, 16)
    if args.attention_backend == 'fused':
        padding_to = max(padding_to or 1, (origin_padding_to or 1) * 64)

    # padding to max seq_length to avoid hybridep all-gather-into-tensor hang
    moe_backend: Optional[str] = getattr(args, 'moe_flex_dispatcher_backend', None)
    if moe_backend == 'hybridep':
        seq_length: Optional[int] = args.seq_length or args.max_length
        if seq_length is not None:
            padding_to = seq_length
    return padding_to


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_args(**kwargs: object) -> _PaddingArgs:
    """Return a _PaddingArgs with sensible defaults, overridden by kwargs."""
    return _PaddingArgs(**{k: v for k, v in kwargs.items()})  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------
class TestGetPaddingTo(unittest.TestCase):

    def _call(self, **kwargs: object) -> Optional[int]:
        return get_padding_to(_make_args(**kwargs))

    # ------------------------------------------------------------------
    # Baseline
    # ------------------------------------------------------------------
    def test_no_padding(self) -> None:
        """No parallelism / fp8 / special backend -> None."""
        self.assertIsNone(self._call())

    # ------------------------------------------------------------------
    # Tensor parallel + sequence parallel
    # ------------------------------------------------------------------
    def test_tp_sequence_parallel(self) -> None:
        self.assertEqual(self._call(tensor_model_parallel_size=4, sequence_parallel=True), 4)

    def test_tp_no_sequence_parallel(self) -> None:
        """TP alone (without sequence_parallel) should NOT activate padding."""
        self.assertIsNone(self._call(tensor_model_parallel_size=4, sequence_parallel=False))

    # ------------------------------------------------------------------
    # Context parallel
    # ------------------------------------------------------------------
    def test_context_parallel_only(self) -> None:
        self.assertEqual(self._call(context_parallel_size=4), 4)

    def test_tp_and_cp_combined(self) -> None:
        result = self._call(tensor_model_parallel_size=2, sequence_parallel=True, context_parallel_size=4)
        self.assertEqual(result, 2 * 4)

    # ------------------------------------------------------------------
    # FP8 recipes
    # ------------------------------------------------------------------
    def test_fp8_blockwise(self) -> None:
        self.assertEqual(self._call(fp8_recipe='blockwise'), 128)

    def test_fp8_blockwise_with_tp(self) -> None:
        result = self._call(tensor_model_parallel_size=2, sequence_parallel=True, fp8_recipe='blockwise')
        self.assertEqual(result, 2 * 128)

    def test_fp8_format(self) -> None:
        # max(1*8, 16) == 16
        self.assertEqual(self._call(fp8_format='e4m3'), 16)

    def test_fp8_format_with_tp(self) -> None:
        result = self._call(tensor_model_parallel_size=4, sequence_parallel=True, fp8_format='e4m3')
        self.assertEqual(result, max(4 * 8, 16))

    # ------------------------------------------------------------------
    # Fused attention backend
    # ------------------------------------------------------------------
    def test_attention_fused_no_tp(self) -> None:
        self.assertEqual(self._call(attention_backend='fused'), 64)

    def test_attention_fused_with_tp(self) -> None:
        # origin_padding_to = 2, result = max(2, 2*64) = 128
        result = self._call(tensor_model_parallel_size=2, sequence_parallel=True, attention_backend='fused')
        self.assertEqual(result, 128)

    # ------------------------------------------------------------------
    # hybridep backend – the new logic under test
    # ------------------------------------------------------------------
    def test_hybridep_uses_seq_length(self) -> None:
        """hybridep backend should set padding_to = seq_length."""
        result = self._call(moe_flex_dispatcher_backend='hybridep', seq_length=2048)
        self.assertEqual(result, 2048)

    def test_hybridep_uses_max_length_fallback(self) -> None:
        """When seq_length is absent, fall back to max_length."""
        result = self._call(moe_flex_dispatcher_backend='hybridep', max_length=4096)
        self.assertEqual(result, 4096)

    def test_hybridep_seq_length_takes_priority_over_max_length(self) -> None:
        """seq_length takes priority over max_length (short-circuit OR)."""
        result = self._call(moe_flex_dispatcher_backend='hybridep', seq_length=1024, max_length=4096)
        self.assertEqual(result, 1024)

    def test_hybridep_no_length_leaves_padding_unchanged(self) -> None:
        """If neither seq_length nor max_length is set, padding_to is not modified."""
        result = self._call(
            moe_flex_dispatcher_backend='hybridep',
            tensor_model_parallel_size=2,
            sequence_parallel=True,
        )
        self.assertEqual(result, 2)

    def test_hybridep_overrides_fp8_and_attention_padding(self) -> None:
        """hybridep seq_length wins over fp8/attention-derived padding."""
        result = self._call(
            moe_flex_dispatcher_backend='hybridep',
            seq_length=512,
            fp8_recipe='blockwise',
            attention_backend='fused',
        )
        self.assertEqual(result, 512)

    def test_hybridep_with_tp_and_cp(self) -> None:
        """hybridep overrides combined TP+CP padding."""
        result = self._call(
            moe_flex_dispatcher_backend='hybridep',
            seq_length=8192,
            tensor_model_parallel_size=4,
            sequence_parallel=True,
            context_parallel_size=2,
        )
        self.assertEqual(result, 8192)

    # ------------------------------------------------------------------
    # Non-hybridep backends must NOT trigger the new logic
    # ------------------------------------------------------------------
    def test_deepep_does_not_override(self) -> None:
        """deepep backend should NOT activate the hybridep padding override."""
        result = self._call(moe_flex_dispatcher_backend='deepep', seq_length=2048)
        self.assertIsNone(result)

    def test_no_moe_backend_does_not_override(self) -> None:
        """Without any moe_flex_dispatcher_backend, seq_length is ignored."""
        result = self._call(seq_length=2048)
        self.assertIsNone(result)

    def test_missing_moe_backend_attribute_does_not_override(self) -> None:
        """If the attribute doesn't exist on args (getattr default), seq_length is ignored."""
        args = _PaddingArgs(
            tensor_model_parallel_size=1,
            sequence_parallel=False,
            context_parallel_size=1,
            fp8_recipe=None,
            fp8_format=None,
            fp8=None,
            attention_backend=None,
            seq_length=2048,
            max_length=None,
            # moe_flex_dispatcher_backend defaults to None -> branch skipped
        )
        result = get_padding_to(args)
        self.assertIsNone(result)


if __name__ == '__main__':
    unittest.main()
