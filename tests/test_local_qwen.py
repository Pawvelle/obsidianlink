"""Regression tests for the local Qwen responder frame preparation."""

from __future__ import annotations

import unittest

import numpy as np

from obsidianlink.agents import LocalQwenResponder


class PrepareFrameRegressionTests(unittest.TestCase):
    """Frame must be C-contiguous before Qwen's image processor sees it.

    The 2026-07-31 Phase 3 VLM run
    ``runs/phase3-vlm-a0/20260731-211507/`` failed with::

        ValueError: At least one stride in the given numpy array is
        negative, and tensors with negative strides are not currently
        supported.

    The Qwen image processor calls ``torch.from_numpy(image).contiguous()``
    on the input array, and PIL-derived views can carry negative strides.
    The responder must normalise the frame before passing it downstream.
    This test class also guards the more general non-contiguous case
    (e.g. ``np.transpose`` reorders strides but does not negate them).
    """

    def test_regression_flipped_view_with_negative_strides_is_accepted(self) -> None:
        base = np.arange(4 * 4 * 3, dtype=np.uint8).reshape(4, 4, 3)
        flipped = base[:, ::-1, :]
        self.assertFalse(flipped.flags.c_contiguous)
        self.assertTrue(any(stride < 0 for stride in flipped.strides))
        prepared = LocalQwenResponder._prepare_frame(flipped)
        self.assertTrue(prepared.flags.c_contiguous)
        self.assertTrue(all(stride > 0 for stride in prepared.strides))
        np.testing.assert_array_equal(prepared, flipped)

    def test_regression_transposed_view_with_reordered_strides_is_accepted(self) -> None:
        # np.transpose swaps axis order without negating strides; the
        # resulting array has positive but reordered strides and is
        # therefore not C-contiguous even though no single stride is
        # negative. This still exercises the ascontiguousarray path.
        base = np.arange(4 * 4 * 3, dtype=np.uint8).reshape(4, 4, 3)
        transposed = np.transpose(base, (1, 0, 2))
        self.assertFalse(transposed.flags.c_contiguous)
        self.assertTrue(all(stride > 0 for stride in transposed.strides))
        self.assertNotEqual(transposed.strides, base.strides)
        prepared = LocalQwenResponder._prepare_frame(transposed)
        self.assertTrue(prepared.flags.c_contiguous)
        np.testing.assert_array_equal(prepared, transposed)

    def test_already_contiguous_frame_is_returned_unchanged(self) -> None:
        frame = np.zeros((2, 2, 3), dtype=np.uint8)
        self.assertTrue(frame.flags.c_contiguous)
        prepared = LocalQwenResponder._prepare_frame(frame)
        self.assertIs(prepared, frame)

    def test_non_ndarray_frame_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "RGB numpy frame"):
            LocalQwenResponder._prepare_frame(object())


if __name__ == "__main__":
    unittest.main()
