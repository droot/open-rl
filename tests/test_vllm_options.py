import os
import unittest
from unittest.mock import patch

from server import vllm_options
from server.vllm_options import gpu_memory_utilization, split_stop


class SplitStopTest(unittest.TestCase):
  def test_none(self):
    self.assertEqual(split_stop(None), (None, None))

  def test_bare_string(self):
    self.assertEqual(split_stop("\n\nUser:"), (["\n\nUser:"], None))

  def test_token_ids(self):
    self.assertEqual(split_stop([151645, 151643]), (None, [151645, 151643]))

  def test_strings(self):
    self.assertEqual(split_stop(["</s>", "\n\n"]), (["</s>", "\n\n"], None))

  def test_mixed(self):
    self.assertEqual(split_stop(["</s>", 151645]), (["</s>"], [151645]))

  def test_empty_sequence(self):
    self.assertEqual(split_stop([]), (None, None))

  def test_bool_is_not_a_token_id(self):
    self.assertEqual(split_stop([True]), (None, None))


if __name__ == "__main__":
  unittest.main()


class GpuMemoryUtilizationTest(unittest.TestCase):
  def test_explicit_fraction_wins(self) -> None:
    with patch.dict(os.environ, {"VLLM_GPU_MEMORY_UTILIZATION": "0.5", "OPEN_RL_ACCELERATOR_MEMORY": str(70 * 1024**3)}):
      self.assertEqual(gpu_memory_utilization(), 0.5)

  def test_placement_claim_does_not_cap_the_device(self) -> None:
    # The claim only decides where the worker lands; once placed it owns the
    # device while it holds it, so a 7 GiB claim still gets the 0.90 default.
    with patch.dict(os.environ, {"OPEN_RL_ACCELERATOR_MEMORY": str(7 * 1024**3)}, clear=True):
      self.assertEqual(gpu_memory_utilization(), 0.90)

  def test_without_a_budget_the_default_holds(self) -> None:
    with patch.dict(os.environ, {}, clear=True):
      self.assertEqual(gpu_memory_utilization(), 0.90)


class CudagraphCaptureSizesTest(unittest.TestCase):
  def test_powers_of_two_up_to_the_sequence_limit_by_default(self):
    with patch.dict(os.environ, {}, clear=False):
      os.environ.pop("VLLM_CUDAGRAPH_CAPTURE_SIZES", None)
      self.assertEqual(vllm_options.cudagraph_capture_sizes(64), [1, 2, 4, 8, 16, 32, 64])
      self.assertEqual(vllm_options.cudagraph_capture_sizes(256)[-1], 256)
      self.assertEqual(vllm_options.compilation_kwargs(8), {"compilation_config": {"cudagraph_capture_sizes": [1, 2, 4, 8]}})

  def test_an_explicit_list_is_sorted_and_capped(self):
    with patch.dict(os.environ, {"VLLM_CUDAGRAPH_CAPTURE_SIZES": "64, 8,16,512"}):
      self.assertEqual(vllm_options.cudagraph_capture_sizes(128), [8, 16, 64])

  def test_default_keeps_vllms_own_list(self):
    with patch.dict(os.environ, {"VLLM_CUDAGRAPH_CAPTURE_SIZES": "default"}):
      self.assertIsNone(vllm_options.cudagraph_capture_sizes(64))
      self.assertEqual(vllm_options.compilation_kwargs(64), {})
