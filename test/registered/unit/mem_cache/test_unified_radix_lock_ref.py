"""CPU-only tests for UnifiedRadixCache request lock lifecycle."""

import unittest
from array import array
from types import SimpleNamespace
from unittest.mock import MagicMock

import torch

from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestUnifiedRadixLockRefScenarios(unittest.TestCase):
    def test_no_insert_without_last_node_skips_lock_release(self):
        cache = object.__new__(UnifiedRadixCache)
        cache.session = MagicMock()
        cache.session.try_cache_finished_req.return_value = False
        cache.disable = False
        cache.req_to_token_pool = MagicMock()
        cache.req_to_token_pool.req_to_token = torch.arange(8).reshape(1, 8)
        cache.token_to_kv_pool_allocator = MagicMock()
        cache._dec_req_lock = MagicMock()
        cache._components_tuple = ()
        cache.enable_session_radix_cache = False

        # Penny keeps req_pool_idx and cache_protected_len on the request.
        req = SimpleNamespace(
            origin_input_ids=array("q", [1, 2, 3]),
            output_ids=array("q"),
            req_pool_idx=0,
            cache_protected_len=0,
            last_node=None,
            swa_prefix_lock_released=False,
        )

        cache.cache_finished_req(req, is_insert=False, kv_len_to_handle=3)

        cache.token_to_kv_pool_allocator.free_segment.assert_called_once()
        freed, kwargs = cache.token_to_kv_pool_allocator.free_segment.call_args
        self.assertEqual(freed[0].tolist(), [0, 1, 2])
        self.assertEqual(kwargs, {"start_pos": 0})
        cache._dec_req_lock.assert_not_called()


if __name__ == "__main__":
    unittest.main()
