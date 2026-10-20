"""Unit tests for hybrid HiCache pool assembly."""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from sglang.srt.mem_cache.base_prefix_cache import EvictParams
from queue import Queue
from typing import Optional

from sglang.srt.mem_cache.hybrid_cache.hybrid_pool_assembler import (
    _evict_mamba_for_device_alloc,
    _evict_swa_for_device_alloc,
    _split_hicache_size,
    build_full_draft_pools,
    build_pool_entry,
    draft_swa_host_size,
    draft_swa_tier,
)
from sglang.srt.mem_cache.pool_host.group import HostPoolGroup
from sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller import (
    HybridCacheController,
)
from sglang.srt.mem_cache.hicache_storage import PoolName
from sglang.srt.mem_cache.memory_pool import HybridLinearKVPool
from sglang.srt.mem_cache.unified_cache.components import ComponentType
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class _Pool:
    def __init__(self, kv_bytes, layer_num=1):
        self._kv_bytes = kv_bytes
        self.layer_num = layer_num

    def get_kv_size_bytes(self):
        return self._kv_bytes


class TestDraftSwaHostTier(CustomTestCase):
    """The windowed DFLASH draft's host tier is a share of the HiCache budget.

    ``build_kv_host_pool`` treats an omitted host_size as the WHOLE configured
    --hicache-size, so the draft's bounded window pool must be given its own
    proportional share or it would claim the cache the target's KV rows are
    sized from (and double the operator's RAM setting).
    """

    def _size(self, hicache_size, peer_bytes, draft_bytes):
        with patch(
            "sglang.srt.mem_cache.hybrid_cache.hybrid_pool_assembler.get_memory",
            return_value=SimpleNamespace(hicache_size=hicache_size),
        ):
            return draft_swa_host_size(
                swa_kv_pool=_Pool(draft_bytes),
                peer_pools=(_Pool(peer_bytes),),
            )

    def test_draft_tier_is_its_own_share(self):
        # 32 GB total, draft device bytes are a quarter of the pool set.
        share = self._size(32, 3 * 10**9, 1 * 10**9)
        self.assertAlmostEqual(share, 8.0)
        self.assertLess(share, 32)

    def test_ratio_mode_leaves_sizing_to_the_pool(self):
        self.assertIsNone(self._size(0, 3 * 10**9, 1 * 10**9))

    def test_target_peers_keep_the_rest_of_the_budget(self):
        with patch(
            "sglang.srt.mem_cache.hybrid_cache.hybrid_pool_assembler.get_memory",
            return_value=SimpleNamespace(hicache_size=32),
        ):
            kv, mamba, draft = _split_hicache_size(
                32, (_Pool(3 * 10**9), _Pool(1 * 10**9), _Pool(4 * 10**9))
            )
        self.assertAlmostEqual(draft, 16.0)
        self.assertAlmostEqual(kv + mamba + draft, 32.0)


class _DraftPool(_Pool):
    def __init__(self, kv_bytes, layer_num=5):
        super().__init__(kv_bytes)
        self.layer_num = layer_num
        self.page_size = 1

    def get_size_per_token(self):
        return 1


class _FakeHostPool:
    def __init__(self, layer_num):
        self.layer_num = layer_num
        self.page_size = 1
        self.layout = "page_first"
        self.device = "cpu"
        self.size = 1024
        self.logical_size = 1024
        self.can_use_write_back_jit = False


class _FakeStorage:
    def __init__(self):
        self.registered = []

    def register_mem_host_pool_v2(self, host_pool, host_pool_name):
        self.registered.append(host_pool_name)


def _seeded_group() -> HostPoolGroup:
    """A real HostPoolGroup with a valid KV anchor, as every stack starts out."""
    return HostPoolGroup(
        [
            build_pool_entry(
                name=PoolName.KV,
                host_pool=_FakeHostPool(12),
                device_pool=_Pool(16 * 10**9, layer_num=12),
                layer_mapping={0: 0},
                transfer_layer_num=12,
                is_anchor=True,
            )
        ]
    )


def _controller(group, storage=None):
    """A real HybridCacheController with its registration state wired by hand.

    Only the constructor's threads/groups are skipped; `register_host_pool_entry`
    is the real method, so the release queue and storage registration are the
    production behavior under test.
    """
    controller = object.__new__(HybridCacheController)
    controller.mem_pool_host = group
    controller.enable_storage = True
    controller.storage_backend = storage if storage is not None else _FakeStorage()
    controller.extra_host_mem_release_queues = {}
    return controller


class TestDraftSwaRegistrationAndBudget(CustomTestCase):
    """The draft tier goes through the controller, and out of the same budget.

    ``HostPoolGroup.add_entry`` alone skips both the per-pool host-release queue
    and the storage-backend registration the controller performs for entries
    added after construction -- on the NIXL profile that would leave the draft's
    windowed rows unbacked. And the split has to come off the configured
    --hicache-size the target stack is about to spend entirely, or the assembled
    group exceeds the operator's setting.
    """

    def _params(self, draft_bytes=4 * 10**9):
        from sglang.srt.mem_cache.cache_init_params import CacheInitParams

        allocator = SimpleNamespace(
            swa_attn_allocator=SimpleNamespace(alloc=lambda n: None, free=lambda i: None)
        )
        return CacheInitParams(
            disable=False,
            req_to_token_pool=SimpleNamespace(),
            token_to_kv_pool_allocator=allocator,
            page_size=1,
            draft_swa_device_pool=_DraftPool(draft_bytes),
            draft_swa_window=2048,
        )

    def test_plain_strategy_registers_the_entry_via_the_controller(self):
        from sglang.srt.mem_cache.hybrid_cache import hybrid_pool_assembler as hpa
        from sglang.srt.mem_cache.hybrid_cache.hybrid_pool_assembler import (
            _PlainKvStrategy,
        )

        captured = {}
        # The SAME group and controller the stack builder returns are the ones
        # asserted on below, and registration is the real method, so the SWA entry
        # is genuinely in the group before component_host_pools reads it back.
        group = _seeded_group()
        storage = _FakeStorage()
        controller = _controller(group, storage)

        def fake_stack(**kwargs):
            captured.update(kwargs)
            return group, controller

        with (
            patch.object(hpa, "build_kv_only_stack", side_effect=fake_stack),
            patch.object(
                hpa,
                "build_kv_host_pool",
                side_effect=lambda **kw: _FakeHostPool(kw["kv_pool"].layer_num),
            ),
            patch.object(
                hpa,
                "get_memory",
                return_value=SimpleNamespace(
                    hicache_size=32,
                    hicache_ratio=2.0,
                    hicache_mem_layout="page_first",
                    hicache_io_backend="kernel",
                    hicache_write_policy="write_through",
                    hicache_storage_backend="default",
                    hicache_host_memory_mode="cache",
                ),
            ),
        ):
            result = _PlainKvStrategy().build(
                cache=SimpleNamespace(evict_host=lambda n, ct: None, page_size=1),
                kvcache=_Pool(16 * 10**9),
                params=self._params(),
                server_args=SimpleNamespace(hicache_host_memory_mode="cache"),
                load_cache_event=None,
                storage_backend="nixl",
            )

        # Draft device bytes are 4 GB of the 20 GB pool set, so a fifth of the
        # 32 GB cache: 6.4 GB for the draft window, 25.6 GB left for the target's
        # own tiers. Nothing is added on top of the operator's setting.
        self.assertAlmostEqual(captured["host_budget_gb"], 32.0 - 6.4)
        entry = group.get_entry(PoolName.SWA)
        self.assertEqual(entry.name, PoolName.SWA)
        self.assertIsNotNone(entry.device_alloc_fn)
        self.assertFalse(entry.is_primary_index_anchor)
        self.assertIn(ComponentType.SWA, result.component_host_pools)
        self.assertIs(result.component_host_pools[ComponentType.SWA], entry.host_pool)
        self.assertEqual(storage.registered, [PoolName.SWA])
        self.assertIsInstance(
            controller.extra_host_mem_release_queues[PoolName.SWA], Queue
        )

    def test_controller_registers_queue_and_storage(self):
        group = _seeded_group()
        storage = _FakeStorage()
        controller = _controller(group, storage)

        entry = build_pool_entry(
            name=PoolName.SWA,
            host_pool=_FakeHostPool(5),
            device_pool=_DraftPool(4 * 10**9),
            layer_mapping={i: i for i in range(5)},
            transfer_layer_num=5,
        )

        controller.register_host_pool_entry(entry)

        self.assertIs(group.get_entry(PoolName.SWA), entry)
        self.assertIsInstance(
            controller.extra_host_mem_release_queues[PoolName.SWA], Queue
        )
        self.assertEqual(storage.registered, [PoolName.SWA])

    def test_tier_is_one_split_of_the_configured_budget(self):
        with patch(
            "sglang.srt.mem_cache.hybrid_cache.hybrid_pool_assembler.get_memory",
            return_value=SimpleNamespace(hicache_size=32),
        ):
            share, peer_budget = draft_swa_tier(
                params=self._params(draft_bytes=4 * 10**9),
                peer_pools=(_Pool(16 * 10**9),),
            )
        # 4 GB draft of 20 GB device bytes = 1/5 of the configured 32 GB.
        self.assertAlmostEqual(share, 6.4)
        self.assertAlmostEqual(peer_budget, 25.6)
        # total stays at the operator's setting, it is not exceeded by the draft
        self.assertAlmostEqual(share + peer_budget, 32.0)
        with patch(
            "sglang.srt.mem_cache.hybrid_cache.hybrid_pool_assembler.get_memory",
            return_value=SimpleNamespace(hicache_size=0),
        ):
            self.assertEqual(
                draft_swa_tier(params=self._params(), peer_pools=()), (None, None)
            )

    def test_no_draft_pool_leaves_the_stack_untouched(self):
        from sglang.srt.mem_cache.cache_init_params import CacheInitParams

        params = CacheInitParams(
            disable=False,
            req_to_token_pool=SimpleNamespace(),
            token_to_kv_pool_allocator=SimpleNamespace(),
            page_size=1,
        )
        self.assertEqual(draft_swa_tier(params=params, peer_pools=(_Pool(1),)), (None, None))


class TestDeviceAllocEviction(CustomTestCase):
    def test_swa_evicts_only_allocation_shortfall(self):
        cache = MagicMock()
        cache.token_to_kv_pool_allocator.swa_available_size.return_value = 8

        _evict_swa_for_device_alloc(cache, required_size=10)

        cache.evict_for_alloc.assert_called_once_with(EvictParams(swa_num_tokens=2))
        cache.evict.assert_not_called()

    def test_mamba_evicts_only_allocation_shortfall(self):
        cache = MagicMock()
        allocator = cache.req_to_token_pool.mamba_allocator
        allocator.schedulable_available_size.return_value = 8

        _evict_mamba_for_device_alloc(cache, required_size=10)

        cache.evict_for_alloc.assert_called_once_with(EvictParams(mamba_num=2))
        cache.evict.assert_not_called()

    def test_sufficient_capacity_skips_eviction(self):
        cache = MagicMock()
        cache.token_to_kv_pool_allocator.swa_available_size.return_value = 10
        cache.req_to_token_pool.mamba_allocator.schedulable_available_size.return_value = (
            10
        )

        _evict_swa_for_device_alloc(cache, required_size=10)
        _evict_mamba_for_device_alloc(cache, required_size=10)

        cache.evict_for_alloc.assert_not_called()
        cache.evict.assert_not_called()


class TestSplitHicacheSize(CustomTestCase):
    def test_splits_total_budget_by_device_bytes(self):
        # scalar and (k, v) tuple return shapes both supported
        shares = _split_hicache_size(
            100, (_Pool(75 * 10**9), _Pool((15 * 10**9, 10 * 10**9)))
        )
        self.assertEqual(shares, (75.0, 25.0))  # proportional to device KV bytes
        self.assertEqual(sum(shares), 100)  # total budget preserved, not doubled

    def test_splits_total_budget_by_device_bytes_three_pools(self):
        # scalar and (k, v) tuple return shapes both supported
        shares = _split_hicache_size(
            100, (_Pool(55 * 10**9), _Pool((15 * 10**9, 10 * 10**9)), _Pool(20 * 10**9))
        )
        self.assertEqual(shares, (55.0, 25.0, 20.0))  # proportional to device KV bytes
        self.assertEqual(sum(shares), 100)  # total budget preserved, not doubled


class TestDraftSidecarPoolDispatch(CustomTestCase):
    def test_full_builder_unwraps_empty_hybrid_linear_pool(self):
        draft_kv_pool = object.__new__(HybridLinearKVPool)
        draft_kv_pool.full_kv_pool = SimpleNamespace(layer_num=0)

        specs, entries = build_full_draft_pools(
            draft_kv_pool=draft_kv_pool,
            tree_cache=None,
            server_args=None,
        )

        self.assertEqual(specs, [])
        self.assertEqual(entries, [])

    def test_full_builder_sizes_sidecar_for_anchor_logical_space(self):
        draft_kv_pool = SimpleNamespace(layer_num=1, size=800)
        draft_host_pool = SimpleNamespace(layer_num=1)
        tree_cache = SimpleNamespace(
            cache_controller=SimpleNamespace(
                mem_pool_host=SimpleNamespace(size=100, logical_size=800),
                page_size=512,
            )
        )
        # The layout comes from the published configuration.
        from sglang.srt.runtime_context import publish, reset_context
        from sglang.srt.server_args import ServerArgs

        server_args = ServerArgs(model_path="dummy", hicache_mem_layout="page_first")
        publish(server_args, role="scheduler")
        self.addCleanup(reset_context)

        with (
            patch(
                "sglang.srt.mem_cache.hybrid_cache.hybrid_pool_assembler."
                "_build_mha_mla_host_pool",
                return_value=draft_host_pool,
            ) as build_host_pool,
            patch(
                "sglang.srt.mem_cache.hybrid_cache.hybrid_pool_assembler."
                "_get_allocator_type",
                return_value="default",
            ),
        ):
            specs, entries = build_full_draft_pools(
                draft_kv_pool=draft_kv_pool,
                tree_cache=tree_cache,
                server_args=server_args,
            )

        self.assertEqual(build_host_pool.call_args.kwargs["host_to_device_ratio"], 1.0)
        self.assertEqual(len(specs), 1)
        self.assertIs(entries[0].host_pool, draft_host_pool)


if __name__ == "__main__":
    unittest.main()
