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
    build_draft_swa_entry,
    build_full_draft_pools,
    build_pool_entry,
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


class TestDraftHostFollowsTargetHost(CustomTestCase):
    """A windowed DFLASH draft's host tier tracks the TARGET host capacity.

    Every ordinary draft sidecar is sized by a host_to_device_ratio against the
    target host pool's logical slot count (see ``build_full_draft_pools``), so
    draft host slots follow the operator's --hicache-size budget and do not
    depend on how many draft GPU slots fit on the card. Windowing the draft's
    GPU pool must not shrink its host tier, and the draft must not take a bite
    out of the target's budget -- it is additional, as it always was.
    """

    TARGET_HOST_LOGICAL_SLOTS = 2_006_976

    def _build(self, draft_slots):
        from sglang.srt.mem_cache.cache_init_params import CacheInitParams
        from sglang.srt.mem_cache.hybrid_cache import hybrid_pool_assembler as hpa

        params = CacheInitParams(
            disable=False,
            req_to_token_pool=SimpleNamespace(),
            token_to_kv_pool_allocator=SimpleNamespace(
                swa_attn_allocator=SimpleNamespace(
                    alloc=lambda n: None, free=lambda i: None
                )
            ),
            page_size=64,
            draft_swa_device_pool=_DraftPool(24064 * 640, size=draft_slots),
            draft_swa_window=2048,
        )
        with (
            patch.object(
                hpa,
                "_build_mha_mla_host_pool",
                side_effect=lambda **kw: _FakeHostPool(kw["pool"].layer_num),
            ) as build_host_pool,
            patch.object(
                hpa,
                "build_kv_host_pool",
                side_effect=lambda **kw: _FakeHostPool(kw["kv_pool"].layer_num),
            ),
            patch.object(hpa, "_get_allocator_type", return_value="default"),
            patch.object(
                hpa,
                "get_memory",
                return_value=SimpleNamespace(
                    hicache_size=32,
                    hicache_ratio=2.0,
                    hicache_mem_layout="page_first",
                ),
            ),
        ):
            entry = build_draft_swa_entry(
                params=params,
                page_size=64,
                target_host_logical_size=self.TARGET_HOST_LOGICAL_SLOTS,
            )
        return entry, build_host_pool

    def test_draft_host_slots_track_target_host_capacity(self):
        # Shrinking only the draft's GPU capacity must leave its host tier at the
        # target's host logical capacity: ratio * draft_slots == 2,006,976 always.
        for draft_slots in (24064, 48128, 96256):
            with self.subTest(draft_slots=draft_slots):
                _, build_host_pool = self._build(draft_slots)
                self.assertIsNotNone(
                    build_host_pool.call_args,
                    "draft host tier must be sized by the ratio path, not a fixed GB share",
                )
                kwargs = build_host_pool.call_args.kwargs
                self.assertEqual(
                    kwargs["host_to_device_ratio"],
                    self.TARGET_HOST_LOGICAL_SLOTS / draft_slots,
                )
                self.assertEqual(
                    round(draft_slots * kwargs["host_to_device_ratio"]),
                    self.TARGET_HOST_LOGICAL_SLOTS,
                )
                self.assertEqual(kwargs["pool_label"], "draft_swa")

    def test_draft_entry_keeps_swa_semantics(self):
        entry, _ = self._build(24064)
        self.assertEqual(entry.name, PoolName.SWA)
        self.assertFalse(entry.is_primary_index_anchor)
        self.assertIsNotNone(entry.device_alloc_fn)
        self.assertIsNotNone(entry.device_free_fn)


class _DraftPool(_Pool):
    def __init__(self, kv_bytes, layer_num=5, size=1024):
        super().__init__(kv_bytes)
        self.layer_num = layer_num
        self.page_size = 1
        self.size = size

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


def _seeded_group(mamba=False) -> HostPoolGroup:
    """A real HostPoolGroup with a valid KV anchor, as every stack starts out."""
    entries = [
        build_pool_entry(
            name=PoolName.KV,
            host_pool=_FakeHostPool(12),
            device_pool=_Pool(16 * 10**9, layer_num=12),
            layer_mapping={0: 0},
            transfer_layer_num=12,
            is_anchor=True,
        )
    ]
    if mamba:
        entries.append(
            build_pool_entry(
                name=PoolName.MAMBA,
                host_pool=_FakeHostPool(1),
                device_pool=_Pool(4 * 10**9, layer_num=1),
                layer_mapping={0: 0},
                transfer_layer_num=12,
            )
        )
    return HostPoolGroup(entries)


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


class TestTargetBudgetAndRegistration(CustomTestCase):
    """The draft goes through the controller; the target keeps its whole budget.

    ``HostPoolGroup.add_entry`` alone skips both the per-pool host-release queue
    and the storage-backend registration the controller performs for entries
    added after construction -- on the NIXL profile that would leave the draft's
    windowed rows unbacked. And the draft's host tier is extra: the target's own
    pools must still be sized from the whole configured --hicache-size, exactly
    as before the draft pool was windowed.
    """

    GROUP_LOGICAL_SIZE = 1024  # the seeded anchor's host logical slots

    def _params(self, draft_slots=None):
        from sglang.srt.mem_cache.cache_init_params import CacheInitParams

        allocator = SimpleNamespace(
            swa_attn_allocator=SimpleNamespace(alloc=lambda n: None, free=lambda i: None)
        )
        req_to_token_pool = SimpleNamespace(
            mamba_map={}, mamba_pool=_Pool(4 * 10**9, layer_num=1)
        )
        return CacheInitParams(
            disable=False,
            req_to_token_pool=req_to_token_pool,
            token_to_kv_pool_allocator=allocator,
            page_size=1,
            draft_swa_device_pool=(
                None if draft_slots is None else _DraftPool(24064 * 640, size=draft_slots)
            ),
            draft_swa_window=2048 if draft_slots is not None else None,
        )

    def _drive(self, strategy, kvcache, stack_fn, mamba_group, draft_slots):
        from sglang.srt.mem_cache.hybrid_cache import hybrid_pool_assembler as hpa

        captured = {}
        draft_kwargs = {}
        # The SAME group and controller the stack builder returns are the ones
        # asserted on below, and registration is the real method, so the SWA entry
        # is genuinely in the group before component_host_pools reads it back.
        group = _seeded_group(mamba=mamba_group)
        storage = _FakeStorage()
        controller = _controller(group, storage)

        def fake_stack(**kwargs):
            captured.update(kwargs)
            return group, controller

        def fake_host_pool(**kwargs):
            draft_kwargs.update(kwargs)
            return _FakeHostPool(kwargs["pool"].layer_num)

        with (
            patch.object(hpa, stack_fn, side_effect=fake_stack),
            patch.object(hpa, "_build_mha_mla_host_pool", side_effect=fake_host_pool),
            patch.object(hpa, "_get_allocator_type", return_value="default"),
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
            result = strategy.build(
                cache=SimpleNamespace(evict_host=lambda n, ct: None, page_size=1),
                kvcache=kvcache,
                params=self._params(draft_slots=draft_slots),
                server_args=SimpleNamespace(hicache_host_memory_mode="cache"),
                load_cache_event=None,
                storage_backend="nixl",
            )
        return captured, draft_kwargs, group, controller, storage, result

    def _assert_registered(self, group, controller, storage, result):
        entry = group.get_entry(PoolName.SWA)
        self.assertEqual(entry.name, PoolName.SWA)
        self.assertIsNotNone(entry.device_alloc_fn)
        self.assertIsNotNone(entry.device_free_fn)
        self.assertFalse(entry.is_primary_index_anchor)
        self.assertIn(ComponentType.SWA, result.component_host_pools)
        self.assertIs(result.component_host_pools[ComponentType.SWA], entry.host_pool)
        self.assertEqual(storage.registered, [PoolName.SWA])
        self.assertIsInstance(
            controller.extra_host_mem_release_queues[PoolName.SWA], Queue
        )

    def test_plain_strategy_registers_the_entry_via_the_controller(self):
        from sglang.srt.mem_cache.hybrid_cache.hybrid_pool_assembler import (
            _PlainKvStrategy,
        )

        captured, draft_kwargs, group, controller, storage, result = self._drive(
            _PlainKvStrategy(),
            _Pool(16 * 10**9, layer_num=12),
            "build_kv_only_stack",
            mamba_group=False,
            draft_slots=24064,
        )

        # The target's own host budget is not cut down to make room for the draft.
        self.assertNotIn("host_budget_gb", captured)
        # The draft's host tier is sized off the target host pool's logical slots,
        # not off a byte share of the 32 GB cache the target just spent.
        self.assertEqual(
            draft_kwargs["host_to_device_ratio"],
            self.GROUP_LOGICAL_SIZE / 24064,
        )
        self.assertEqual(draft_kwargs["pool_label"], "draft_swa")
        self._assert_registered(group, controller, storage, result)

    def test_mamba_strategy_registers_the_entry_via_the_controller(self):
        from sglang.srt.mem_cache.hybrid_cache.hybrid_pool_assembler import (
            _MambaStrategy,
        )

        captured, draft_kwargs, group, controller, storage, result = self._drive(
            _MambaStrategy(),
            SimpleNamespace(
                full_attention_layer_id_mapping={},
                use_mla=False,
                full_kv_pool=_Pool(16 * 10**9, layer_num=12),
            ),
            "build_hybrid_mamba_stack",
            mamba_group=True,
            draft_slots=6016,
        )

        self.assertNotIn("host_budget_gb", captured)
        # A draft GPU pool 4x smaller still gets the target's full host capacity.
        self.assertEqual(
            draft_kwargs["host_to_device_ratio"],
            self.GROUP_LOGICAL_SIZE / 6016,
        )
        self.assertEqual(
            round(6016 * draft_kwargs["host_to_device_ratio"]), self.GROUP_LOGICAL_SIZE
        )
        self._assert_registered(group, controller, storage, result)

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

    def test_no_draft_pool_keeps_the_plain_stack_untouched(self):
        from sglang.srt.mem_cache.hybrid_cache.hybrid_pool_assembler import (
            _PlainKvStrategy,
        )

        captured, draft_kwargs, group, controller, storage, result = self._drive(
            _PlainKvStrategy(),
            _Pool(16 * 10**9, layer_num=12),
            "build_kv_only_stack",
            mamba_group=False,
            draft_slots=None,
        )

        self.assertNotIn("host_budget_gb", captured)
        self.assertEqual(draft_kwargs, {})
        self.assertNotIn(PoolName.SWA, group.entry_map)
        self.assertNotIn(ComponentType.SWA, result.component_host_pools)
        self.assertEqual(storage.registered, [])
        self.assertEqual(result.pools_desc, "KV")


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
