"""CPU-only tests for the bounded (windowed) DFLASH draft KV pool.

``--speculative-draft-window-size`` already windowed what the draft *attends*
(``DFlashWorkerV2.use_compact_draft_cache``); these cover the physical pool
behind it, which used to be a 1:1 shadow of the target's whole location space.
The draft pool is now an all-windowed ``SWAKVPool`` behind a
``SWATokenToKVPoolAllocator`` that borrows the target's own allocator as its full
side, so the draft's slots are assigned, reclaimed and re-mapped by the
machinery hybrid-SWA models already run -- the real ``SWAComponent`` actions the
HiCache restore and eviction paths emit are exercised here rather than an
approximation of them.

Real pools, allocators and component objects on the CPU device. The paged
``alloc_extend`` path runs Triton, so its chunked-prefill case is separated out
and skipped unless a CUDA device is present.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.managers.scheduler_components.invariant_checker import (
    SchedulerInvariantChecker,
)
from sglang.srt.managers.scheduler_components.pool_stats_observer import (
    SchedulerPoolStatsObserver,
)
from sglang.srt.mem_cache import kv_cache_builder
from sglang.srt.mem_cache.allocator.swa import SWATokenToKVPoolAllocator
from sglang.srt.mem_cache.allocator.token import TokenToKVPoolAllocator
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool
from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool
from sglang.srt.mem_cache.unified_cache.cache_action import (
    FreeComponentDeviceSlot,
    RebuildFullToSWAMapping,
)
from sglang.srt.mem_cache.unified_cache.components.swa_component import SWAComponent
from sglang.srt.speculative.base_spec_worker import BaseSpecWorker, HiCacheDraftMode
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")

DEVICE = "cpu"
DTYPE = torch.float32
HEADS, HEAD_DIM, DRAFT_LAYERS = 4, 8, 5
PAGE = 1  # the CPU allocator path; the paged (page 64) cases are the CUDA ones
FULL_LOCS = 1 << 14  # the target's location space
DRAFT_SLOTS = 1024  # the draft's bounded pool
WINDOW = 512


def _build(size_swa=DRAFT_SLOTS, size_full=FULL_LOCS):
    """The target's own pool/allocator, and a windowed draft pool bound to them."""
    target_pool = MHATokenToKVPool(
        size=size_full,
        page_size=PAGE,
        dtype=DTYPE,
        head_num=HEADS,
        head_dim=HEAD_DIM,
        layer_num=1,
        device=DEVICE,
        enable_memory_saver=False,
    )
    draft_pool = SWAKVPool(
        size=0,
        size_swa=size_swa,
        page_size=PAGE,
        dtype=DTYPE,
        head_num=HEADS,
        head_dim=HEAD_DIM,
        swa_attention_layer_ids=list(range(DRAFT_LAYERS)),
        full_attention_layer_ids=[],
        device=DEVICE,
    )
    target_allocator = TokenToKVPoolAllocator(
        size_full, DTYPE, DEVICE, target_pool, False
    )
    composite = SWATokenToKVPoolAllocator(
        size=size_full,
        size_swa=size_swa,
        page_size=PAGE,
        dtype=DTYPE,
        device=DEVICE,
        kvcache=draft_pool,
        need_sort=False,
        full_attn_allocator=target_allocator,
    )
    return target_pool, draft_pool, target_allocator, composite


def _component(composite, window=WINDOW, page_size=PAGE):
    """The real SWA component, built the way the tree builds it."""
    cache = SimpleNamespace(
        token_to_kv_pool_allocator=composite,
        req_to_token_pool=SimpleNamespace(),
    )
    params = CacheInitParams(
        disable=False,
        req_to_token_pool=cache.req_to_token_pool,
        token_to_kv_pool_allocator=composite,
        page_size=page_size,
        sliding_window_size=window,
    )
    return SWAComponent(cache, params)


class TestWindowedDraftPool(CustomTestCase):
    def test_scheduler_checks_target_and_draft_capacity_separately(self):
        # The real 64K-prefill failure: cached target rows exceed the entire
        # draft pool, while each pool independently conserves all of its slots.
        allocator = SimpleNamespace(
            size=24064,
            size_full=1368256,
            size_swa=24064,
            page_size=64,
            available_size=lambda: 768,
            full_available_size=lambda: 1302784,
            swa_available_size=lambda: 768,
        )
        req_pool = SimpleNamespace(
            schedulable_token_capacity=lambda size: size,
            mamba_allocator=SimpleNamespace(available_size=lambda: 24),
            mamba_pool=SimpleNamespace(size=36),
        )
        tree = MagicMock()
        tree.is_tree_cache.return_value = True
        tree.supports_mamba.return_value = True
        tree.full_evictable_size.return_value = 65472
        tree.swa_evictable_size.return_value = 23296
        tree.mamba_evictable_size.return_value = 12
        for name in (
            "full_protected_size",
            "swa_protected_size",
            "mamba_protected_size",
            "session_held_tokens",
            "session_held_full_tokens",
            "session_held_swa_tokens",
            "session_held_mamba_slots",
        ):
            getattr(tree, name).return_value = 0
        worker = SimpleNamespace(
            is_hybrid_swa=False,
            model_runner=SimpleNamespace(
                model_config=SimpleNamespace(),
                draft_swa_kv_pool=object(),
                draft_swa_window=2048,
                mtp_draft_device_pools=(),
            ),
            get_memory_pool=lambda: (req_pool, allocator),
        )
        memory = SimpleNamespace(
            disable_radix_cache=False,
            radix_eviction_policy="lru",
            enable_session_radix_cache=False,
        )
        parallel = SimpleNamespace(
            dcp_enabled=False,
            config=SimpleNamespace(enable_dp_attention=False),
        )
        with (
            patch.object(
                kv_cache_builder, "get_resolved_model_impl", return_value=None
            ),
            patch.object(kv_cache_builder, "uses_ssm_state", return_value=True),
            patch.object(kv_cache_builder, "is_deepseek_dsa", return_value=False),
            patch.object(
                kv_cache_builder,
                "resolve_decode_retraction_backup",
                return_value="cpu_tensor",
            ),
            patch.object(kv_cache_builder, "get_memory", return_value=memory),
            patch.object(kv_cache_builder, "get_parallel", return_value=parallel),
            patch.object(
                kv_cache_builder,
                "get_disagg",
                return_value=SimpleNamespace(
                    disaggregation_decode_enable_radix_cache=False
                ),
            ),
            patch.object(
                kv_cache_builder,
                "get_schedule",
                return_value=SimpleNamespace(chunked_prefill_size=4096),
            ),
            patch.object(kv_cache_builder, "init_mm_embedding_cache"),
            patch.object(
                kv_cache_builder, "create_tree_cache", return_value=tree
            ) as create,
        ):
            result = kv_cache_builder.build_kv_cache(
                server_args=SimpleNamespace(
                    enable_mamba_extra_buffer=lambda: True,
                    enable_mamba_extra_buffer_lazy=lambda: False,
                ),
                model_config=SimpleNamespace(hf_config=None, is_multimodal=False),
                tp_worker=worker,
                page_size=64,
                spec_algorithm=SpeculativeAlgorithm.DFLASH,
                attn_tp_cpu_group=None,
                tp_cpu_group=None,
                attn_cp_cpu_group=None,
                enable_metrics=False,
                enable_kv_cache_events=False,
                ps=SimpleNamespace(pp_rank=0, pp_size=1, tp_size=1, tp_rank=0),
                tp_group=None,
                pp_group=SimpleNamespace(cpu_group=None),
                enable_hierarchical_cache=True,
            )
        # Cache construction still sees the target's real model geometry.
        self.assertFalse(create.call_args.args[0].is_hybrid_swa)
        fields = dict(
            tree_cache=tree,
            token_to_kv_pool_allocator=allocator,
            req_to_token_pool=req_pool,
            is_hybrid_swa=result.is_hybrid_swa,
            is_hybrid_ssm=result.is_hybrid_ssm,
            full_tokens_per_layer=result.full_tokens_per_layer,
            swa_tokens_per_layer=result.swa_tokens_per_layer,
            max_total_num_tokens=allocator.size_full,
            get_last_batch=lambda: None,
            get_running_batch=lambda: None,
        )
        observer = SchedulerPoolStatsObserver(
            **fields,
            session_controller=SimpleNamespace(sessions={}),
            hisparse_coordinator=None,
            enable_hisparse=False,
        )
        checker = SchedulerInvariantChecker(
            **fields,
            disaggregation_mode=None,
            page_size=64,
            pool_stats_observer=observer,
        )
        with patch(
            "sglang.srt.managers.scheduler_components.invariant_checker.get_parallel",
            return_value=parallel,
        ):
            leaked, messages = checker._check_all_pools(observer.get_pool_stats())
            self.assertFalse(leaked, "\n".join(messages))
            self.assertEqual(len(messages), 3)
            # A real missing draft slot must still trigger the existing check.
            tree.swa_evictable_size.return_value -= 1
            leaked, _ = checker._check_all_pools(observer.get_pool_stats())
            self.assertTrue(leaked)

    def test_hicache_plan_leaves_windowed_draft_to_swa_component(self):
        _, draft_pool, _, _ = _build()
        target_runner = SimpleNamespace(spec_algorithm=SpeculativeAlgorithm.DFLASH)
        worker = SimpleNamespace(
            target_worker=SimpleNamespace(model_runner=target_runner),
            _draft_model_runners=lambda: (
                SimpleNamespace(token_to_kv_pool=draft_pool),
            ),
        )
        with patch(
            "sglang.srt.speculative.base_spec_worker.get_memory",
            return_value=SimpleNamespace(enable_hierarchical_cache=True),
        ):
            plan = BaseSpecWorker._build_hicache_draft_plan(worker)
        self.assertEqual(plan.mode, HiCacheDraftMode.NONE)
        self.assertEqual(plan.device_pools, ())
        self.assertEqual(target_runner.mtp_draft_device_pools, ())

    def test_target_pool_identity_survives_the_draft_binding(self):
        """The composite answers for the TARGET's pool, not the draft's."""
        target_pool, draft_pool, _, composite = _build()

        self.assertIs(composite.get_kvcache(), target_pool)
        self.assertIs(composite._windowed_kv_pool, draft_pool)
        self.assertIs(composite.full_attn_allocator.get_kvcache(), target_pool)
        # The windowed pool owns the table, and the allocator's translation goes
        # through it -- not through the target pool _kvcache, which has no SWA
        # sub-pool and keeps its own get_cpu_copy/load_cpu_copy semantics.
        self.assertIs(
            draft_pool.full_to_swa_index_mapping, composite.full_to_swa_index_mapping
        )
        self.assertFalse(hasattr(target_pool, "full_to_swa_index_mapping"))
        self.assertTrue(
            torch.equal(
                composite.translate_loc_from_full_to_swa(torch.tensor([7])),
                draft_pool.translate_loc_from_full_to_swa(torch.tensor([7])),
            )
        )
        self.assertEqual(composite.size_full, FULL_LOCS)
        self.assertEqual(composite.size_swa, DRAFT_SLOTS)

    def test_hybrid_mamba_strategy_still_serves_the_target(self):
        """Strategy selection sees the target's pool family + the draft window.

        The draft's bounded pool must not redirect the stack build: the FULL
        entry has to keep the target's real layers, buffers and QSA/Mamba tiers.
        """
        from sglang.srt.mem_cache.hybrid_cache.hybrid_pool_assembler import (
            _MambaStrategy,
            _MambaSwaStrategy,
        )
        from sglang.srt.mem_cache.memory_pool import HybridLinearKVPool
        from sglang.srt.mem_cache.unified_cache.components import ComponentType

        class _TargetPool(HybridLinearKVPool):
            def __init__(self):  # identity only; no buffers needed for selection
                pass

        components = {ComponentType.FULL, ComponentType.MAMBA, ComponentType.SWA}

        self.assertTrue(_MambaStrategy().matches(_TargetPool(), components))
        self.assertFalse(_MambaSwaStrategy().matches(_TargetPool(), components))
        # A real hybrid-SWA target still routes to the composite-pool strategy.
        _, draft_pool, _, _ = _build()
        self.assertTrue(_MambaSwaStrategy().matches(draft_pool, components))
        # A plain (non-hybrid, non-mamba) target with a windowed draft pool is
        # served by the catch-all strategy with the draft entry, not rejected.
        from sglang.srt.mem_cache.hybrid_cache.hybrid_pool_assembler import (
            _PlainKvStrategy,
        )

        plain_components = {ComponentType.FULL, ComponentType.SWA}

        plain_pool = MHATokenToKVPool(
            size=1024,
            page_size=PAGE,
            dtype=DTYPE,
            head_num=HEADS,
            head_dim=HEAD_DIM,
            layer_num=1,
            device=DEVICE,
            enable_memory_saver=False,
        )
        self.assertTrue(_PlainKvStrategy().matches(plain_pool, plain_components))
        self.assertTrue(_PlainKvStrategy().matches(plain_pool, {ComponentType.FULL}))

    def test_swa_component_governs_the_draft_window(self):
        """The existing component accepts the draft binding and takes its window."""
        _, _, _, composite = _build()
        component = _component(composite)

        self.assertIs(component.cache.token_to_kv_pool_allocator, composite)
        self.assertEqual(component.sliding_window_size, WINDOW)
        self.assertEqual(component.full_window_pages, WINDOW // PAGE)
        self.assertEqual(component.component_type.name, "SWA")

    def test_bounded_draft_side_stops_allocation_before_the_target(self):
        """Paired allocation is limited by the window bound, not target capacity."""
        _, _, target_allocator, composite = _build()

        held = []
        while True:
            locs = composite.alloc(256)
            if locs is None:
                break
            held.append(locs)
        allocated = sum(int(x.numel()) for x in held)

        self.assertEqual(allocated, DRAFT_SLOTS)
        self.assertEqual(composite.swa_available_size(), 0)
        self.assertGreater(composite.full_available_size(), FULL_LOCS // 2)
        self.assertLess(target_allocator.available_size(), FULL_LOCS)
        # Every held location resolves inside the small pool, not its own index.
        mapping = composite.full_to_swa_index_mapping[torch.cat(held)]
        # Slot 0 is the padded dummy row; live draft slots stay inside the pool.
        self.assertTrue((mapping > 0).all())
        self.assertTrue((mapping <= DRAFT_SLOTS).all())
        self.assertEqual(len(held), DRAFT_SLOTS // 256)

    def test_component_frees_and_remaps_slots_like_the_lifecycle_does(self):
        """The component's own eviction/restore actions recycle and re-point slots.

        ``FreeComponentDeviceSlot`` is what out-of-window eviction and device
        eviction emit; ``RebuildFullToSWAMapping`` is what an L2 (HiCache/NIXL)
        load-back emits after both sides allocated. A long run therefore stays
        inside the bounded pool and a returned prefix reads its new rows.
        """
        _, draft_pool, _, composite = _build()
        component = _component(composite)
        first = composite.alloc(DRAFT_SLOTS)
        self.assertIsNotNone(first)
        self.assertIsNone(composite.alloc(1))  # the pool is full

        component.apply_component_action(
            FreeComponentDeviceSlot(
                [first[:512]], component_type=component.component_type
            )
        )
        self.assertEqual(composite.swa_available_size(), 512)
        self.assertTrue(
            (draft_pool.translate_loc_from_full_to_swa(first[:512]) == 0).all()
        )

        # An L2 (HiCache/NIXL) load-back keeps the prefix's full locations and
        # allocates ONLY windowed slots for them, through the entry's
        # device_alloc_fn -- the windowed sub-allocator.
        restored = composite.swa_attn_allocator.alloc(512)
        self.assertIsNotNone(restored)
        self.assertTrue((restored > 0).all())
        component.apply_component_action(
            RebuildFullToSWAMapping(
                [first[:512]],
                [restored],
                component_type=component.component_type,
            )
        )
        remapped = draft_pool.translate_loc_from_full_to_swa(first[:512])
        self.assertTrue(torch.equal(remapped, restored))
        self.assertEqual(composite.swa_available_size(), 0)

        # The restored rows are actually addressable at the new slots.
        rows = torch.ones(512, HEADS, HEAD_DIM, dtype=DTYPE)
        draft_pool.set_kv_buffer(
            SimpleNamespace(layer_id=0), first[:512], rows.clone(), rows.clone()
        )
        self.assertTrue(
            torch.equal(draft_pool.get_key_buffer(0)[restored.long()], rows)
        )

    def test_windowed_writes_land_in_mapped_slots_only(self):
        """A write addressed by target locations touches its slot, not its loc."""
        _, draft_pool, _, composite = _build()
        locs = composite.alloc(128)
        rows = torch.arange(128 * HEADS * HEAD_DIM, dtype=DTYPE).view(
            128, HEADS, HEAD_DIM
        )
        for layer in range(DRAFT_LAYERS):
            draft_pool.set_kv_buffer(
                SimpleNamespace(layer_id=layer), locs, rows.clone(), rows.clone()
            )

        slots = draft_pool.translate_loc_from_full_to_swa(locs)
        self.assertEqual(slots.numel(), 128)
        self.assertTrue((slots > 0).all() and (slots <= DRAFT_SLOTS).all())
        kbuf = draft_pool.get_key_buffer(0)
        # The windowed sub-pool has its own rows (+ the reserved page); it is not
        # as wide as the target's location space.
        self.assertLess(kbuf.shape[0], FULL_LOCS)
        self.assertEqual(kbuf.shape[0], DRAFT_SLOTS + PAGE)
        self.assertTrue(torch.equal(kbuf[slots.long()], rows))
        self.assertEqual(int(kbuf.abs().sum().item()), int(rows.abs().sum().item()))

    def test_padding_location_keeps_the_sentinel(self):
        """The -1 last_loc the paged/commit paths pass maps to -1."""
        _, _, _, composite = _build()
        self.assertEqual(
            int(composite.translate_loc_from_full_to_swa(torch.tensor([-1]))[0]), -1
        )


@unittest.skipUnless(
    torch.cuda.is_available(), "paged alloc_extend runs Triton; GPU validation only"
)
class TestPagedChunkedPrefillBeyondCapacity(CustomTestCase):
    """The shipped page-64 geometry: chunked prefill past the physical draft pool."""

    PAGE = 64

    def _build_paged(self):
        from sglang.srt.mem_cache.allocator.paged import PagedTokenToKVPoolAllocator

        target_pool = MHATokenToKVPool(
            size=131072,  # covers the whole 98,304-token prompt + slack
            page_size=self.PAGE,
            dtype=DTYPE,
            head_num=HEADS,
            head_dim=HEAD_DIM,
            layer_num=1,
            device="cuda",
            enable_memory_saver=False,
        )
        draft_pool = SWAKVPool(
            size=0,
            size_swa=8192,
            page_size=self.PAGE,
            dtype=DTYPE,
            head_num=HEADS,
            head_dim=HEAD_DIM,
            swa_attention_layer_ids=list(range(DRAFT_LAYERS)),
            full_attention_layer_ids=[],
            device="cuda",
        )
        return draft_pool, SWATokenToKVPoolAllocator(
            size=131072,
            size_swa=8192,
            page_size=self.PAGE,
            dtype=DTYPE,
            device="cuda",
            kvcache=draft_pool,
            need_sort=False,
            full_attn_allocator=PagedTokenToKVPoolAllocator(
                131072, self.PAGE, DTYPE, "cuda", target_pool, False
            ),
        )

    def test_long_chunked_prefill_with_window_reclaim(self):
        """Reclaim happens BEFORE the next chunk, from the real location history.

        The tree's SWA component frees the locations that left the window on each
        commit; this drives its own eviction action on the locations the allocator
        actually handed out, page-aligned (a partial page would free live
        neighbours), advancing a frontier so no location is ever freed twice, and
        never past the tail so `last_loc` stays mapped for the next extend.
        """
        from sglang.srt.mem_cache.allocator.paged import PagedTokenToKVPoolAllocator

        draft_pool, composite = self._build_paged()
        window, chunk = 1024, 4096
        component = _component(composite, window=window, page_size=self.PAGE)
        # What must stay mapped across an allocation: the retained window plus
        # the chunk about to be written. Page-rounded, because freeing part of a
        # page would take live neighbours with it.
        keep = -(-window // self.PAGE) * self.PAGE
        self.assertLessEqual(keep + chunk, 8192)

        seq_len, locs_all, freed, freed_ranges = 0, None, 0, []
        for step in range(24):  # 98,304 tokens through an 8,192-slot pool
            if locs_all is not None:
                stale_upto = (locs_all.numel() - keep) // self.PAGE * self.PAGE
                if stale_upto > freed:
                    stale = locs_all[freed:stale_upto]
                    component.apply_component_action(
                        FreeComponentDeviceSlot(
                            [stale], component_type=component.component_type
                        )
                    )
                    freed_ranges.append((freed, stale_upto))
                    self.assertLess(stale_upto, locs_all.numel())  # tail stays live
                    freed = stale_upto

            prefix = torch.tensor([seq_len], dtype=torch.int64, device="cuda")
            lens = torch.tensor([seq_len + chunk], dtype=torch.int64, device="cuda")
            # last_loc is the previous chunk's last ALLOCATED location -- the
            # paged allocator extends from it, and its mapping must still stand.
            last = (
                torch.tensor([-1], dtype=torch.int64, device="cuda")
                if locs_all is None
                else locs_all[-1:].to(torch.int64)
            )
            if locs_all is not None:
                self.assertGreater(
                    int(draft_pool.translate_loc_from_full_to_swa(last)[0]), 0
                )

            locs = composite.alloc_extend(
                prefix, prefix.cpu(), lens, lens.cpu(), last, chunk
            )
            self.assertIsNotNone(
                locs, f"prefill stalled after {step} chunks ({freed} slots freed)"
            )
            locs_all = locs if locs_all is None else torch.cat([locs_all, locs])
            seq_len += chunk
            self.assertLessEqual(composite.swa_available_size(), 8192)

        self.assertEqual(seq_len, 24 * chunk)
        self.assertIsInstance(
            composite.full_attn_allocator, PagedTokenToKVPoolAllocator
        )

        # No location is freed twice: the frontier slices are disjoint and sorted.
        self.assertGreater(len(freed_ranges), 1)
        for (_, prev_end), (next_start, _) in zip(freed_ranges, freed_ranges[1:]):
            self.assertEqual(prev_end, next_start)

        # The final assertion reads the ACTUAL last-window locations, not offsets:
        # the live tail is mapped into the pool, and the reclaimed prefix is not.
        live = draft_pool.translate_loc_from_full_to_swa(
            locs_all[-window:].to(torch.int64)
        )
        self.assertTrue((live > 0).all())
        self.assertTrue((live <= 8192).all())
        first_freed = locs_all[0]
        self.assertEqual(
            int(draft_pool.translate_loc_from_full_to_swa(first_freed.reshape(1))[0]),
            0,
        )
        self.assertGreater(composite.swa_available_size(), 0)


if __name__ == "__main__":
    unittest.main()
