"""A request that already holds a req_to_token row is not counted twice at admission.

Port of sgl-project/sglang#44401 (upstream head 99e79ef77e7e6b8b345af5206f05104cd34115b0,
merged as f9bdb74a5ae9e0ae6cf37237d54af0086c2d55c9). The adder is a pass-through fake so
the row-accounting gate is the only thing deciding admission; everything else (the real
Scheduler._get_new_batch_prefill_raw loop, the real runtime context) is the shipped source.

This checkout keeps the row index on the Req itself (Req.req_pool_idx) rather than
upstream's Req.kv.holds_kv, and PrefillAdder is constructed directly rather than through
a class seam, so the fixture patches sglang.srt.managers.scheduler.PrefillAdder.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt import runtime_context as rc
from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.managers.schedule_policy import AddReqResult
from sglang.srt.managers.scheduler import Scheduler

register_cpu_ci(est_time=3, suite="base-a-test-cpu")

NUM_ROWS = 4


class _Req:
    def __init__(self, holds_row):
        # In this fork a request holds a req_to_token row iff req_pool_idx is set
        # (ReqToTokenPool.alloc reuses it instead of taking a free slot).
        self.req_pool_idx = 7 if holds_row else None
        self.inflight_middle_chunks = 0
        self.mamba_pool_idx = None
        self.session = None
        self.lora_id = None

    def finished(self):
        return False

    def init_next_round_input(self, tree_cache=None):
        pass


class _Adder:
    """Admits whatever the gate lets through; the final chunk joins first."""

    def __init__(self, *args, **kwargs):
        self.can_run_list = []
        self.preempt_list = []
        self.new_chunked_req = None
        self.rem_chunk_tokens = 0

    def add_chunked_req(self, req):
        self.can_run_list.append(req)
        return None

    def add_one_req(self, req, has_chunked_req, truncation_align_size):
        self.can_run_list.append(req)
        return AddReqResult.CONTINUE


class _BatchBuilt(Exception):
    pass


class TestHeldRowAdmission(CustomTestCase):
    def setUp(self):
        rc.reset_context()
        rc.publish(ServerArgs(model_path="dummy"), role="test")
        rc.get_context().override("test", pp_max_micro_batch_size=NUM_ROWS)

    def tearDown(self):
        rc.reset_context()

    def _scheduler(self, disaggregation_mode, running, final_chunk, waiting, num_free):
        s = Scheduler.__new__(Scheduler)
        s.grammar_manager = SimpleNamespace(has_waiting_grammars=lambda: False)
        s.enable_hierarchical_cache = False
        s.enable_priority_preemption = False
        s.is_hybrid_swa = False
        s.waiting_queue = waiting
        s.chunked_req = final_chunk
        s.min_free_slots_delayer = None
        s.policy = Mock()
        s.enable_dynamic_chunking = False
        s.chunked_prefill_size = 8192
        s.tp_worker = SimpleNamespace(
            model_runner=SimpleNamespace(attn_backend=SimpleNamespace())
        )
        s.page_size = 1
        s.tree_cache = Mock()
        s.token_to_kv_pool_allocator = None
        s.new_token_ratio_tracker = SimpleNamespace(current=0.0)
        s.max_prefill_tokens = 1 << 20
        s.is_mixed_chunk = False
        s.priority_scheduling_preemption_threshold = 0
        s.max_prefill_bs = 0
        s.max_running_requests = NUM_ROWS
        s.dllm_config = None
        s.enable_lora = False
        s.disaggregation_mode = disaggregation_mode
        s.enable_hicache_storage = False
        s.truncation_align_size = None
        s.model_config = None
        s.enable_overlap = False
        s.spec_algorithm = None
        s.server_args = Mock()
        # The running requests hold NUM_ROWS - num_free rows.
        s.req_to_token_pool = SimpleNamespace(available_size=lambda: num_free)
        return s

    def _admitted(self, disaggregation_mode, running, final_chunk, waiting, num_free):
        running_batch = ScheduleBatch(reqs=running)
        s = self._scheduler(disaggregation_mode, running, final_chunk, waiting, num_free)

        def capture(can_run_list, *args, **kwargs):
            raise _BatchBuilt(list(can_run_list))

        with (
            patch.object(ScheduleBatch, "init_new", side_effect=capture),
            patch("sglang.srt.managers.scheduler.PrefillAdder", _Adder),
            patch("sglang.srt.managers.scheduler.set_time_batch"),
        ):
            with self.assertRaises(_BatchBuilt) as built:
                s._get_new_batch_prefill_raw(
                    prefill_delayer_single_pass=None, running_batch=running_batch
                )
        return built.exception.args[0], running_batch.batch_is_full

    def _held_chunk_admits_the_waiting_request(self, disaggregation_mode):
        # 2 running rows + 1 free row, and a final chunk that already holds its
        # row: admitting it must not consume the last free row too.
        running = [_Req(holds_row=True), _Req(holds_row=True)]
        final_chunk = _Req(holds_row=True)
        waiting = _Req(holds_row=False)
        can_run_list, batch_is_full = self._admitted(
            disaggregation_mode, running, final_chunk, [waiting], num_free=1
        )
        self.assertEqual(can_run_list, [final_chunk, waiting])
        self.assertFalse(batch_is_full)

    def test_final_chunk_leaves_the_free_row_to_a_waiting_request(self):
        for mode in (DisaggregationMode.NULL, DisaggregationMode.PREFILL):
            with self.subTest(mode=mode):
                self._held_chunk_admits_the_waiting_request(mode)

    def test_no_held_row_is_still_bounded_by_the_free_rows(self):
        # Nothing in can_run_list holds a row, so accounting is unchanged and the
        # single free row admits exactly one waiting request.
        running = [_Req(holds_row=True), _Req(holds_row=True)]
        first, second = _Req(holds_row=False), _Req(holds_row=False)
        can_run_list, batch_is_full = self._admitted(
            DisaggregationMode.NULL, running, None, [first, second], num_free=1
        )
        self.assertEqual(can_run_list, [first])
        self.assertTrue(batch_is_full)

    def test_held_rows_restore_the_pool_term_only(self):
        s = self._scheduler(
            DisaggregationMode.NULL, [], None, [], num_free=1
        )
        # Pool term: the held rows are given back.
        self.assertEqual(s.get_num_allocatable_reqs(0, num_held_rows=2), 3)
        # Pipeline/max-running term is never inflated by held rows.
        self.assertEqual(s.get_num_allocatable_reqs(3, num_held_rows=9), 1)
        # No held rows -> exactly the pre-fix number.
        self.assertEqual(s.get_num_allocatable_reqs(0), 1)


if __name__ == "__main__":
    unittest.main()
