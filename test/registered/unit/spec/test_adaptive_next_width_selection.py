"""CPU-only checks for the shipped Next fixed width configuration and for WHEN
a width change is allowed to reach the worker.

``configs/pennyroyal/adaptive-next.json`` is an ordinary file for the existing
``--speculative-adaptive-config`` flag -- no Penny-specific policy lives in the
runtime. It fixes the verify width by active concurrency: a single-request
batch (C1) runs W8 (steps 7 at topk=1, draft width = steps + 1) and every C>1
batch runs W4 (steps 3), while the global ``DEFAULT_ADAPTIVE_CONFIG`` and an
ordinary fixed-width launch stay unchanged. One candidate per tier means the
width is a lookup on the batch size and the acceptance EMA can never move it
(the 3.0-era C1 W4/W8 adaptation and its anti-oscillation hold live on only as
capabilities of the machinery, exercised below through explicit two-candidate
tables).

Three mechanics are pinned here:

* the shipped numbers, read back through the real loader/path, plus the launch
  width and the captured-bucket geometry a wide C1 needs. ``validate_candidates``
  still refuses a candidate above the launch width, the fixed launch-maximum
  draft-token bound follows the table, ``cuda_graph_bs_for_step`` gives the
  W8 state only the BS1 bucket, and with no BS1 bucket a C1 batch pads up
  into the C>=2 slot and stays W4 rather than selecting wide shapes nothing
  captured;
* a verify completion feeds the policy of the batch it came from and cannot
  reconfigure the worker. The width is applied by the next actual forward, from
  that forward's own batch size (``activate_step_by_batch``, which is also where
  the outgoing backends are drained before anything is repointed). A stale C1
  completion landing while a newer C4 batch is in flight must not hand that
  batch wide speculation -- the donor reference had exactly that, a pending C1
  steps=7 overriding the next C4 policy of steps=3.

* the C1 anti-oscillation hold ported from the donor controller: after a width
  switch a multi-candidate C1 slot re-decides only once `switch_grace_batches`
  completed feedback batches have passed, and a reversal of the previous switch
  doubles that hold up to `max_grace_batches`, while the acceptance EMA keeps
  learning through the hold and C>=2 routing stays immediate. A tier that omits
  the keys (every other configuration, including the defaults and the shipped
  fixed table) keeps the switch-on-every-decision policy.

All evidence is CPU-only (no CUDA on this host): real capture, replay, numerics
and any hardware/performance claim remain GPU-window work.
"""

import json
import os
import shutil
import tempfile
import unittest
from types import SimpleNamespace

from sglang.srt.speculative import adaptive_spec_params
from sglang.srt.speculative.adaptive_runtime_state import (
    AdaptiveController,
    SpecRuntimeState,
)
from sglang.srt.speculative.base_spec_worker import BaseSpecWorker
from sglang.srt.speculative.eagle_worker_v2 import EAGLEWorkerV2
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=8, suite="base-a-test-cpu")

# The shipped configuration, resolved from the repository layout and consumed by
# nothing but the existing loader -- these checks fail if the file moves, stops
# being valid JSON, or if the width policy drifts back into the runtime.
ADAPTIVE_NEXT_CONFIG = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        *[os.pardir] * 4,
        "configs",
        "pennyroyal",
        "adaptive-next.json",
    )
)

NEXT_CANDIDATE_STEPS = [7]  # shipped fixed C1 width: W8 verify window at topk=1
# A two-candidate table, for the hold and transition machinery the runtime
# still implements for explicit operator configurations.
WIDE_NARROW_CANDIDATE_STEPS = [3, 7]
FAST = {"ema_alpha": 1.0, "warmup_batches": 0, "update_interval": 1}
# The anti-oscillation hold the shipped C1 entry is required to carry: a 40
# completed-feedback-batch hold after a switch, doubled per direction reversal
# and capped at 80, so W4/W8 cannot ping-pong batch after batch.
C1_HOLD = {"switch_grace_batches": 40, "grace_backoff": 2.0, "max_grace_batches": 80}


def _state(steps):
    return SpecRuntimeState(
        speculative_num_steps=steps,
        speculative_num_draft_tokens=steps + 1,
        draft_attn_backend=None,
        cuda_graph_runner=None,
        target_attn_backend=SimpleNamespace(name=f"target-{steps}"),
        target_graph_runner=None,
        draft_extend_attn_backend=None,
        cuda_graph_runner_for_draft_extend=None,
    )


class _WorkerStub:
    """Enough AdaptiveSpecWorker to watch which width reaches the worker, and when."""

    def __init__(self, launch_steps):
        self.speculative_num_steps = launch_steps
        self.events = []
        self.capture_requests = {}

    def build_adaptive_runtime_state(
        self, speculative_num_steps, speculative_num_draft_tokens, cuda_graph_bs=None
    ):
        self.capture_requests[speculative_num_steps] = cuda_graph_bs
        return _state(speculative_num_steps)

    def apply_runtime_state(self, state):
        if state.speculative_num_steps == self.speculative_num_steps:
            return
        self.events.append(f"apply:{state.speculative_num_steps}")
        self.speculative_num_steps = state.speculative_num_steps


class _ConfigCase(CustomTestCase):
    def setUp(self):
        super().setUp()
        self._dir = tempfile.mkdtemp(prefix="adaptive-next-")

    def tearDown(self):
        shutil.rmtree(self._dir, ignore_errors=True)

    def _params(self, initial_steps, cfg_path):
        return adaptive_spec_params.AdaptiveSpeculativeParams(
            initial_steps=initial_steps, cfg_path=cfg_path
        )

    def _temp_config(self, cfg):
        """A same-shaped table with the EMA knobs pinned, for the transition runs."""
        path = os.path.join(self._dir, "cfg.json")
        with open(path, "w") as f:
            json.dump(cfg, f)
        return path

    def _controller(self, cfg, launch_steps=7, cuda_graph_bs=(1, 2, 4, 8)):
        config_path = self._temp_config(cfg) if isinstance(cfg, dict) else cfg
        worker = _WorkerStub(launch_steps)
        controller = AdaptiveController(worker, config_path=config_path)
        controller.init_states(cuda_graph_bs=list(cuda_graph_bs))
        return controller, worker


class TestShippedNextConfig(_ConfigCase):
    def test_the_shipped_file_is_plain_json_that_the_loader_accepts(self):
        with open(ADAPTIVE_NEXT_CONFIG) as f:
            shipped = json.load(f)
        self.assertEqual(shipped["1"]["candidate_steps"], NEXT_CANDIDATE_STEPS)
        self.assertEqual(shipped["2"]["candidate_steps"], [3])
        # The one non-numeric key is prose, not a batch-size tier.
        self.assertEqual(
            [k for k in shipped if k.isdigit()], ["1", "2"], "tiers: C1 and C>=2"
        )
        self.assertEqual(
            adaptive_spec_params.resolve_candidate_steps_from_config(
                ADAPTIVE_NEXT_CONFIG
            ),
            [3, 7],
        )

    def test_the_shipped_tiers_are_single_candidates(self):
        # Fixed by concurrency: one candidate per tier, no hysteresis or hold
        # keys needed or honoured, so the acceptance EMA can never move the
        # width and every slot keeps the plain switch-on-every-decision
        # defaults (a grace of zero changes nothing for a single candidate).
        with open(ADAPTIVE_NEXT_CONFIG) as f:
            shipped = json.load(f)
        self.assertEqual(shipped["1"], {"candidate_steps": NEXT_CANDIDATE_STEPS})
        self.assertEqual(shipped["2"], {"candidate_steps": [3]})
        params = self._params(7, ADAPTIVE_NEXT_CONFIG)
        self.assertEqual(params._slots[1].switch_grace_batches, 0)
        self.assertEqual(params._slots[2].switch_grace_batches, 0)

    def test_the_shipped_policy_never_moves_under_a_cycling_workload(self):
        # The workload whose acceptance cycles between rejecting and fully
        # accepting every 10 batches -- 400 completed feedback batches -- is
        # the shape that ping-ponged the 3.0-era C1 table (39 re-decisions on
        # the exact base, the reported 47-in-72-seconds / 1081-in-130-turns
        # problem). With one candidate per tier the fixed lookup cannot move:
        # zero width changes, C1 stays W8 and every concurrent batch stays W4.
        params = self._params(NEXT_CANDIDATE_STEPS[-1], ADAPTIVE_NEXT_CONFIG)
        stream = ([[7]] * 10 + [[0]] * 10) * 20
        trace = []
        last = params.get_steps_for_batch(1)
        for i, counts in enumerate(stream, start=1):
            params.on_verify_complete(counts, batch_size=1)
            steps = params.get_steps_for_batch(1)
            if steps != last:
                trace.append((i, steps))
                last = steps
        self.assertEqual(trace, [])
        self.assertEqual(params.get_steps_for_batch(1), 7)
        self.assertEqual(params.get_steps_for_batch(6), 3)

    def test_the_shipped_widths_are_w8_at_c1_and_w4_at_every_concurrency(self):
        params = self._params(NEXT_CANDIDATE_STEPS[-1], ADAPTIVE_NEXT_CONFIG)
        self.assertEqual(params._slots[1].candidate_steps, NEXT_CANDIDATE_STEPS)
        self.assertEqual(params._slots[2].candidate_steps, [3])
        # steps + 1 is the draft width, so C1 runs W8 and C>1 runs W4.
        self.assertEqual({s + 1 for s in params._slots[1].candidate_steps}, {8})
        self.assertEqual(params.get_steps_for_batch(1), 7)
        for batch_size in (2, 3, 4, 6, 8, 32, 256):
            self.assertEqual(params.get_steps_for_batch(batch_size), 3)

    def test_a_two_candidate_c1_table_damps_a_cycling_workload_without_freezing_it(
        self,
    ):
        # The hold machinery, kept for explicit operator tables: an explicit
        # two-candidate C1 entry with the shipped hold keys and the real
        # default EMA (alpha 0.2, warmup 10, interval 5) and W4/W8 thresholds,
        # driven by a workload whose acceptance cycles between rejecting and
        # fully accepting every 10 batches -- 400 completed feedback batches.
        # Without the hold the same stream re-decides into 39 width changes (a
        # flip per ~10-batch swing, the reported 47-in-72-seconds /
        # 1081-in-130-turns shape). With the hold it is 5, spaced by the
        # 40-batch hold and its 80-batch reversal cap, and both widths are
        # still selected.
        params = self._params(
            WIDE_NARROW_CANDIDATE_STEPS[-1],
            self._temp_config(
                {
                    "1": {"candidate_steps": WIDE_NARROW_CANDIDATE_STEPS, **C1_HOLD},
                    "2": {"candidate_steps": [3]},
                }
            ),
        )
        stream = ([[7]] * 10 + [[0]] * 10) * 20
        trace = []
        last = params.get_steps_for_batch(1)
        for i, counts in enumerate(stream, start=1):
            params.on_verify_complete(counts, batch_size=1)
            steps = params.get_steps_for_batch(1)
            if steps != last:
                trace.append((i, steps))
                last = steps
        self.assertEqual(trace, [(20, 3), (65, 7), (155, 3), (245, 7), (335, 3)])
        self.assertLessEqual(
            len(trace), len(stream) // C1_HOLD["switch_grace_batches"] + 1
        )
        self.assertEqual(
            {steps for _, steps in trace}, set(WIDE_NARROW_CANDIDATE_STEPS)
        )
        gaps = [b - a for (a, _), (b, _) in zip(trace, trace[1:])]
        self.assertTrue(all(gap >= C1_HOLD["switch_grace_batches"] for gap in gaps))

    def test_c1_feedback_never_disturbs_the_shipped_fixed_widths(self):
        params = self._params(7, ADAPTIVE_NEXT_CONFIG)
        # Feed the policy through the real path with legitimate state: one
        # accepted-draft count per request in the batch (C1 -> one entry),
        # full acceptance, mid acceptance, then rejection, past warmup (10)
        # and across several update intervals (5). One candidate per tier
        # makes the lookup a function of the batch size, so every decision
        # re-selects the fixed width: C1 stays W8 and every concurrent batch
        # stays W4.
        for counts in [[7]] * 10 + [[3]] * 10 + [[0]] * 10:
            self.assertIsNone(params.on_verify_complete(counts, batch_size=1))
            self.assertEqual(params.get_steps_for_batch(1), 7)
            self.assertEqual(params.get_steps_for_batch(4), 3)
        # The EMA itself did learn -- the feedback reached the policy; the
        # fixed candidates simply give it nowhere to move.
        self.assertLess(params._slots[1].ema_accept_len, 1.0)
        self.assertEqual(params.get_steps_for_batch(6), 3)

    def test_the_wide_launch_allocation_follows_the_shipped_table(self):
        # The fixed launch-maximum buffers (QSA pending ring, request
        # reservation) are sized off the widest width the table can reach:
        # the shipped fixed table caps the verify window at W8's 8 tokens.
        from sglang.srt.runtime_context import _adaptive_draft_token_bound

        self.assertEqual(_adaptive_draft_token_bound(ADAPTIVE_NEXT_CONFIG), 8)
        self.assertEqual(_adaptive_draft_token_bound(None), 8)

    def test_wide_candidates_need_the_widest_launch_width(self):
        # The fixed C1 width (steps 7) exceeds a W4 launch, so such a launch
        # has to be told, not silently truncated at run time.
        worker = _WorkerStub(launch_steps=3)
        controller = AdaptiveController(worker, config_path=ADAPTIVE_NEXT_CONFIG)
        with self.assertRaises(ValueError) as ctx:
            controller.init_states()
        self.assertIn("7", str(ctx.exception))

    def test_global_defaults_and_the_fixed_width_union_are_unchanged(self):
        defaults = adaptive_spec_params.DEFAULT_ADAPTIVE_CONFIG
        self.assertEqual(
            {bs: e["candidate_steps"] for bs, e in defaults.items()},
            {"1": [1, 3, 7], "8": [0, 1, 3], "32": [0, 1], "64": [0]},
        )
        self.assertEqual(
            adaptive_spec_params.resolve_candidate_steps_from_config(), [0, 1, 3, 7]
        )
        self.assertEqual(defaults["1"]["candidate_steps"], [1, 3, 7])
        # A default-config launch still starts from the launch step it was given.
        params = self._params(3, None)
        self.assertEqual(params._bs_list, [1, 8, 32, 64])
        self.assertEqual(params.get_steps_for_batch(1), 3)
        self.assertEqual(
            params.candidate_steps, [0, 1, 3, 7], "the global table, not the Next one"
        )

    def test_the_penny_policy_stays_out_of_the_shared_runtime(self):
        # Repo rule: reusable code stays general, agreed configuration is an
        # external surface. The Next table lives only as the shipped JSON, so
        # the shared module neither names it nor special-cases it.
        leaked = [
            name
            for name in dir(adaptive_spec_params)
            if "penny" in name.lower() or "next" in name.lower()
        ]
        self.assertEqual(leaked, [], "Penny policy leaked into the shared runtime")
        with self.assertRaises(FileNotFoundError):
            adaptive_spec_params.resolve_candidate_steps_from_config("penny-next")


class TestCapturedBucketGeometry(_ConfigCase):
    def test_the_wide_states_capture_only_the_bucket_c1_can_reach(self):
        _, worker = self._controller(ADAPTIVE_NEXT_CONFIG)
        self.assertEqual(worker.capture_requests[7], [1])
        # Under the shipped fixed table C1 routes only to steps 7, so the W4
        # state captures for the concurrent buckets alone -- no BS1 bucket on
        # the narrow states.
        self.assertEqual(worker.capture_requests[3], [2, 4, 8])
        # The two-candidate table an operator can still configure explicitly
        # keeps every bucket on W4, BS1 included: there C1 can settle narrow.
        _, adaptive_worker = self._controller(
            {
                "1": {"candidate_steps": WIDE_NARROW_CANDIDATE_STEPS},
                "2": {"candidate_steps": [3]},
            }
        )
        self.assertEqual(adaptive_worker.capture_requests[3], [1, 2, 4, 8])
        self.assertEqual(adaptive_worker.capture_requests[7], [1])

    def test_a_configured_bs1_bucket_keeps_c1_on_the_wide_slot(self):
        params = self._params(7, ADAPTIVE_NEXT_CONFIG)
        params.set_cuda_graph_bs([1, 2, 4, 8])
        self.assertEqual(params.get_steps_for_batch(1), 7)
        self.assertEqual(params._route(1).candidate_steps, NEXT_CANDIDATE_STEPS)
        self.assertEqual(params.get_steps_for_batch(2), 3)

    def test_without_a_bs1_bucket_c1_pads_into_the_narrow_slot(self):
        # No BS1 graph: the wide states have no bucket to capture, and a C1
        # batch replays the BS2 graph, so it must route as C>=2 (W4) instead of
        # selecting wide shapes nothing captured.
        params = self._params(7, ADAPTIVE_NEXT_CONFIG)
        params.set_cuda_graph_bs([2, 4, 8])
        self.assertEqual(params.cuda_graph_bs_for_step(7), [])
        self.assertEqual(params.cuda_graph_bs_for_step(3), [2, 4, 8])
        self.assertEqual(params.get_steps_for_batch(1), 3)


class TestC1WidthHold(_ConfigCase):
    """The anti-oscillation hold machinery, seen through the controller.

    The shipped 3.1 policy is fixed-width, so the hold is exercised here
    through an explicit two-candidate table -- the machinery an operator can
    still configure. The native runtime switched width 47 times in 72 s of
    acceptance-only feedback and 1081 times over a 130-turn presentation: the
    controller re-decides every batch, so candidates the EMA cannot separate
    trade places forever. These walk the deterministic stream through the real
    path and pin the bound.
    """

    @staticmethod
    def _alternating(batches):
        return [[0] if i % 2 else [3] for i in range(1, batches + 1)]

    def _policy_switches(self, controller, feedback):
        """Width changes the C1 policy decides on, as (completed batch, steps)."""
        switches = []
        last = controller.params.get_steps_for_batch(1)
        for i, counts in enumerate(feedback, start=1):
            controller.on_verify_complete(counts, batch_size=1)
            steps = controller.params.get_steps_for_batch(1)
            if steps != last:
                switches.append((i, steps))
                last = steps
        return switches

    def test_an_alternating_c1_stream_holds_one_width_between_holds(self):
        controller, _ = self._controller(
            {
                "1": {
                    "candidate_steps": WIDE_NARROW_CANDIDATE_STEPS,
                    **FAST,
                    **C1_HOLD,
                },
                "2": {"candidate_steps": [3]},
            }
        )

        # Without the hold every one of these 200 batches flips the width. With
        # it: one switch, a 40-batch hold, the first reversal doubled to 80, the
        # next capped at 80 -- and the width still moves at batch 123, so the
        # hold slows the controller instead of freezing adaptation.
        switches = self._policy_switches(controller, self._alternating(200))
        self.assertEqual(switches, [(1, 3), (42, 7), (123, 3)])
        self.assertEqual(controller.params._slots[1]._grace, 80.0, "hold is capped")

    def test_the_c1_hold_never_delays_the_fixed_w4_routing_of_c2_batches(self):
        controller, worker = self._controller(
            {
                "1": {
                    "candidate_steps": WIDE_NARROW_CANDIDATE_STEPS,
                    **FAST,
                    **C1_HOLD,
                },
                "2": {"candidate_steps": [3]},
            }
        )

        for counts in self._alternating(60):
            controller.on_verify_complete(counts, batch_size=1)
            self.assertEqual(controller.params.get_steps_for_batch(2), 3)
            self.assertEqual(controller.params.get_steps_for_batch(8), 3)
            # Every concurrent C>=2 forward runs W4, immediately, whatever C1
            # is holding -- and the first one is what applies the narrow state.
            controller.activate_step_by_batch(4)
        self.assertEqual(worker.events, ["apply:3"])

    def test_a_slot_without_a_hold_flips_while_the_c1_slot_is_held(self):
        # Slot state is independent per tier: the identical reversal reaches a
        # C>=2 slot with no hold keys on its own next batch.
        controller, _ = self._controller(
            {
                "1": {
                    "candidate_steps": WIDE_NARROW_CANDIDATE_STEPS,
                    **FAST,
                    **C1_HOLD,
                },
                "2": {"candidate_steps": WIDE_NARROW_CANDIDATE_STEPS, **FAST},
            }
        )
        controller.on_verify_complete([0], batch_size=1)
        self.assertEqual(controller.params.get_steps_for_batch(1), 3)

        for _ in range(3):
            controller.on_verify_complete([0, 0], batch_size=2)
            self.assertEqual(controller.params.get_steps_for_batch(2), 3)
            self.assertEqual(controller.params._slots[2]._grace_until, 0)
            controller.on_verify_complete([3, 3], batch_size=2)
            self.assertEqual(controller.params.get_steps_for_batch(2), 7)

        # C1 spent none of that on re-decisions: its hold is still armed.
        self.assertEqual(controller.params.get_steps_for_batch(1), 3)
        self.assertEqual(controller.params._slots[1]._grace_until, 41)


class TestWidthAppliesAtTheForwardBoundary(_ConfigCase):
    def test_a_stale_c1_completion_cannot_widen_the_in_flight_c4_batch(self):
        controller, worker = self._controller(
            {
                "1": {"candidate_steps": WIDE_NARROW_CANDIDATE_STEPS, **FAST},
                "2": {"candidate_steps": [3]},
            }
        )
        self.assertEqual(worker.speculative_num_steps, 7)  # launched at the widest

        # C1 settles at W4 and the next actual batch -- a C4 one -- starts.
        controller.on_verify_complete([0], batch_size=1)
        controller.activate_step_by_batch(1)
        self.assertEqual(worker.events, ["apply:3"])
        controller.activate_step_by_batch(4)
        self.assertEqual(worker.speculative_num_steps, 3)

        # The OLDER C1 batch's result lands now, after the C4 forward started,
        # and its policy wants W8.
        before = list(worker.events)
        controller.on_verify_complete([3], batch_size=1)
        self.assertEqual(controller.params.get_steps_for_batch(1), 7)
        self.assertEqual(
            worker.events, before, "a completion callback reconfigured the worker"
        )
        self.assertEqual(worker.speculative_num_steps, 3)

        # The C4 batch keeps decoding (overlap: further of its own iterations).
        # Its width comes from its own batch size, not the pending C1 policy.
        controller.activate_step_by_batch(4)
        controller.on_verify_complete([7], batch_size=1)
        self.assertEqual(controller.params.get_steps_for_batch(1), 7)
        self.assertEqual(worker.events, before, "C4 was forced wide by a stale C1")
        self.assertEqual(worker.speculative_num_steps, 3)
        self.assertEqual(controller.params.get_steps_for_batch(4), 3)

        # The next ACTUAL C1 forward is where the wide state is applied.
        controller.activate_step_by_batch(1)
        self.assertEqual(worker.events, before + ["apply:7"])
        self.assertEqual(worker.speculative_num_steps, 7)

    def test_a_stale_c2_completion_cannot_narrow_the_in_flight_c1_batch(self):
        controller, worker = self._controller(
            {
                "1": {"candidate_steps": WIDE_NARROW_CANDIDATE_STEPS, **FAST},
                "2": {"candidate_steps": [3, 7], **FAST},
            }
        )
        # A C1 batch is replaying its W8 graph.
        controller.activate_step_by_batch(1)
        self.assertEqual(worker.speculative_num_steps, 7)

        # A delayed completion of an earlier C2 batch downshifts the C>=2 slot.
        before = list(worker.events)
        controller.on_verify_complete([0, 0], batch_size=2)
        self.assertEqual(controller.params.get_steps_for_batch(2), 3)
        self.assertEqual(worker.events, before, "a completion callback repointed")
        self.assertEqual(worker.speculative_num_steps, 7)

        # The wide C1 batch runs on; the narrow width lands at the next C2 forward.
        controller.activate_step_by_batch(1)
        self.assertEqual(worker.events, before)
        controller.activate_step_by_batch(2)
        self.assertEqual(worker.events, before + ["apply:3"])
        self.assertEqual(worker.speculative_num_steps, 3)

    def test_the_same_observations_give_the_same_transitions_on_every_rank(self):
        # TP2: each rank runs its own controller off the shared acceptance
        # observations, so two ranks must walk identical widths with no extra
        # collective and no per-rank timing input.
        cfg = {
            "1": {"candidate_steps": WIDE_NARROW_CANDIDATE_STEPS, **FAST},
            "2": {"candidate_steps": [3, 7], **FAST},
        }
        script = [
            (1, [3]),
            (1, [7]),
            (4, [0, 0, 0, 0]),
            (1, [0]),
            (4, [3, 3, 3, 3]),
        ]
        traces = []
        for _ in range(2):
            controller, worker = self._controller(cfg)
            trace = [worker.speculative_num_steps]
            for batch_size, counts in script:
                controller.activate_step_by_batch(batch_size)
                controller.on_verify_complete(counts, batch_size=batch_size)
                trace.append((batch_size, worker.speculative_num_steps))
                controller.activate_step_by_batch(batch_size)
                trace.append((batch_size, worker.speculative_num_steps))
            trace.append(worker.events)
            traces.append(trace)
        self.assertEqual(traces[0], traces[1])

    def test_reactivating_the_active_width_applies_nothing(self):
        controller, worker = self._controller(ADAPTIVE_NEXT_CONFIG)
        controller.activate_step_by_batch(1)  # the launch width is already active
        self.assertEqual(worker.events, [])
        controller.activate_step_by_batch(2)
        self.assertEqual(worker.events, ["apply:3"])
        for _ in range(3):
            controller.activate_step_by_batch(2)
        self.assertEqual(worker.events, ["apply:3"])


class TestFixedWidthAndNonAdaptiveUnaffected(_ConfigCase):
    def test_a_worker_without_a_controller_ignores_both_hooks(self):
        # Fixed W4 / non-adaptive: the result processor still calls the hook,
        # and nothing about the worker's width may move.
        class _Fixed:
            speculative_num_steps = 3
            speculative_num_draft_tokens = 4
            adaptive_controller = None

        fixed = _Fixed()
        EAGLEWorkerV2.on_verify_complete_cpu(fixed, [3, 3], batch_size=2)
        EAGLEWorkerV2.activate_step_by_batch(fixed, 2)
        BaseSpecWorker.on_verify_complete_cpu(fixed, [3], batch_size=1)
        BaseSpecWorker.activate_step_by_batch(fixed, 4)
        self.assertEqual(
            (fixed.speculative_num_steps, fixed.speculative_num_draft_tokens), (3, 4)
        )


if __name__ == "__main__":
    unittest.main()
