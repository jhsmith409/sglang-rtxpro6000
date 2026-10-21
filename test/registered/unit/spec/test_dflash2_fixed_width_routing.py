"""CPU-only checks for DFlash2 fixed batch-size width routing.

The approved policy (``configs/pennyroyal/adaptive-dflash2.json``) drafts a
**block6** verify window when a decode batch holds exactly one request and a
**block4** window from two requests up. It reuses the existing
``AdaptiveController``, its per-batch-size slots and its cached runtime-state
switching with the ``steps + 1`` convention already used at topk=1: BS1 lists
``candidate_steps=[5]`` and BS2 lists ``candidate_steps=[3]``.

This is concurrency routing, not acceptance-driven adaptation. A DFLASH draft
emits its whole block in one forward, so there is no per-step drafting loop for
an acceptance EMA to shorten; a tier with more than one candidate would be
general acceptance-adaptive DFlash, which is not implemented and must be refused
by name rather than silently averaged over.

Four mechanics are pinned, because each is a live bug class the moment the width
stops being launch-fixed:

* the shipped policy is data, read back through the real loader, and it is what
  sizes the one-time maximum-width provisioning;
* routing goes through the padded decode bucket, so a C3 batch (which replays
  the C4 graph) narrows, and without a BS1 bucket C1 narrows too rather than
  selecting wide shapes nothing captured;
* a prepared width owns every width-bound resource -- the draft model's
  convolution block size, the positional offsets, the verify spec metadata, the
  folded draft sampler and the ``[cap, width]`` block scratch -- and a switch
  moves them together with no recapture and no reallocation across repeated
  joins/drains (the pre-existing guards compared only batch capacity, so
  updating the scalar width alone left wide buffers feeding a narrow round);
* the width is applied at the batch's own decode-forward boundary, never from a
  verify-completion callback, so a stale C1 completion cannot widen a C4 batch
  already in flight -- and that batch's result keeps its own stride.

All evidence is CPU-only (no CUDA on this host): real graph capture/replay,
kernel numerics and any throughput claim remain GPU-window work.
"""

import json
import os
import shutil
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.arg_groups import overrides as overrides_mod
from sglang.srt.arg_groups.overrides import resolution_result
from sglang.srt.arg_groups.speculative_hook import handle_speculative_decoding
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.model_executor import cuda_graph_config as cgc
from sglang.srt.model_executor.cuda_graph_config import Backend, CudaGraphConfig
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.runtime_context import get_context, get_exec, get_spec
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative import adaptive_spec_params, dflash_info_v2
from sglang.srt.speculative import dflash_worker_v2 as dflash_module
from sglang.srt.speculative.adaptive_runtime_state import AdaptiveController
from sglang.srt.speculative.dflash_info import DFlashVerifyInput
from sglang.srt.speculative.dflash_worker_v2 import DFlashWorkerV2
from sglang.srt.speculative.draft_worker_common import make_draft_input_v2
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=14, suite="base-a-test-cpu")

# Resolved from the repository layout and consumed by nothing but the existing
# loader: these checks fail if the policy moves, stops being valid JSON, or if
# the approved widths drift back into the runtime.
DFLASH2_POLICY = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        *[os.pardir] * 4,
        "configs",
        "pennyroyal",
        "adaptive-dflash2.json",
    )
)

WIDE, NARROW = 6, 4  # verify block widths
WIDE_STEPS, NARROW_STEPS = WIDE - 1, NARROW - 1  # width == steps + 1 at topk=1
MAX_RUNNING_REQUESTS = 6  # the shipped Flash-Next/DFlash2 capacity
BUCKETS = [1, 2, 3, 4, 5, 6]  # that capacity's spec-decode capture list
VOCAB = 64
HIDDEN = 8

# Live worker attribute -> the field of the per-width state that owns it.
_WIDTH_ATTRS = {
    "_draft_block_ids_buf": "ids",
    "_draft_block_positions_buf": "positions",
    "_draft_block_tokens_buf": "tokens",
    "_draft_verify_out_cache_loc_buf": "cache_loc",
    "_draft_block_end_buf": "block_end",
    "_draft_seq_lens_cpu_buf": "seq_lens_cpu",
}
# The block-shaped scratch (row-major [cap, width]); the rest are per-request.
_BLOCK_SHAPE_ATTRS = [a for a in _WIDTH_ATTRS if not a.endswith(("end_buf", "cpu_buf"))]


def _published_config(backend=Backend.FULL, bs=None, max_bs=MAX_RUNNING_REQUESTS):
    return CudaGraphConfig.from_dict(
        {
            "decode": {
                "backend": backend,
                "max_bs": max_bs,
                "bs": list(BUCKETS if bs is None else bs),
            }
        }
    )


def _table(mapping):
    return {str(bs): {"candidate_steps": list(steps)} for bs, steps in mapping.items()}


def _capture_fixture(seen):
    """Stand-in for ``capture_decode_graph``: records the geometry each state
    captured under and hands back a distinct runner per call."""
    calls = []

    def capture(*, model_runner):
        cfg = get_exec().graph.cuda_graph_config
        calls.append(model_runner.role)
        seen.append(
            {
                "role": model_runner.role,
                "width": get_spec().speculative_num_draft_tokens,
                "bs": (
                    None
                    if cfg is None or cfg.decode.bs is None
                    else list(cfg.decode.bs)
                ),
                "workspace": model_runner.init_new_workspace,
                "hooks": len(model_runner.capture_tail_hooks),
            }
        )
        return SimpleNamespace(
            runner=SimpleNamespace(
                name=f"{model_runner.role}-graph-{len(calls)}",
                capture_bs=list(BUCKETS),
            ),
            memory_phase=f"{model_runner.role}_verify",
            memory_usage_gb=0.25,
            capture_time=0.5,
        )

    return capture


# ------------------------------------------------------- capability guard ----#


def _guard_reason(**fields):
    """``adaptive_unsupported_reason`` over a minimal resolved-config stand-in."""
    cfg = SimpleNamespace(
        **{
            "speculative_algorithm": "DFLASH",
            "speculative_eagle_topk": None,
            "speculative_adaptive_config": DFLASH2_POLICY,
            "enable_dp_attention": False,
            "enable_multi_layer_eagle": False,
            "enable_two_batch_overlap": False,
            "enable_pdmux": False,
            **fields,
        }
    )
    with (
        patch.object(overrides_mod, "resolving_view", lambda sa: cfg),
        patch.object(overrides_mod, "resolved_view", lambda sa: cfg),
    ):
        return adaptive_spec_params.adaptive_unsupported_reason(cfg)


class TestCapabilityGuard(CustomTestCase):
    def test_dflash_with_the_shipped_policy_is_supported(self):
        self.assertIsNone(_guard_reason())

    def test_dflash_single_tier_policy_is_supported(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "cfg.json")
            with open(path, "w") as f:
                json.dump(_table({4: [NARROW_STEPS]}), f)
            self.assertIsNone(_guard_reason(speculative_adaptive_config=path))

    def test_dflash_multi_candidate_tier_is_refused_by_name(self):
        # Two candidates in one tier is acceptance-adaptive DFlash, which needs a
        # per-step decision loop the draft does not have: refuse it, do not let
        # the EMA pick a width the profile was never qualified on.
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "cfg.json")
            with open(path, "w") as f:
                json.dump(_table({1: [NARROW_STEPS, WIDE_STEPS], 2: [NARROW_STEPS]}), f)
            reason = _guard_reason(speculative_adaptive_config=path)
        self.assertIsNotNone(reason)
        self.assertIn("one candidate", reason)
        self.assertIn("BS 1", reason)
        self.assertIn(str(WIDE_STEPS), reason)

    def test_the_global_default_table_is_not_a_dflash_policy(self):
        # A DFLASH launch that forgets --speculative-adaptive-config must not
        # quietly run acceptance-adaptive on the EAGLE-shaped defaults.
        reason = _guard_reason(speculative_adaptive_config=None)
        self.assertIsNotNone(reason)
        self.assertIn("DFLASH", reason)

    def test_a_missing_dflash_policy_file_is_refused_not_ignored(self):
        reason = _guard_reason(
            speculative_adaptive_config="/nonexistent/adaptive-dflash2.json"
        )
        self.assertIsNotNone(reason)
        self.assertIn("DFLASH", reason)

    def test_topologies_the_state_swap_does_not_handle_stay_refused(self):
        for field in (
            "enable_dp_attention",
            "enable_two_batch_overlap",
            "enable_pdmux",
            "enable_multi_layer_eagle",
        ):
            with self.subTest(field=field):
                self.assertIsNotNone(_guard_reason(**{field: True}))

    def test_topk_above_one_stays_refused_for_dflash(self):
        reason = _guard_reason(speculative_eagle_topk=2)
        self.assertIsNotNone(reason)
        self.assertIn("topk", reason)

    def test_dflash_routing_is_refused_under_a_full_prefill_graph(self):
        # A width switch repoints the target attention backend, and a FULL
        # prefill graph replays through whichever backend is live with the
        # metadata only its own capture filled in. The other prefill backends
        # re-enter the current backend, so they keep the routing.
        for backend, supported in (
            (Backend.FULL, False),
            (Backend.BREAKABLE, True),
            (Backend.TC_PIECEWISE, True),
            (Backend.DISABLED, True),
        ):
            with self.subTest(backend=backend):
                graph_config = _published_config()
                graph_config.prefill.backend = backend
                reason = _guard_reason(cuda_graph_config=graph_config)
                if supported:
                    self.assertIsNone(reason)
                else:
                    self.assertIsNotNone(reason)
                    self.assertIn("full", reason)
                    self.assertIn("DFLASH", reason)

    def test_the_full_prefill_graph_refusal_is_dflash_only(self):
        # A limitation of THIS width-routing integration, not a new global rule:
        # EAGLE/EAGLE3 adaptive keep whatever they did with a full prefill graph.
        graph_config = _published_config()
        graph_config.prefill.backend = Backend.FULL
        for algo in ("EAGLE", "EAGLE3"):
            with self.subTest(algo=algo):
                self.assertIsNone(
                    _guard_reason(
                        speculative_algorithm=algo, cuda_graph_config=graph_config
                    )
                )

    def test_eagle_and_other_algorithms_keep_their_existing_answers(self):
        self.assertIsNone(
            _guard_reason(
                speculative_algorithm="EAGLE", speculative_adaptive_config=None
            )
        )
        self.assertIsNone(
            _guard_reason(
                speculative_algorithm="EAGLE3",
                speculative_adaptive_config=None,
                speculative_eagle_topk=1,
            )
        )
        for algo in ("NGRAM", "DSPARK", "FROZEN_KV_MTP", "STANDALONE"):
            with self.subTest(algo=algo):
                reason = _guard_reason(
                    speculative_algorithm=algo, speculative_adaptive_config=None
                )
                self.assertIsNotNone(reason, algo)
                self.assertIn(algo, reason)


# ---------------------------------------------------------- policy content ----#


class TestShippedPolicyFile(CustomTestCase):
    def setUp(self):
        super().setUp()
        with open(DFLASH2_POLICY) as f:
            self.shipped = json.load(f)

    def test_the_shipped_file_routes_block6_at_c1_and_block4_from_c2(self):
        self.assertEqual([k for k in self.shipped if k.isdigit()], ["1", "2"])
        self.assertEqual(self.shipped["1"]["candidate_steps"], [WIDE_STEPS])
        self.assertEqual(self.shipped["2"]["candidate_steps"], [NARROW_STEPS])
        # steps + 1 is the verify width, so the two tiers are exactly W6 / W4.
        self.assertEqual(
            {
                steps + 1
                for steps in adaptive_spec_params.resolve_candidate_steps_from_config(
                    DFLASH2_POLICY
                )
            },
            {NARROW, WIDE},
        )

    def test_the_launch_maximum_follows_the_widest_routed_width(self):
        # The request reservation, the req_to_token headroom, the QSA pending
        # ring and the eager fixed-max buffers are sized once off this bound, so
        # a narrower routed width reuses what the graphs already bind.
        from sglang.srt.runtime_context import _adaptive_draft_token_bound

        self.assertEqual(_adaptive_draft_token_bound(DFLASH2_POLICY), WIDE)

    def test_the_global_defaults_are_untouched(self):
        defaults = adaptive_spec_params.DEFAULT_ADAPTIVE_CONFIG
        self.assertEqual(
            {bs: entry["candidate_steps"] for bs, entry in defaults.items()},
            {"1": [1, 3, 7], "8": [0, 1, 3], "32": [0, 1], "64": [0]},
        )
        self.assertEqual(
            adaptive_spec_params.resolve_candidate_steps_from_config(), [0, 1, 3, 7]
        )

    def test_the_policy_lives_only_in_the_shipped_file(self):
        # Repo rule: reusable code stays general, agreed configuration is an
        # external surface. The worker must neither name the tier keys nor
        # hard-code 6/4 -- it consumes whatever the loader returns.
        with open(dflash_module.__file__) as f:
            source = f.read()
        for marker in ('"1":', '"2":', "candidate_steps = [6", "= 6", "= 4"):
            self.assertNotIn(marker, source, marker)
        self.assertNotIn("6 if", source)


# ------------------------------------------------------------ routing only ----#


class TestRoutedWidthSelection(CustomTestCase):
    def setUp(self):
        super().setUp()
        self._dir = tempfile.mkdtemp(prefix="dflash2-routing-")

    def tearDown(self):
        shutil.rmtree(self._dir, ignore_errors=True)

    def _params(self, buckets=BUCKETS, initial_steps=WIDE_STEPS, cfg=DFLASH2_POLICY):
        params = adaptive_spec_params.AdaptiveSpeculativeParams(
            initial_steps=initial_steps, cfg_path=cfg
        )
        params.set_cuda_graph_bs(list(buckets))
        return params

    def test_batch_size_alone_picks_the_width(self):
        params = self._params()
        self.assertEqual(params.candidate_steps, [NARROW_STEPS, WIDE_STEPS])
        self.assertEqual(params.get_steps_for_batch(1), WIDE_STEPS)
        for batch_size in (2, 3, 4, 5, 6, 32, 256):
            self.assertEqual(params.get_steps_for_batch(batch_size), NARROW_STEPS)

    def test_a_padded_bucket_routes_on_the_padded_bucket(self):
        # A C3 batch replays the C4 graph; the padded bucket is in the C2+ tier,
        # so it must not select the C1 width whose graph was never captured.
        params = self._params(buckets=[1, 2, 4, 6])
        self.assertEqual(params.get_steps_for_batch(1), WIDE_STEPS)
        for batch_size in (2, 3, 4, 5, 6):
            self.assertEqual(params.get_steps_for_batch(batch_size), NARROW_STEPS)

    def test_without_a_bs1_bucket_c1_stays_narrow(self):
        params = self._params(buckets=[2, 3, 4, 5, 6])
        self.assertEqual(params.cuda_graph_bs_for_step(WIDE_STEPS), [])
        self.assertEqual(params.cuda_graph_bs_for_step(NARROW_STEPS), [2, 3, 4, 5, 6])
        self.assertEqual(params.get_steps_for_batch(1), NARROW_STEPS)

    def test_no_amount_of_acceptance_feedback_moves_a_routed_width(self):
        # Perfect acceptance at C1 and total rejection at C4, then the reverse,
        # far past warmup and the update interval: every tier is a single
        # candidate, so the EMA learns and the width does not move.
        params = self._params()
        for round_index in range(60):
            wide_feed = [[0]] if round_index % 2 else [list(range(WIDE))]
            narrow_feed = [list(range(NARROW)) * 4] if round_index % 2 else [[0] * 4]
            for counts in wide_feed:
                self.assertIsNone(params.on_verify_complete(counts, batch_size=1))
            for counts in narrow_feed:
                self.assertIsNone(params.on_verify_complete(counts, batch_size=4))
            self.assertEqual(params.get_steps_for_batch(1), WIDE_STEPS)
            self.assertEqual(params.get_steps_for_batch(4), NARROW_STEPS)

    def test_the_ema_still_learns_without_ever_switching(self):
        # The slots are fed (not stubbed out); only the candidate set is a
        # singleton, which is what makes the routing deterministic.
        params = self._params(buckets=[1, 2, 4, 8])
        for _ in range(30):
            params.on_verify_complete([0], batch_size=1)
        slot = params._slots[1]
        self.assertEqual(slot.current_steps, WIDE_STEPS)
        self.assertLess(slot.ema_accept_len, float(WIDE_STEPS - 1))
        self.assertGreater(slot._batch_count, 0)


# --------------------------------------------------- prepared width states ----#


class _SeqLens(torch.Tensor):
    """CPU decode lengths; ``record_stream`` is a CUDA notion, so no-op it."""

    def record_stream(self, stream):
        return None


class _DraftModel:
    def __init__(self, block_size):
        self.block_size = int(block_size)
        self.seen = []

    def set_block_size(self, block_size):
        self.seen.append(int(block_size))
        self.block_size = int(block_size)


class _FakeSampler:
    """Stand-in for the folded draft sampler, whose static outputs one graph
    bakes and whose block width must match the width that graph was captured at.
    """

    def __init__(self, block_size):
        self.block_size = int(block_size)
        self.staged = []

    def stage_sampling_params(self, *, bs, sampling_info):
        self.staged.append(int(bs))


def _runner_stub(role, pool, events):
    calls: list = []

    def _get_attention_backend(init_new_workspace=False):
        events.append(f"{role}-backend:{init_new_workspace}")
        return SimpleNamespace(name=f"{role}-backend-{len(events)}", pool=pool)

    def init_cuda_graphs(capture_decode_cuda_graph=True):
        # Mirrors ModelRunner.capture_cuda_graphs: everything but the decode
        # boundary reads the published config, and the captured runner becomes
        # the live one -- with the decode phase off there is no runner at all.
        with cgc.adaptive_launch_capture_scope():
            events.append(f"{role}-provisioning")
            if capture_decode_cuda_graph:
                captured = dflash_module.capture_decode_graph(model_runner=runner)
                runner.decode_cuda_graph_runner = captured.runner
            else:
                runner.decode_cuda_graph_runner = None

    def forward(fb):
        # Records the forward batch this width's draft ran with, then hands back
        # hidden states shaped for the block the batch was built with.
        calls.append(fb)
        n = int(fb.input_ids.numel())
        return SimpleNamespace(
            logits_output=LogitsProcessorOutput(
                next_token_logits=None,
                hidden_states=torch.zeros((n, HIDDEN), dtype=torch.bfloat16),
            ),
            can_run_graph=False,
        )

    runner = SimpleNamespace(
        role=role,
        device="cpu",
        forward=forward,
        forwarded=calls,
        attn_backend=SimpleNamespace(name=f"{role}-launch-backend", pool=pool),
        decode_cuda_graph_runner=SimpleNamespace(name=f"{role}-launch-graph"),
        capture_tail_hooks=[],
        init_new_workspace=False,
        token_to_kv_pool=pool,
        req_to_token_pool=SimpleNamespace(
            req_to_token=torch.zeros((8, 128), dtype=torch.int32),
            size=MAX_RUNNING_REQUESTS,
        ),
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(model_type="plain", get_text_config=lambda: None),
            linear_attn_registry_result=None,
        ),
        model=SimpleNamespace(
            lm_head=SimpleNamespace(weight=torch.zeros((1, 1))),
            get_input_embeddings=lambda: torch.nn.Embedding(VOCAB, HIDDEN),
        ),
        _get_attention_backend=_get_attention_backend,
        maybe_capture_gdn_recovery_graphs=lambda **kwargs: events.append(
            f"{role}-recovery"
        ),
        init_cuda_graphs=init_cuda_graphs,
    )
    return runner


class _Worker(DFlashWorkerV2):
    """Enough DFlashWorkerV2 to drive the real width state machine without a GPU.

    ``build_adaptive_runtime_state``, ``apply_runtime_state``,
    ``_override_worker_state``, ``_private_capture_scope`` and the decode
    boundary run for real; only the capture itself is a fixture, so a failure
    points at width ownership rather than at a missing device.
    """

    def __init__(
        self, launch_width, config_path, captures, target_graphs, real_sampler=False
    ):
        self.device = "cpu"
        self.gpu_id = 0
        self.ps = SimpleNamespace(tp_rank=0)
        self.shared_pool = SimpleNamespace(name="shared-pools")
        self.events = []
        self.captures = captures
        self.target_graphs = target_graphs
        self.model_runner = _runner_stub("target", self.shared_pool, self.events)
        self._target_worker = SimpleNamespace(
            model_runner=self.model_runner,
            forward_batch_generation=self._target_forward,
        )
        self.draft_model_runner = _runner_stub("draft", self.shared_pool, self.events)
        self._draft_worker = self.draft_model_runner
        self.draft_model = _DraftModel(launch_width)
        self.selector = None
        # False keeps the fixture's sampler stand-in; True runs the worker's REAL
        # sampler builder, which is where an unreachable width used to blow up.
        self._use_real_sampler = bool(real_sampler)
        self._warned_sampling_fallback = False
        self._draft_probs_buf = None
        self._logged_first_verify = False
        self._need_mamba_verify_commit = False
        self._additional_graph_memory_usage = {}
        self._additional_graph_time_usage = {}
        self._draft_sampler = None
        self._draft_greedy_gather_cap = 0
        self._draft_greedy_local_cap = 0
        self._draft_greedy_index_cap = 0
        self._draft_greedy_best_rank_buf = None
        self._draft_greedy_rank_index_buf = None
        self._draft_greedy_selected_ids_buf = None
        self._use_fused_kv_materialize = False
        self._fused_kv_helper = None
        self._use_triton_prepare_block = False
        self._use_triton_accept_bonus = False
        self._use_triton_compact_rebuild = False
        self.adaptive_controller = None
        self._dflash_widths = {}
        self._active_width = None
        self._selector_sample = None
        self.draft_window_size = None
        self.use_compact_draft_cache = False
        self._mask_token_id = 3
        self._noise_embed_scale = 1.0
        self.block_size = int(launch_width)
        self.speculative_num_draft_tokens = int(launch_width)
        # Mirrors the real __init__: the launch width's set is provisioned and
        # installed before anything can switch widths.
        self._apply_width_state(self._dflash_width(self.block_size))
        if config_path is not None:
            self.speculative_num_steps = int(launch_width) - 1
            self.adaptive_controller = AdaptiveController(self, config_path=config_path)

    def _maybe_build_draft_sampler(self):
        if self._use_real_sampler:
            return DFlashWorkerV2._maybe_build_draft_sampler(self)
        return _FakeSampler(self.block_size)

    def _greedy_sample_from_vocab_parallel_head(self, *, hidden_states, lm_head):
        return torch.arange(hidden_states.shape[0], dtype=torch.long)

    def _append_target_hidden_to_draft_kv_by_loc(self, **kwargs):
        return None

    def _target_forward(
        self, *, batch, forward_batch, is_verify, skip_attn_backend_init
    ):
        spec = forward_batch.spec_info
        self.target_graphs.append(
            {
                "width": spec.draft_token_num,
                "tokens": int(forward_batch.input_ids.numel()),
                "positions": int(forward_batch.positions.numel()),
                "cache_loc": int(forward_batch.out_cache_loc.numel()),
            }
        )
        n = int(forward_batch.input_ids.numel())
        return SimpleNamespace(
            logits_output=SimpleNamespace(
                next_token_logits=torch.zeros((n, VOCAB), dtype=torch.float32),
                hidden_states=torch.zeros((n, HIDDEN), dtype=torch.bfloat16),
            ),
            can_run_cuda_graph=False,
            routed_experts_output=None,
            indexer_topk_output=None,
        )


def _verify_input_kwargs(batch, spec):
    """The three shapes the target verify forward is built on."""
    return SimpleNamespace(
        spec_info=spec,
        input_ids=spec.draft_token,
        positions=spec.positions,
        out_cache_loc=batch.out_cache_loc,
    )


def _fake_assign_cache_locs(
    *,
    req_pool_indices,
    req_to_token,
    start_offset,
    end_offset,
    batch_size,
    draft_token_num,
    device,
):
    """Device kernel stand-in: the block is the committed prefix plus its offsets.

    Only the shape matters here (the eager fallback reshapes it to
    ``[bs, draft_token_num]``), but the values track the prefix lengths so a
    width mix-up shows up as a reshape failure.
    """
    base = start_offset.to(torch.int64).reshape(-1, 1)
    offs = torch.arange(draft_token_num, dtype=torch.int64, device=base.device)
    return (base + offs).reshape(-1)


class _WidthCase(CustomTestCase):
    """Drives the real startup order of ``Scheduler.init_all_cuda_graphs``:
    plan the launch-width scope, capture the launch width, register that state,
    then let ``init_states`` build the other routed width."""

    def setUp(self):
        super().setUp()
        self.captures = []
        override = get_context().override_server_args(
            speculative_algorithm="DFLASH",
            speculative_adaptive=True,
            speculative_adaptive_config=DFLASH2_POLICY,
            speculative_eagle_topk=1,
            speculative_num_steps=WIDE_STEPS,
            speculative_num_draft_tokens=WIDE,
            max_running_requests=MAX_RUNNING_REQUESTS,
            cuda_graph_config=_published_config(),
        )
        override.install()
        self.addCleanup(override.restore)
        self.addCleanup(cgc.clear_adaptive_launch_capture_scope)
        patchers = [
            patch.object(
                dflash_module, "capture_decode_graph", _capture_fixture(self.captures)
            ),
            patch.object(
                dflash_module,
                "DecodeCudaGraphRunner",
                lambda runner, **kw: SimpleNamespace(
                    name=f"target-graph-{kw['speculative_num_draft_tokens']}",
                    capture_bs=list(BUCKETS),
                ),
            ),
            patch.object(
                dflash_module, "get_available_gpu_memory", lambda *a, **k: 0.0
            ),
        ]
        for patcher in patchers:
            patcher.start()
            self.addCleanup(patcher.stop)

    def _patch(self, target, name, value):
        patcher = patch.object(target, name, value)
        patcher.start()
        self.addCleanup(patcher.stop)
        return patcher

    def _launched(
        self,
        launch_width=WIDE,
        decode_graphs=True,
        buckets=None,
        max_running_requests=MAX_RUNNING_REQUESTS,
        low_memory=False,
        real_sampler=False,
    ):
        if low_memory:
            # The <1 GB fallback: a CUDA device that is out of memory, so the
            # worker skips the launch capture itself.
            self._patch(dflash_module, "is_cuda", lambda: True)
            self._patch(dflash_module, "get_available_gpu_memory", lambda *a, **k: 0.5)
        worker = _Worker(
            launch_width, DFLASH2_POLICY, self.captures, [], real_sampler=real_sampler
        )
        if (
            buckets is not None
            or max_running_requests != MAX_RUNNING_REQUESTS
            or not decode_graphs
        ):
            get_context().override(
                "test.capture_scope",
                cuda_graph_config=_published_config(
                    backend=Backend.FULL if decode_graphs else Backend.DISABLED,
                    bs=buckets,
                    max_bs=max_running_requests,
                ),
                max_running_requests=max_running_requests,
            )
        if not decode_graphs:
            # With the decode phase off, the target worker captured no verify
            # graph either -- the slot is empty, which is what the routed states
            # are held against.
            worker.model_runner.init_cuda_graphs(capture_decode_cuda_graph=False)
        worker.prepare_adaptive_launch_capture()
        worker.init_cuda_graphs()
        cgc.clear_adaptive_launch_capture_scope()
        return worker

    @staticmethod
    def _run(worker, bs):
        with (
            patch.object(
                DFlashVerifyInput,
                "prepare_for_verify",
                lambda self_, batch, target_worker: (
                    _verify_input_kwargs(batch, self_),
                    False,
                ),
            ),
            patch.object(
                dflash_module,
                "assign_extend_cache_locs_func",
                _fake_assign_cache_locs,
            ),
        ):
            return worker.forward_batch_generation(_decode_batch(bs))


def _decode_batch(bs):
    seq_lens = _SeqLens(torch.arange(bs, dtype=torch.int64) * 7 + 11)
    return SimpleNamespace(
        forward_mode=ForwardMode.DECODE,
        is_extend_in_batch=False,
        spec_info=make_draft_input_v2(
            bonus_tokens=torch.arange(bs, dtype=torch.int64) + 1,
            new_seq_lens=seq_lens,
        ),
        seq_lens=seq_lens,
        seq_lens_cpu=None,
        seq_lens_sum=0,
        req_pool_indices=torch.arange(bs, dtype=torch.int64),
        sampling_info=None,
        has_grammar=False,
        return_logprob=False,
        out_cache_loc=None,
        reqs=[SimpleNamespace(decode_batch_idx=0) for _ in range(bs)],
    )


class TestPreparedWidthStates(_WidthCase):
    def test_only_the_reachable_width_bucket_pairs_are_captured(self):
        self._launched()
        by_width = {c["width"]: c for c in self.captures}
        self.assertEqual(sorted(by_width), [NARROW, WIDE])
        self.assertEqual(by_width[WIDE]["bs"], [1], "block6 only where C1 reaches it")
        self.assertEqual(by_width[NARROW]["bs"], [2, 3, 4, 5, 6], "block4 from C2 up")
        # Each extra width's capture ran against a fresh workspace (the launch
        # capture keeps the shared one, exactly as before this change)...
        self.assertTrue(by_width[NARROW]["workspace"], by_width[NARROW])
        # ... and folded in only its own sampler hook.
        for entry in by_width.values():
            self.assertEqual(entry["hooks"], 1, entry)

    def test_the_launch_width_stays_active_after_the_extra_build(self):
        worker = self._launched()
        self.assertEqual(worker.block_size, WIDE)
        self.assertEqual(worker.speculative_num_steps, WIDE_STEPS)
        self.assertEqual(worker.speculative_num_draft_tokens, WIDE)
        self.assertFalse(worker.draft_model_runner.init_new_workspace)
        # The launch hook survives; no other width's hook leaked onto the runner.
        self.assertEqual(len(worker.draft_model_runner.capture_tail_hooks), 1)
        self.assertIs(worker._draft_sampler, worker._dflash_widths[WIDE].draft_sampler)
        narrow_sampler = worker._dflash_widths[NARROW].draft_sampler
        self.assertIsNot(narrow_sampler, worker._dflash_widths[WIDE].draft_sampler)
        self.assertEqual(narrow_sampler.block_size, NARROW)
        # The published config was never mutated in place.
        published = get_exec().graph.cuda_graph_config
        self.assertEqual(published.decode.bs, BUCKETS)
        self.assertEqual(published.decode.backend, Backend.FULL)

    def test_a_width_switch_moves_every_bound_resource_together(self):
        worker = self._launched()
        controller = worker.adaptive_controller
        mark = len(worker.draft_model.seen)
        for batch_size, width in ((4, NARROW), (1, WIDE), (6, NARROW), (2, NARROW)):
            controller.activate_step_by_batch(batch_size)
            state = controller._states[width - 1]
            width_state = state.width_state
            self.assertEqual(worker.block_size, width)
            self.assertEqual(worker.speculative_num_draft_tokens, width)
            self.assertEqual(worker.speculative_num_steps, width - 1)
            self.assertEqual(
                worker.draft_model.block_size, width, "convolution block layout"
            )
            self.assertEqual(worker._block_pos_offsets.numel(), width)
            self.assertEqual(worker._draft_block_spec_info.draft_token_num, width)
            for name in _BLOCK_SHAPE_ATTRS:
                buf = getattr(worker, name)
                self.assertEqual(buf.shape[1], width, f"{name} serves another width")
                self.assertTrue(buf.is_contiguous(), f"{name} is a strided view")
                self.assertIs(buf, getattr(width_state, _WIDTH_ATTRS[name]))
            for name in ("_draft_block_end_buf", "_draft_seq_lens_cpu_buf"):
                self.assertEqual(
                    getattr(worker, name).shape,
                    (MAX_RUNNING_REQUESTS,),
                    name,
                )
            self.assertIs(worker._draft_sampler, width_state.draft_sampler)
            self.assertIs(
                worker.draft_model_runner.decode_cuda_graph_runner,
                state.cuda_graph_runner,
            )
            self.assertIs(
                worker.draft_model_runner.attn_backend, state.draft_attn_backend
            )
            self.assertIs(
                worker._target_worker.model_runner.decode_cuda_graph_runner,
                state.target_graph_runner,
            )
            self.assertIs(
                worker._target_worker.model_runner.attn_backend,
                state.target_attn_backend,
            )
            # No large allocation was duplicated: every width addresses the one
            # KV / Mamba / QSA-ring pool.
            self.assertIs(
                worker.draft_model_runner.token_to_kv_pool, worker.shared_pool
            )
            self.assertIs(
                worker._target_worker.model_runner.token_to_kv_pool,
                worker.shared_pool,
            )
            # Generic accounting follows the active width.
            self.assertEqual(get_spec().speculative_num_draft_tokens, width)
        self.assertEqual(
            worker.draft_model.seen[mark:],
            [NARROW, WIDE, NARROW],
            "exactly one conv relayout per real switch",
        )

    def test_repeated_joins_and_drains_recapture_and_reallocate_nothing(self):
        worker = self._launched()
        controller = worker.adaptive_controller
        captured = len(self.captures)
        pointers = {
            width: {
                field: getattr(worker._dflash_widths[width], field).data_ptr()
                for field in _WIDTH_ATTRS.values()
            }
            for width in (WIDE, NARROW)
        }
        for _ in range(8):
            for batch_size in (1, 2, 3, 5, 6):
                controller.activate_step_by_batch(batch_size)
                worker.on_verify_complete_cpu([0] * batch_size, batch_size)
                worker._ensure_draft_block_buffers(batch_size)
                controller.activate_step_by_batch(1)
        self.assertEqual(len(self.captures), captured, "a transition recaptured")
        for width, by_field in pointers.items():
            for field, ptr in by_field.items():
                got = getattr(worker._dflash_widths[width], field).data_ptr()
                self.assertEqual(got, ptr, f"{field} moved under width {width}")
        self.assertEqual(controller.candidate_steps, [NARROW_STEPS, WIDE_STEPS])
        self.assertEqual(sorted(controller._states), [NARROW_STEPS, WIDE_STEPS])

    def test_the_routed_provisioning_covers_the_whole_launch_capacity(self):
        worker = self._launched()
        for width in (WIDE, NARROW):
            state = worker._dflash_widths[width]
            self.assertGreaterEqual(state.scratch_cap, MAX_RUNNING_REQUESTS, width)
            self.assertGreaterEqual(state.bonus_cap, MAX_RUNNING_REQUESTS, width)
            self.assertEqual(tuple(state.ids.shape), (MAX_RUNNING_REQUESTS, width))
            self.assertEqual(
                tuple(state.out_tokens[0].shape), (MAX_RUNNING_REQUESTS, width)
            )

    def test_the_outgoing_backends_are_drained_before_anything_is_repointed(self):
        from sglang.srt.layers.attention.hybrid_linear_attn_backend import (
            HybridLinearAttnBackend,
        )

        worker = self._launched()

        def recovery_backend(tag):
            backend = HybridLinearAttnBackend.__new__(HybridLinearAttnBackend)
            backend.name = tag
            backend._recovery_event_pending = True
            backend._recovery_event = SimpleNamespace(
                wait=lambda: worker.events.append(f"drain:{tag}")
            )
            backend.token_to_kv_pool = worker.shared_pool
            backend.req_to_token_pool = SimpleNamespace(req_to_token=None)
            return backend

        shared_extend = recovery_backend("draft")
        worker.model_runner.attn_backend = recovery_backend("target")
        worker.draft_model_runner.attn_backend = shared_extend
        worker.draft_model_runner.draft_attn_backend = shared_extend  # deduped
        worker.events.clear()

        worker.adaptive_controller.activate_step_by_batch(4)
        worker.adaptive_controller.activate_step_by_batch(4)  # re-select: no re-drain
        self.assertEqual(
            worker.events,
            ["drain:target", "drain:draft"],
            "outgoing backends were not drained first",
        )

    def test_a_routed_state_with_no_captured_graph_still_switches(self):
        # Decode graphs off: the widths still differ in model metadata and
        # scratch, and nothing at all is captured or folded for any of them.
        worker = self._launched(decode_graphs=False)
        self.assertEqual(self.captures, [])
        controller = worker.adaptive_controller
        self.assertEqual(sorted(controller._states), [NARROW_STEPS, WIDE_STEPS])
        for steps, width in ((NARROW_STEPS, NARROW), (WIDE_STEPS, WIDE)):
            state = controller._states[steps]
            self.assertIsNone(state.width_state.draft_sampler, width)
            self.assertIsNone(state.cuda_graph_runner, width)
            self.assertIsNone(state.target_graph_runner, width)
        controller.activate_step_by_batch(4)
        self.assertEqual(worker.block_size, NARROW)
        self.assertIsNone(worker.draft_model_runner.decode_cuda_graph_runner)
        self.assertIsNone(worker._target_worker.model_runner.decode_cuda_graph_runner)
        self.assertIsNone(worker._draft_sampler, "a capture-only sampler was built")
        controller.activate_step_by_batch(1)
        self.assertEqual(worker.block_size, WIDE)
        self.assertIsNone(worker.draft_model_runner.decode_cuda_graph_runner)


# ------------------------------------------------- the decode forward path ----#


# ------------------------------ widths that reach no bucket / capture nothing ----#


class TestUnreachableAndUncapturedWidths(_WidthCase):
    """Startup must survive a routed width that captures nothing.

    Two ways a width ends up with no graphs, both from decisions the launch
    already made: its tier reaches no decode bucket, or the launch capture was
    skipped (decode graphs off, or the <1 GB low-memory fallback). Neither may
    turn into a capture anyway -- the capture-only folded sampler is sized off
    ``decode.bs``, so building one for an empty bucket list is a startup crash --
    and both run through the REAL sampler builder and the REAL controller
    pruning, not the fixture's stand-in.
    """

    def _real_tp_group(self):
        self._patch(
            dflash_module, "get_tp_group", lambda: SimpleNamespace(world_size=1)
        )

    def test_a_width_that_reaches_no_bucket_starts_without_a_sampler(self):
        # Shipped policy, canonical buckets [1] (max-running-requests=1): the
        # real controller prunes block4 to no bucket at all.
        self._real_tp_group()
        worker = self._launched(buckets=[1], max_running_requests=1, real_sampler=True)
        params = worker.adaptive_controller.params
        self.assertEqual(params.cuda_graph_bs_for_step(WIDE_STEPS), [1])
        self.assertEqual(params.cuda_graph_bs_for_step(NARROW_STEPS), [])

        # The real builder ran: block6 got a real folded sampler sized for the
        # one bucket it can reach, and block4 got none rather than max([]).
        wide = worker._dflash_widths[WIDE]
        narrow = worker._dflash_widths[NARROW]
        self.assertIsInstance(wide.draft_sampler, dflash_module._DflashDraftSampler)
        self.assertEqual(wide.draft_sampler.block_size, WIDE)
        self.assertEqual(wide.draft_sampler.out.numel(), 1 * (WIDE - 1))
        self.assertIsNone(narrow.draft_sampler, "a capture-only sampler was built")
        self.assertIsNone(
            worker.adaptive_controller._states[NARROW_STEPS].cuda_graph_runner
        )
        self.assertIsNone(
            worker.adaptive_controller._states[NARROW_STEPS].target_graph_runner
        )
        self.assertEqual([c["width"] for c in self.captures], [WIDE])
        self.assertEqual(self.captures[0]["bs"], [1])

    def test_a_c2_batch_on_a_c1_only_capture_list_still_routes_narrow(self):
        # The same startup, driven at the forward boundary: no C2 graph exists,
        # so the narrow round must run without one rather than stay wide.
        self._real_tp_group()
        worker = self._launched(buckets=[1], max_running_requests=4, real_sampler=True)
        self.assertEqual({c["width"] for c in self.captures}, {WIDE})
        for bs, width in ((1, WIDE), (3, NARROW), (1, WIDE)):
            result = self._run(worker, bs)
            self.assertEqual(result.speculative_num_draft_tokens, width)
            self.assertEqual(
                worker.draft_model_runner.decode_cuda_graph_runner is None,
                width == NARROW,
            )

    def test_the_low_memory_no_capture_decision_holds_for_every_width(self):
        # <1 GB free after target backend init: the worker skips the launch
        # capture. init_states must not hand the other width "no pruning" (which
        # is what an empty list degrades to inside the params) and capture both
        # widths' graphs anyway.
        worker = self._launched(low_memory=True)
        self.assertEqual(self.captures, [], "a width captured after the fallback")
        self.assertIsNone(worker._draft_sampler)
        controller = worker.adaptive_controller
        extra = controller._states[NARROW_STEPS]
        self.assertIsNone(extra.cuda_graph_runner)
        self.assertIsNone(extra.target_graph_runner)
        self.assertIsNone(extra.width_state.draft_sampler)
        # Switching is still just a pointer swap, and between two eager widths at
        # that: neither the draft nor (in the extra state) the target graph runner
        # exists. The launch state keeps the fixture's pre-seeded target runner
        # because the target worker's own capture decision lives in
        # TpModelWorker.init_cuda_graphs, which this fixture does not simulate.
        controller.activate_step_by_batch(4)
        self.assertEqual(worker.block_size, NARROW)
        self.assertIsNone(worker.draft_model_runner.decode_cuda_graph_runner)
        self.assertIsNone(worker._target_worker.model_runner.decode_cuda_graph_runner)
        self.assertIsNone(worker._draft_sampler)
        controller.activate_step_by_batch(1)
        self.assertEqual(worker.block_size, WIDE)
        self.assertIsNone(worker.draft_model_runner.decode_cuda_graph_runner)
        self.assertIsNotNone(
            worker._target_worker.model_runner.decode_cuda_graph_runner
        )


class TestDecodeBoundary(_WidthCase):
    """One verify round per concurrency level through the real worker."""

    def test_each_batch_verifies_at_the_width_its_own_size_maps_to(self):
        worker = self._launched()
        for bs, width in ((1, WIDE), (2, NARROW), (6, NARROW), (1, WIDE)):
            result = self._run(worker, bs)
            self.assertEqual(worker.block_size, width)
            self.assertEqual(result.speculative_num_draft_tokens, width)
            self.assertEqual(worker.draft_model.block_size, width)
            seen = worker.target_graphs[-1]
            self.assertEqual(seen["width"], width, "verify spec metadata")
            self.assertEqual(seen["tokens"], bs * width, "verify token count")
            self.assertEqual(seen["positions"], bs * width, "positional offsets")
            self.assertEqual(seen["cache_loc"], bs * width, "verify KV slots")
            self.assertEqual(result.accept_lens.shape, (bs,))
            self.assertEqual(result.next_token_ids.numel(), bs * width)

    def test_c1_c2_c1_and_c2_c6_c1_reuse_the_prepared_states(self):
        worker = self._launched()
        captured = len(self.captures)
        for bs, width in ((1, WIDE), (2, NARROW), (1, WIDE)):
            self.assertEqual(self._run(worker, bs).speculative_num_draft_tokens, width)
        for bs, width in ((2, NARROW), (6, NARROW), (1, WIDE)):
            self.assertEqual(self._run(worker, bs).speculative_num_draft_tokens, width)
        self.assertEqual(len(self.captures), captured, "a round recaptured a graph")

    def test_a_narrow_round_reuses_the_provisioned_scratch(self):
        worker = self._launched()
        self._run(worker, 1)
        ptrs = {
            width: getattr(worker._dflash_widths[width], "ids").data_ptr()
            for width in (WIDE, NARROW)
        }
        self._run(worker, 6)
        self._run(worker, 1)
        for width, ptr in ptrs.items():
            self.assertEqual(worker._dflash_widths[width].ids.data_ptr(), ptr, width)

    def test_an_in_flight_batch_keeps_its_own_width_after_a_concurrency_change(self):
        # Overlap: the C4 round's result is processed after the worker has moved
        # on to the C1 width. The recorded stride is what slices the accepted
        # tokens, so it has to stay 4 -- otherwise the next round's layout reads
        # the previous one's block geometry.
        worker = self._launched()
        in_flight = self._run(worker, 4)
        self.assertEqual(in_flight.speculative_num_draft_tokens, NARROW)
        self._run(worker, 1)
        self.assertEqual(worker.block_size, WIDE)
        self.assertEqual(in_flight.speculative_num_draft_tokens, NARROW)
        self.assertEqual(in_flight.next_token_ids.numel(), 4 * NARROW)

    def test_the_prepared_over_allocation_covers_the_widest_routed_width(self):
        # ``prepare_for_decode`` runs before the width is chosen, so its
        # over-allocation must be sized off the routed maximum (block6 -> 12
        # slots/request), never off a narrower currently-active width: an
        # under-reservation writes past the over-allocated req_to_token mapping.
        reserves = {}

        def capture_reserve(reqs, *, reserve, page_size):
            reserves[get_spec().speculative_num_draft_tokens] = (reserve, page_size)
            cur = [int(r.kv.kv_allocated_len) for r in reqs]
            nxt = [c + reserve for c in cur]
            return cur, nxt, sum(n - c for n, c in zip(nxt, cur))

        for width in (NARROW, WIDE):
            get_context().override(
                "test.width",
                speculative_num_steps=width - 1,
                speculative_num_draft_tokens=width,
            )
            req = SimpleNamespace(
                kv=SimpleNamespace(kv_allocated_len=64),
                kv_committed_len=64,
                sampling_params=SimpleNamespace(top_k=1),
                decode_batch_idx=0,
            )
            batch = SimpleNamespace(
                device="cpu",
                batch_size=lambda: 1,
                maybe_evict_swa=lambda: None,
                seq_lens_cpu=None,
                seq_lens_sum=0,
                reqs=[req],
                req_pool_indices=torch.arange(1, dtype=torch.int64),
                tree_cache=SimpleNamespace(
                    token_to_kv_pool_allocator=SimpleNamespace(page_size=64)
                ),
                token_to_kv_pool_allocator=SimpleNamespace(page_size=64),
                req_to_token_pool=SimpleNamespace(req_to_token=None),
            )
            spec = make_draft_input_v2(
                bonus_tokens=torch.ones((1,), dtype=torch.int64),
                new_seq_lens=torch.ones((1,), dtype=torch.int64),
            )
            with (
                patch.object(
                    dflash_info_v2, "page_aligned_decode_alloc_lens", capture_reserve
                ),
                patch.object(
                    dflash_info_v2,
                    "alloc_for_spec_decode",
                    lambda *a, **k: None,
                ),
            ):
                spec.prepare_for_decode(batch)
            self.assertEqual(reserves[width], (2 * WIDE, 64), f"width {width}")
            self.assertEqual(req.decode_batch_idx, 1)
            self.assertEqual(batch.seq_lens_sum, 64)


# ---------------------------------- selector sampling scratch per width ----#


class TestSelectorSamplingScratch(_WidthCase):
    """Normal (non-greedy) selector sampling across width transitions.

    ``_selector_sampling_accept`` scatters the selector's sparse q into a dense
    ``[bs, gamma, vocab]`` buffer for the accept kernel, and ``gamma`` is the
    width minus one -- so that scratch belongs to a width and cannot outlive a
    switch. Driven through the real method with the real torch scatter and a
    stubbed accept kernel: alternating C1/C2/C6 rounds must never re-grow a
    worker-wide buffer (each realloc used to double its capacity, the OOM path)
    and must still leave the buffer zeroed for the next round.
    """

    TOP_K = 2

    def _round(self, worker, batch_size):
        """One selector-sampling accept round at whatever width this batch routes to."""
        worker.activate_step_by_batch(batch_size)
        width = worker.block_size
        gamma = width - 1
        # Unique ids per (row, position) slot: the scatter must land exactly on
        # the q rows, so duplicated ids inside one slice would muddy the check.
        candidate_ids = torch.arange(
            batch_size * gamma * self.TOP_K, dtype=torch.int64
        ).reshape(batch_size, gamma, self.TOP_K)
        q_rows = torch.rand((batch_size, gamma, self.TOP_K), dtype=torch.float32)
        recorded = []

        def _accept(**kwargs):
            probs = kwargs["draft_probs"]
            assert probs.shape == (batch_size, gamma, VOCAB), probs.shape
            assert kwargs["gamma"] == gamma
            assert kwargs["verify_num_draft_tokens"] == width
            recorded.append(probs.clone())
            return (
                torch.ones(batch_size, dtype=torch.int64),
                torch.zeros(batch_size, dtype=torch.int64),
                None,
            )

        self._patch(dflash_module, "accept_sampling", _accept)
        accept_len, bonus = worker._selector_sampling_accept(
            candidates=torch.full((batch_size, width), 1, dtype=torch.int64),
            next_token_logits=torch.randn(batch_size, VOCAB),
            candidate_ids=candidate_ids,
            q_rows=q_rows,
            sampling_info=SimpleNamespace(is_all_greedy=False),
            draft_input=None,
        )
        self.assertEqual(accept_len.shape, (batch_size,))
        self.assertEqual(bonus.shape, (batch_size,))
        return width, recorded[0], candidate_ids, q_rows

    @staticmethod
    def _assert_q_association(probs, candidate_ids, q_rows):
        for row in range(probs.shape[0]):
            for slot in range(probs.shape[1]):
                ids = candidate_ids[row, slot]
                mass = probs[row, slot]
                torch.testing.assert_close(mass[ids], q_rows[row, slot])
                outside = mass.clone()
                outside[ids] = 0.0
                assert float(outside.sum()) == 0.0

    def test_repeated_joins_and_drains_never_reallocate_the_scratch(self):
        worker = self._launched()
        for _ in range(6):
            for batch_size in (1, 2, 6, 1, 3, 1):
                width, probs, ids, q_rows = self._round(worker, batch_size)
                self._assert_q_association(probs, ids, q_rows)
                # The finally-clear ran: the live buffer is back to zero, so the
                # next round cannot read this round's mass at stale ids.
                self.assertEqual(
                    float(worker._draft_probs_buf[:batch_size].abs().sum()),
                    0.0,
                    f"width {width} left q behind",
                )

        pointers = {
            width: worker._dflash_widths[width].draft_probs.data_ptr()
            for width in (WIDE, NARROW)
        }
        for width, state in worker._dflash_widths.items():
            self.assertEqual(state.probs_cap, MAX_RUNNING_REQUESTS, width)
            self.assertEqual(
                tuple(state.draft_probs.shape), (MAX_RUNNING_REQUESTS, width - 1, VOCAB)
            )

        # The largest batch the run can serve, on both widths: still no new buffer.
        self._round(worker, MAX_RUNNING_REQUESTS)
        self._round(worker, 1)
        for width, ptr in pointers.items():
            self.assertEqual(
                worker._dflash_widths[width].draft_probs.data_ptr(), ptr, width
            )

    def test_the_live_scratch_pointer_follows_the_active_width(self):
        worker = self._launched()
        self._round(worker, 1)
        self.assertIs(worker._draft_probs_buf, worker._dflash_widths[WIDE].draft_probs)
        self.assertEqual(worker._draft_probs_buf.shape[1], WIDE - 1)
        self._round(worker, 4)
        self.assertIs(
            worker._draft_probs_buf, worker._dflash_widths[NARROW].draft_probs
        )
        self.assertEqual(worker._draft_probs_buf.shape[1], NARROW - 1)

    def test_a_single_fixed_width_still_grows_the_scratch_lazily(self):
        # No policy -> no routing -> the buffer stays exactly the pre-change
        # lazy-growth scratch, sized off the batches that actually arrived.
        worker = _Worker(
            WIDE, config_path=None, captures=self.captures, target_graphs=[]
        )
        self._round(worker, 1)
        self.assertEqual(
            worker._draft_probs_buf.shape[0], 1, "no ceiling without routing"
        )
        self._round(worker, 3)
        self.assertEqual(tuple(worker._draft_probs_buf.shape), (3, WIDE - 1, VOCAB))
        self._round(worker, 2)
        self.assertEqual(
            worker._draft_probs_buf.shape[0], 3, "reallocated while it still fit"
        )


# ---------------------------------------------- fixed width / no policy ----#


class TestFixedWidthAndNoPolicyPaths(CustomTestCase):
    """Without the policy, DFLASH keeps exactly one width as before."""

    def setUp(self):
        super().setUp()
        self.captures = []
        override = get_context().override_server_args(
            speculative_algorithm="DFLASH",
            speculative_adaptive=False,
            speculative_adaptive_config=None,
            speculative_eagle_topk=1,
            speculative_num_steps=1,
            speculative_num_draft_tokens=WIDE,
            max_running_requests=MAX_RUNNING_REQUESTS,
            cuda_graph_config=_published_config(),
        )
        override.install()
        self.addCleanup(override.restore)
        self.addCleanup(cgc.clear_adaptive_launch_capture_scope)
        for patcher in (
            patch.object(
                dflash_module, "capture_decode_graph", _capture_fixture(self.captures)
            ),
            patch.object(
                dflash_module, "get_available_gpu_memory", lambda *a, **k: 0.0
            ),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)

    def _plain(self, launch_width=WIDE):
        return _Worker(
            launch_width, config_path=None, captures=self.captures, target_graphs=[]
        )

    def test_no_policy_means_one_width_and_the_full_capture_list(self):
        worker = self._plain()
        worker.prepare_adaptive_launch_capture()
        worker.init_cuda_graphs()
        self.assertIsNone(worker.adaptive_controller)
        self.assertEqual(list(worker._dflash_widths), [WIDE])
        self.assertEqual([c["width"] for c in self.captures], [WIDE])
        self.assertEqual(self.captures[0]["bs"], BUCKETS, "no pruning without a policy")
        self.assertEqual(cgc._adaptive_launch_capture_scope, None)

    def test_activation_and_feedback_are_no_ops_without_the_policy(self):
        worker = self._plain(NARROW)
        worker.init_cuda_graphs()
        for batch_size in (1, 2, 6):
            worker.activate_step_by_batch(batch_size)
        worker.on_verify_complete_cpu([0], batch_size=1)
        self.assertEqual(worker.block_size, NARROW)
        self.assertEqual(worker.speculative_num_draft_tokens, NARROW)
        self.assertEqual(worker.draft_model.seen, [NARROW], "no relayout")

    def test_a_fixed_launch_still_verifies_at_its_own_width(self):
        worker = self._plain(WIDE)
        worker.init_cuda_graphs()
        result = _WidthCase._run(worker, 1)
        self.assertEqual(result.speculative_num_draft_tokens, WIDE)
        self.assertEqual(worker.target_graphs[-1]["width"], WIDE)
        result = _WidthCase._run(worker, 6)
        self.assertEqual(result.speculative_num_draft_tokens, WIDE)

    def test_scratch_still_grows_lazily_on_the_fixed_path(self):
        worker = self._plain(WIDE)
        worker._ensure_draft_block_buffers(2)
        self.assertEqual(tuple(worker._draft_block_ids_buf.shape), (2, WIDE))
        worker._ensure_draft_block_buffers(5)
        self.assertEqual(tuple(worker._draft_block_ids_buf.shape), (5, WIDE))
        worker._ensure_draft_block_buffers(3)
        self.assertEqual(tuple(worker._draft_block_ids_buf.shape), (5, WIDE))


# ------------------------------------------------------- arg resolution ------#


def _make_dflash_args(**overrides) -> ServerArgs:
    """A config record for the DFLASH launch path; no weights are read.

    ``model_path="dummy"`` short-circuits ``ServerArgs.__post_init__`` so the
    speculative hook is invoked directly, the same pattern as the sibling
    server_args tests. The draft path is a local directory holding a bare
    ``config.json`` because the hook reads the draft architecture (NEXTN/EAGLE
    aliasing) before the policy is consulted; the draft attention backend is
    named explicitly so nothing else reaches for the checkpoint.
    """
    draft_dir = tempfile.mkdtemp(prefix="dflash-draft-")
    with open(os.path.join(draft_dir, "config.json"), "w") as f:
        json.dump({"model_type": "qwen2", "architectures": ["Qwen2ForCausalLM"]}, f)
    args = ServerArgs(model_path="dummy")
    args.speculative_algorithm = "DFLASH"
    args.speculative_draft_model_path = draft_dir
    args.speculative_draft_attention_backend = "triton"
    args.device = "cuda"
    args.max_running_requests = MAX_RUNNING_REQUESTS
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


class TestArgResolution(CustomTestCase):
    """The launch must carry the widest routed width, so every graph-bound
    allocation is provisioned for it once; nothing here may drift into the
    fixed-width DFLASH path."""

    def setUp(self):
        super().setUp()
        self._draft_dirs = []

    def tearDown(self):
        for path in self._draft_dirs:
            shutil.rmtree(path, ignore_errors=True)
        super().tearDown()

    def _args(self, **overrides) -> ServerArgs:
        args = _make_dflash_args(**overrides)
        self._draft_dirs.append(args.speculative_draft_model_path)
        return args

    def test_the_policy_launches_at_the_widest_routed_width(self):
        args = self._args(
            speculative_adaptive=True,
            speculative_adaptive_config=DFLASH2_POLICY,
            speculative_num_steps=None,
            speculative_num_draft_tokens=None,
        )
        handle_speculative_decoding(args)
        self.assertTrue(resolution_result(args, "speculative_adaptive"))
        self.assertEqual(resolution_result(args, "speculative_num_steps"), WIDE_STEPS)
        self.assertEqual(resolution_result(args, "speculative_num_draft_tokens"), WIDE)
        self.assertEqual(args.max_speculative_num_draft_tokens, WIDE)

    def test_a_narrower_explicit_launch_width_is_lifted_with_a_warning(self):
        # The policy owns the widths; a flag naming a narrower one must not
        # silently under-provision what the captured graphs bind.
        args = self._args(
            speculative_adaptive=True,
            speculative_adaptive_config=DFLASH2_POLICY,
            speculative_num_steps=NARROW_STEPS,
            speculative_num_draft_tokens=NARROW,
        )
        with self.assertLogs(
            "sglang.srt.arg_groups.speculative_hook", "WARNING"
        ) as logs:
            handle_speculative_decoding(args)
        self.assertEqual(resolution_result(args, "speculative_num_steps"), WIDE_STEPS)
        self.assertEqual(resolution_result(args, "speculative_num_draft_tokens"), WIDE)
        self.assertTrue(any("widest routed width" in m for m in logs.output))

    def test_a_block_size_conflicting_with_the_policy_is_refused(self):
        args = self._args(
            speculative_adaptive=True,
            speculative_adaptive_config=DFLASH2_POLICY,
            speculative_dflash_block_size=NARROW,
        )
        with self.assertRaises(ValueError) as ctx:
            handle_speculative_decoding(args)
        self.assertIn("--speculative-dflash-block-size", str(ctx.exception))

    def test_a_multi_candidate_dflash_policy_disables_adaptive_without_guessing(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "cfg.json")
            with open(path, "w") as f:
                json.dump(_table({1: [NARROW_STEPS, WIDE_STEPS], 2: [NARROW_STEPS]}), f)
            args = self._args(
                speculative_adaptive=True,
                speculative_adaptive_config=path,
                speculative_num_steps=None,
                speculative_num_draft_tokens=NARROW,
            )
            with self.assertLogs(
                "sglang.srt.arg_groups.speculative_hook", "WARNING"
            ) as logs:
                handle_speculative_decoding(args)
        self.assertFalse(resolution_result(args, "speculative_adaptive"))
        self.assertTrue(any("adaptive" in m.lower() for m in logs.output))
        # Back to the ordinary fixed-width DFLASH launch.
        self.assertEqual(resolution_result(args, "speculative_num_steps"), 1)
        self.assertEqual(
            resolution_result(args, "speculative_num_draft_tokens"), NARROW
        )

    def test_the_full_prefill_graph_launch_keeps_one_fixed_width(self):
        # Through the real resolution pipeline: a DFLASH launch asking for the
        # routing policy plus a full prefill graph loses adaptive and keeps its
        # explicitly requested width, rather than capturing a prefill graph that
        # would assert on replay after the first width switch.
        args = _make_dflash_args(
            speculative_adaptive=True,
            speculative_adaptive_config=DFLASH2_POLICY,
            speculative_num_steps=None,
            speculative_num_draft_tokens=WIDE,
            cuda_graph_config={"prefill": {"backend": "full"}},
        )
        with self.assertLogs(
            "sglang.srt.arg_groups.speculative_hook", "WARNING"
        ) as logs:
            handle_speculative_decoding(args)
        self.assertFalse(resolution_result(args, "speculative_adaptive"))
        self.assertEqual(resolution_result(args, "speculative_num_steps"), 1)
        self.assertEqual(resolution_result(args, "speculative_num_draft_tokens"), WIDE)
        self.assertTrue(
            any("prefill" in m and "full" in m for m in logs.output),
            "the refusal must name the conflicting prefill graph setting",
        )

    def test_a_breakable_prefill_graph_launch_keeps_the_routing(self):
        # The shipped default prefill backend stays supported by the routing.
        args = _make_dflash_args(
            speculative_adaptive=True,
            speculative_adaptive_config=DFLASH2_POLICY,
            speculative_num_steps=None,
            speculative_num_draft_tokens=None,
            cuda_graph_config={"prefill": {"backend": "breakable"}},
        )
        handle_speculative_decoding(args)
        self.assertTrue(resolution_result(args, "speculative_adaptive"))
        self.assertEqual(resolution_result(args, "speculative_num_steps"), WIDE_STEPS)

    def test_the_fixed_no_policy_launch_is_unchanged(self):
        args = self._args(speculative_num_steps=3, speculative_num_draft_tokens=NARROW)
        with self.assertLogs(
            "sglang.srt.arg_groups.speculative_hook", "WARNING"
        ) as logs:
            handle_speculative_decoding(args)
        self.assertFalse(resolution_result(args, "speculative_adaptive"))
        self.assertEqual(resolution_result(args, "speculative_num_steps"), 1)
        self.assertEqual(
            resolution_result(args, "speculative_num_draft_tokens"), NARROW
        )
        self.assertTrue(
            any("speculative_num_steps" in m for m in logs.output),
            "the pre-existing DFLASH step override must still be reported",
        )


if __name__ == "__main__":
    import sys

    sys.exit(__import__("pytest").main([__file__, "-v"]))
