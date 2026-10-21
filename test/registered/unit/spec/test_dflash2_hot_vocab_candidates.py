"""FR-Spec hot-vocabulary candidate projection for DFlash2 (DFlash2DraftModel).

Without a token map, DFlash2 projects the *full* target lm_head and then takes
the selector top-k. With ``--speculative-token-map`` the draft projects a
reduced head built from that map's rows of the same target Parameter and has to
turn the shortlist *positions* back into the ORIGINAL global token ids before
the selector codebook lookup and before acceptance. These fixtures pin that
contract on CPU: permuted/noncontiguous ids, FP8 values travelling with their
matching per-row scales, an untouched target head, a full-vocab anchor outside
the hot set, q association through the real acceptance scatter, one allocation
shared by the eager and captured paths (including startup when folding is
disabled), and the unchanged no-map behavior. TP1 full-width heads only (the
qualified FR-Spec topology); ``test_hot_vocab_shard_selection.py`` covers the
sharded TP>1 assembly that ``install_hot_candidate_head`` delegates to.
"""

import sys
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from sglang.kernels.ops.gemm.sm120_online_fp8 import (
    dequantize_rowwise_weight,
    replace_linear_weight_rowwise_fp8,
    rowwise_scale_of,
)
from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod
from sglang.srt.models.dflash import (
    CandidateSelector,
    DFlash2DraftModel,
)
from sglang.srt.speculative import dflash_worker_v2 as worker_mod
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=15, suite="base-a-test-cpu")

# The resident rowwise head projects through a Triton GEMV below its row limit
# and through the dequantizing fallback above it (sm120_online_fp8._MAX_KERNEL_ROWS).
# CPU fixtures stay above that limit so they exercise the same numerics without a
# GPU driver, exactly as the rowwise fixtures in test_dflash_logits.py do; the
# kernel path itself is validated on hardware.
CPU_SAFE_ROWS = 40


class _DraftStub:
    """A real object carrying the DFlash2 hot-candidate methods and the
    attributes ``DFlash2DraftModel.__init__`` sets, without the backbone."""

    install_hot_candidate_head = DFlash2DraftModel.install_hot_candidate_head
    compute_candidates = DFlash2DraftModel.compute_candidates
    _transform_unary_logits = DFlash2DraftModel._transform_unary_logits

    def __init__(self, lm_head, top_k: int) -> None:
        self.lm_head = lm_head
        self.candidate_head = None
        self.candidate_selector = SimpleNamespace(top_k=int(top_k))
        self.draft_config = SimpleNamespace(
            output_multiplier=1.0, final_logit_softcapping=None
        )


def _worker_shim(*, block_size=8, selector, draft_model, lm_head=None, tp_rank=0):
    """Worker stand-in with the composite's real method bindings."""
    worker = SimpleNamespace(
        block_size=block_size,
        selector=selector,
        ps=SimpleNamespace(tp_rank=tp_rank),
        device="cpu",
        use_compact_draft_cache=False,
        draft_model=draft_model,
        _draft_probs_buf=None,
        _draft_worker=SimpleNamespace(alloc_memory_pool=lambda **kwargs: None),
        _target_worker=SimpleNamespace(
            model_runner=SimpleNamespace(
                model=SimpleNamespace(lm_head=lm_head),
            )
        ),
    )
    worker._install_selector_head = (
        lambda head: worker_mod.DFlashWorkerV2._install_selector_head(worker, head)
    )
    return worker


def _patch_env(monkeypatch, token_map, eager_sampler=False):
    """Pin the two module globals the install path reads: the spec bag and envs."""
    monkeypatch.setattr(
        worker_mod,
        "get_spec",
        lambda: SimpleNamespace(speculative_token_map=token_map),
    )
    monkeypatch.setattr(
        worker_mod,
        "envs",
        SimpleNamespace(
            SGLANG_DFLASH_EAGER_DRAFT_SAMPLER=SimpleNamespace(get=lambda: eager_sampler)
        ),
    )


def _contract_topk(scores, k, sorted=False, deterministic=False):
    """CPU stand-in pinning flashinfer.top_k's call contract (see the note in
    test_dflash_logits.py about why these fixtures must pin their top-k)."""
    assert scores.is_contiguous()
    assert sorted and deterministic
    return torch.topk(scores, k, dim=-1)


def _use_cpu_top_k(monkeypatch):
    monkeypatch.setattr("sglang.srt.models.dflash._flashinfer_top_k", _contract_topk)


def _tp1(monkeypatch):
    monkeypatch.setattr(
        "sglang.srt.models.dflash.get_parallel", lambda: SimpleNamespace(tp_size=1)
    )


def _dense_head(rows: int, cols: int, seed: int = 0):
    torch.manual_seed(seed)
    weight = nn.Parameter(
        torch.randn(rows, cols, dtype=torch.bfloat16), requires_grad=False
    )
    return SimpleNamespace(
        weight=weight,
        org_vocab_size=rows,
        quant_method=UnquantizedLinearMethod(),
    )


def _rowwise_head(rows: int, cols: int, seed: int = 0):
    """A target head in the resident Flash-Next state: BF16 values quantized to
    E4M3 with one FP32 scale per output channel, behind the unquantized method
    the NVFP4 checkpoint leaves on lm_head."""
    torch.manual_seed(seed)
    dense = torch.randn(rows, cols, dtype=torch.bfloat16)
    head = nn.Linear(cols, rows, bias=False, dtype=torch.bfloat16)
    head.weight.data.copy_(dense)
    replace_linear_weight_rowwise_fp8(head)
    head.quant_method = UnquantizedLinearMethod()
    head.org_vocab_size = rows
    return head


def _snapshot(head):
    data = head.weight.detach().float().clone()
    scale = rowwise_scale_of(head.weight)
    return data, None if scale is None else scale.clone()


def _assert_untouched(head, snapshot):
    data, scale = snapshot
    assert head.weight.shape == data.shape
    torch.testing.assert_close(head.weight.detach().float(), data)
    if scale is None:
        assert rowwise_scale_of(head.weight) is None
    else:
        torch.testing.assert_close(rowwise_scale_of(head.weight), scale)


def test_hot_head_narrows_the_projection_and_returns_permuted_global_ids(monkeypatch):
    """Noncontiguous, permuted map ids: top-k is taken over the hot rows only and
    the shortlist positions come back as the ORIGINAL global ids (map order, not
    sorted, never a hot-set position)."""
    _use_cpu_top_k(monkeypatch)
    _tp1(monkeypatch)
    hot_ids = [13, 3, 0, 9, 6]  # permuted, noncontiguous, spans the head
    head = _dense_head(16, 4, seed=7)
    snapshot = _snapshot(head)
    hidden = torch.randn(5, 4, dtype=torch.bfloat16)
    model = _DraftStub(head, top_k=3)

    assert model.install_hot_candidate_head(head, torch.tensor(hot_ids)) == len(hot_ids)
    candidate_ids, unary_logits = model.compute_candidates(hidden)

    assert candidate_ids.shape == (5, 3)
    assert unary_logits.shape == (5, 3)
    full = (hidden.bfloat16() @ head.weight.T).float()
    expected_logits, expected_pos = torch.topk(full[:, hot_ids], 3, dim=-1)
    torch.testing.assert_close(candidate_ids, torch.tensor(hot_ids)[expected_pos])
    torch.testing.assert_close(unary_logits, expected_logits, rtol=1e-2, atol=1e-2)
    assert model.candidate_head.width == len(hot_ids)
    assert set(candidate_ids.flatten().tolist()) <= set(hot_ids)
    _assert_untouched(head, snapshot)

    # An out-of-map row made dominant is shortlisted only by a draft with no
    # installed hot head: the candidate set is narrowed by the map, not by luck.
    head.weight.data[15] = 100.0
    pos_hidden = torch.rand(5, 4, dtype=torch.bfloat16) + 0.1
    narrow_ids, _ = model.compute_candidates(pos_hidden)
    assert 15 not in set(narrow_ids.flatten().tolist())
    full_ids, _ = _DraftStub(head, top_k=3).compute_candidates(pos_hidden)
    assert all(15 in row for row in full_ids.tolist())


def test_hot_fp8_head_selects_values_and_scales_row_aligned(monkeypatch):
    """An FP8 draft row is meaningless without its own per-row scale, so values
    and scales must follow the same permutation, and the target head keeps its
    full width and its own scales."""
    _use_cpu_top_k(monkeypatch)
    _tp1(monkeypatch)
    hot_ids = [10, 1, 7, 4, 11]
    head = _rowwise_head(12, 4, seed=3)
    snapshot = _snapshot(head)
    target_scale = rowwise_scale_of(head.weight)
    assert target_scale is not None
    hidden = torch.randn(CPU_SAFE_ROWS, 4, dtype=torch.bfloat16)
    model = _DraftStub(head, top_k=3)

    model.install_hot_candidate_head(head, torch.tensor(hot_ids))

    selected = model.candidate_head.weight
    assert selected.dtype is torch.float8_e4m3fn
    assert selected.shape == (len(hot_ids), 4)
    selected_scale = rowwise_scale_of(selected)
    assert selected_scale is not None
    torch.testing.assert_close(selected_scale, target_scale[hot_ids])
    torch.testing.assert_close(
        dequantize_rowwise_weight(selected),
        dequantize_rowwise_weight(head.weight)[hot_ids],
    )

    candidate_ids, unary_logits = model.compute_candidates(hidden)
    dense = dequantize_rowwise_weight(head.weight, torch.float32).to(torch.bfloat16)
    expected_logits, expected_pos = torch.topk(
        (hidden.bfloat16() @ dense.T).float()[:, hot_ids], 3, dim=-1
    )
    torch.testing.assert_close(candidate_ids, torch.tensor(hot_ids)[expected_pos])
    torch.testing.assert_close(unary_logits, expected_logits, rtol=1e-2, atol=1e-2)

    # No requantization and no BF16 duplicate: same tensor, same dtype, same
    # scale object, same full row count.
    assert head.weight.dtype is torch.float8_e4m3fn
    assert head.weight.shape == (12, 4)
    assert rowwise_scale_of(head.weight) is target_scale
    _assert_untouched(head, snapshot)


def test_hot_anchor_and_q_reach_the_real_acceptance_scatter(monkeypatch):
    """Full-vocabulary codebooks stay: a verified anchor outside the hot set
    still scores, the walk emits only hot-set tokens, and the real
    ``_selector_sampling_accept`` scatter carries each row's full q mass at the
    ORIGINAL global ids of the full target vocabulary."""
    _use_cpu_top_k(monkeypatch)
    _tp1(monkeypatch)
    vocab, hot_ids, top_k, slots = 16, [13, 3, 0, 9, 6], 4, 3
    head = _dense_head(vocab, 4, seed=11)
    model = _DraftStub(head, top_k=top_k)
    model.install_hot_candidate_head(head, torch.tensor(hot_ids))
    torch.manual_seed(5)
    selector = model.candidate_selector = CandidateSelector(
        hidden_size=4, vocab_size=vocab, state_rank=2, top_k=top_k
    )
    with torch.no_grad():
        selector.predecessor_codebook.normal_()
        selector.successor_codebook.normal_()
        selector.hidden_projection.weight.normal_()

    hidden = torch.randn(2, slots, 4)
    # Anchor 15 is outside the map: the codebooks are full width, so it scores.
    anchor_token_ids = torch.tensor([15, hot_ids[0]], dtype=torch.int64)
    # The one lattice helper drives both the graph-folded sampler and the eager
    # fallback, so this covers the two paths' shared candidate contract.
    candidate_ids, scores = worker_mod._selector_lattice(
        model, hidden, anchor_token_ids
    )
    assert candidate_ids.shape == (2, slots, top_k)
    assert scores.shape == (2, slots, top_k, top_k)
    assert torch.isfinite(scores).all()
    assert set(candidate_ids.flatten().tolist()) <= set(hot_ids)

    tokens, q_rows = selector.sample_path(
        candidate_ids=candidate_ids,
        scores=scores,
        uniforms=torch.rand(2, slots, dtype=torch.float32),
        temperatures=torch.ones(2, dtype=torch.float32),
        greedy_mask=torch.tensor([True, False]),
    )
    assert set(tokens.flatten().tolist()) <= set(hot_ids)

    seen = {}

    def _record_accept(**kwargs):
        probs = kwargs["draft_probs"]
        assert probs.shape == (2, slots, vocab), "dense q must span the full vocab"
        seen["probs"] = probs.clone()
        return (
            torch.ones(2, dtype=torch.int64),
            torch.zeros(2, dtype=torch.int64),
            None,
        )

    monkeypatch.setattr(worker_mod, "accept_sampling", _record_accept)
    worker = _worker_shim(selector=selector, draft_model=model)
    # candidates carry the anchor at position 0, so block width is slots + 1.
    accept_len, bonus = worker_mod.DFlashWorkerV2._selector_sampling_accept(
        worker,
        candidates=torch.full((2, slots + 1), 1, dtype=torch.int64),
        next_token_logits=torch.randn(2, vocab),
        candidate_ids=candidate_ids,
        q_rows=q_rows,
        sampling_info=SimpleNamespace(is_all_greedy=False),
        draft_input=None,
    )
    assert accept_len.shape == (2,) and bonus.shape == (2,)
    probs = seen["probs"]
    for row in range(2):
        for edge in range(slots):
            mass = probs[row, edge]
            ids = candidate_ids[row, edge]
            assert float(mass.sum()) == pytest.approx(1.0, abs=1e-5)
            # Every unit of q sits on exactly the global candidate ids -- including
            # the greedy row where the walk is a point mass, so per-entry
            # positivity would be the wrong claim.
            torch.testing.assert_close(mass[ids], q_rows[row, edge])
            outside = mass.clone()
            outside[ids] = 0.0
            assert float(outside.sum()) < 1e-6
    # The finally-clear returns the reusable buffer to zero for the next step.
    assert float(worker._draft_probs_buf[:2].abs().sum()) == 0.0


def test_map_is_installed_at_startup_even_when_folding_and_graphs_are_off(
    monkeypatch, tmp_path
):
    """Allocation and map validation happen once, before graph setup, and are
    reused by the eager path: with the fold screen switched off (and likewise
    when the decode graph is disabled or memory is low) nothing is loaded or
    allocated per request, and an unusable map still fails at startup."""
    map_path = tmp_path / "hot-ids.pt"
    hot_ids = [11, 2, 5, 0]
    torch.save(hot_ids, map_path)
    head = _rowwise_head(12, 4, seed=9)
    snapshot = _snapshot(head)
    model = _DraftStub(None, top_k=3)
    worker = _worker_shim(selector=object(), draft_model=model, lm_head=head)
    _patch_env(monkeypatch, str(map_path), eager_sampler=True)
    monkeypatch.setattr(
        worker_mod,
        "get_exec",
        lambda: SimpleNamespace(
            graph=SimpleNamespace(
                cuda_graph_config=SimpleNamespace(decode=SimpleNamespace(bs=[1]))
            )
        ),
    )
    built = []
    monkeypatch.setattr(
        worker_mod, "_SelectorDraftSampler", lambda **kwargs: built.append(1)
    )

    worker_mod.DFlashWorkerV2.alloc_memory_pool(worker)

    installed = model.candidate_head
    assert installed is not None and installed.width == len(hot_ids)
    assert model.lm_head is head
    assert installed.weight.shape == (len(hot_ids), 4)

    # SGLANG_DFLASH_EAGER_DRAFT_SAMPLER=1 keeps the sampler eager: the fold
    # screen must not build a sampler, yet the installed head stays as it was.
    assert worker_mod.DFlashWorkerV2._maybe_build_draft_sampler(worker) is None
    assert built == []
    assert model.candidate_head is installed
    assert installed.token_ids.tolist() == hot_ids
    _assert_untouched(head, snapshot)

    # A second startup pass neither reloads the map nor reallocates the head.
    def _explode(_path):
        raise AssertionError("token map loaded more than once")

    monkeypatch.setattr(worker_mod, "load_token_map", _explode)
    worker_mod.DFlashWorkerV2.alloc_memory_pool(worker)
    worker_mod.DFlashWorkerV2._install_selector_head(worker, head)
    assert model.candidate_head is installed


def test_unusable_hot_maps_are_rejected_before_any_allocation(monkeypatch):
    """Duplicates, narrow maps, and non-integer/2-D maps are refused up front.
    ``load_token_map`` and ``shared_hot_lm_head`` do not reject duplicates, and a
    repeated id would collapse the shortlist so the scattered q sums to less
    than 1 per row -- so the map, not the acceptance path, has to refuse it."""
    _tp1(monkeypatch)
    head = _dense_head(8, 4, seed=17)
    snapshot = _snapshot(head)

    cases = [
        (torch.tensor([5, 5, 2, 1]), "duplicates"),
        (torch.tensor([4, 2]), "top_k"),
        (torch.tensor([[3, 1], [0, 7]]), "one-dimensional"),
        (torch.tensor([3.0, 1.0, 7.0]), "integer"),
        (torch.tensor([3, 99, 5]), "not rows of this rank's lm_head"),
    ]
    for ids, expect in cases:
        model = _DraftStub(head, top_k=3)
        with pytest.raises(ValueError, match=expect):
            model.install_hot_candidate_head(head, ids)
        assert model.candidate_head is None
    _assert_untouched(head, snapshot)


def test_no_token_map_keeps_the_full_target_head_and_full_width_candidates(monkeypatch):
    """Preservation (27B's profile): with the option unset the composite just
    points the draft at the full target head, installs no reduced head, and
    candidates still come from the full org vocab."""
    _use_cpu_top_k(monkeypatch)
    _tp1(monkeypatch)
    head = _dense_head(12, 4, seed=13)
    hidden = torch.randn(4, 4, dtype=torch.bfloat16)
    model = _DraftStub(None, top_k=3)
    worker = _worker_shim(selector=object(), draft_model=model, lm_head=head)
    _patch_env(monkeypatch, None)

    worker_mod.DFlashWorkerV2.alloc_memory_pool(worker)

    assert model.lm_head is head
    assert model.candidate_head is None
    candidate_ids, unary_logits = model.compute_candidates(hidden)
    expected_logits, expected_ids = torch.topk(
        (hidden.bfloat16() @ head.weight.T).float(), 3, dim=-1
    )
    torch.testing.assert_close(candidate_ids, expected_ids)
    torch.testing.assert_close(unary_logits, expected_logits)


def test_map_on_a_draft_without_a_selector_fails_loudly(monkeypatch):
    """The hot head replaces the DFlash2 candidate projection only; a plain
    DFLASH draft has no selector, so the option must not be accepted silently."""
    _patch_env(monkeypatch, "configs/hot.pt")
    with pytest.raises(ValueError, match="selector"):
        worker_mod.DFlashWorkerV2._require_selector_for_token_map(None)
    worker_mod.DFlashWorkerV2._require_selector_for_token_map(object())


@pytest.mark.parametrize(
    "head,expect",
    [
        (
            SimpleNamespace(
                weight=torch.empty(8, 4, dtype=torch.int8),
                quant_method=type(
                    "PackedMethod", (), {"apply": lambda *a, **k: None}
                )(),
            ),
            "needs a dense or rowwise-FP8",
        ),
        (
            SimpleNamespace(
                weight=torch.empty(8, 4, dtype=torch.float8_e4m3fn),
                quant_method=UnquantizedLinearMethod(),
            ),
            "needs a dense or rowwise-FP8",
        ),
    ],
)
def test_unsupported_head_representation_with_a_map_fails_loudly(head, expect):
    model = _DraftStub(head, top_k=3)
    with pytest.raises(RuntimeError, match=expect):
        model.install_hot_candidate_head(head, torch.tensor([3, 1, 0, 5]))
    assert model.candidate_head is None


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
