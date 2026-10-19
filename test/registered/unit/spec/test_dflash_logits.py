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
    _grouped_conv,
    _radix_topk,
)
from sglang.srt.speculative.dflash_utils import parse_dflash_draft_config
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=35, suite="base-a-test-cpu")


def test_dflash_unary_logit_transform():
    logits = torch.tensor([[-100.0, 0.0, 100.0]], dtype=torch.bfloat16)
    for fields in ({}, {"output_multiplier": 0.2, "final_logit_softcapping": 20.0}):
        config = parse_dflash_draft_config(
            draft_hf_config={
                "num_hidden_layers": 5,
                "dflash_config": {
                    "selector_rank": 256,
                    "selector_top_k": 16,
                    **fields,
                },
            }
        )
        actual = DFlash2DraftModel._transform_unary_logits(
            SimpleNamespace(draft_config=config), logits
        )
        expected = logits.float() * config.output_multiplier
        if config.final_logit_softcapping is not None:
            expected = torch.tanh(expected / config.final_logit_softcapping)
            expected *= config.final_logit_softcapping
        torch.testing.assert_close(actual, expected)


def test_selector_greedy_row_walk_is_deterministic_in_a_mixed_batch():
    """A greedy row walks the argmax, so the q it hands verify has to be the point
    mass there. Greedy reaches the selector as top_k=1 with the temperature reset
    to 1.0, so a softmax q stays a real distribution and verify would
    rejection-sample a deterministic request against it. The row must also not
    depend on who else is in the batch."""
    selector = CandidateSelector(hidden_size=4, vocab_size=16, state_rank=2, top_k=4)
    torch.manual_seed(1)
    candidate_ids = torch.randint(0, 16, (2, 3, 4))
    scores = torch.randn(2, 3, 4, 4)
    uniforms = torch.tensor([[0.2, 0.7, 0.4], [0.8, 0.1, 0.6]])
    temperatures = torch.tensor([1.0, 0.7])
    greedy_mask = torch.tensor([True, False])

    mixed_tokens, mixed_q = selector.sample_path(
        candidate_ids=candidate_ids,
        scores=scores,
        uniforms=uniforms,
        temperatures=temperatures,
        greedy_mask=greedy_mask,
    )
    assert torch.all((mixed_q[0] == 0) | (mixed_q[0] == 1))
    for row in range(2):
        tokens, q_rows = selector.sample_path(
            candidate_ids=candidate_ids[row : row + 1],
            scores=scores[row : row + 1],
            uniforms=uniforms[row : row + 1],
            temperatures=temperatures[row : row + 1],
            greedy_mask=greedy_mask[row : row + 1],
        )
        torch.testing.assert_close(mixed_tokens[row], tokens[0])
        torch.testing.assert_close(mixed_q[row], q_rows[0])


def test_selector_rejects_a_quantized_target_lm_head():
    """The candidate matmuls read the lm_head weight directly, so a packed or
    absent weight would be read as if it were dense."""
    model = SimpleNamespace(
        lm_head=SimpleNamespace(weight=torch.empty(8, 4, dtype=torch.int8)),
        candidate_selector=SimpleNamespace(top_k=4),
    )
    with pytest.raises(RuntimeError, match="requires a dense"):
        DFlash2DraftModel.compute_candidates(model, torch.randn(2, 4))


def _flashinfer_contract_topk(scores, k, sorted=False, deterministic=False):
    """Stand-in for flashinfer.top_k pinning its call contract: contiguous
    input (its CHECK_INPUT) and the explicit sorted/deterministic flags
    _radix_topk relies on (the real kernel defaults both to False)."""
    assert scores.is_contiguous()
    assert sorted and deterministic
    return torch.topk(scores, k, dim=-1)


class _FakeQuantMethod:
    """Projects through a captured dense weight, asserting the packed-head
    call contract (packed dtype, no bias). The padded tail comes out as
    dominant garbage so a masking regression surfaces as wrong candidates."""

    def __init__(self, dense_weight, num_padded):
        self.dense_weight = dense_weight
        self.num_padded = num_padded
        self.called = False

    def apply(self, layer, x, bias):
        self.called = True
        assert layer.weight.dtype == torch.int8
        assert bias is None
        logits = torch.matmul(x, self.dense_weight.T)
        pad = logits.new_full((logits.shape[0], self.num_padded), 100.0)
        full = torch.cat([logits, pad], dim=-1)
        # A strided view, like a kernel writing into a wider workspace: the
        # projection must materialize it before flashinfer's radix top-k.
        return torch.stack([full, full], dim=-1)[..., 0]


def test_selector_projects_a_quantized_target_lm_head_through_its_quant_method(
    monkeypatch,
):
    """Packed head weights must be projected through their quantization method,
    with the padded-vocab tail masked out of the top-k on contiguous logits:
    flashinfer's radix top-k rejects non-contiguous input, so a plain crop view
    would fail at capture on any padded vocab."""
    torch.manual_seed(0)
    hidden = torch.randn(2, 4)
    dense_weight = torch.randn(6, 4)

    quant_method = _FakeQuantMethod(dense_weight, num_padded=2)
    lm_head = SimpleNamespace(
        # Mimic a 2:1 packed head and two padded vocabulary rows.
        weight=torch.empty(8, 2, dtype=torch.int8),
        quant_method=quant_method,
        org_vocab_size=6,
    )
    model = SimpleNamespace(
        lm_head=lm_head,
        candidate_selector=SimpleNamespace(top_k=4),
        _transform_unary_logits=lambda logits: logits.float(),
    )
    monkeypatch.setattr(
        "sglang.srt.models.dflash.get_parallel",
        lambda: SimpleNamespace(tp_size=1),
    )
    monkeypatch.setattr(
        "sglang.srt.models.dflash._flashinfer_top_k", _flashinfer_contract_topk
    )

    candidate_ids, unary_logits = DFlash2DraftModel.compute_candidates(model, hidden)

    expected_logits, expected_ids = torch.topk(
        torch.matmul(hidden, dense_weight.T), 4, dim=-1
    )
    assert quant_method.called
    torch.testing.assert_close(candidate_ids, expected_ids)
    torch.testing.assert_close(unary_logits, expected_logits)


def test_selector_gathers_global_candidates_across_vocab_shards(monkeypatch):
    """Pins the TP gather contract on the quantized path: the per-shard
    org-vocab restriction, the global id offset, and the fp32 cast before the
    all-gather -- a
    regression in any of them returns wrong global candidates only under TP,
    which no single-rank test observes."""
    torch.manual_seed(0)
    k = 4
    # bf16 like production: makes the fp32 upcast before the gather observable.
    hidden = torch.randn(2, 4, dtype=torch.bfloat16)
    full_weight = torch.randn(12, 4, dtype=torch.bfloat16)  # org vocab 12, 6+6

    # This process plays rank 1 of tp=2: org rows 6..12 as local rows 0..6,
    # plus two dominant padded columns that must never reach the candidates.
    quant_method = _FakeQuantMethod(full_weight[6:], num_padded=2)
    lm_head = SimpleNamespace(
        weight=torch.empty(8, 2, dtype=torch.int8),
        quant_method=quant_method,
        shard_indices=SimpleNamespace(num_org_elements=6, org_vocab_start_index=6),
    )
    model = SimpleNamespace(
        lm_head=lm_head,
        candidate_selector=SimpleNamespace(top_k=k),
        _transform_unary_logits=lambda logits: logits.float(),
    )

    # Rank 0's gathered contribution, synthesized from the reference weights.
    rank0_vals, rank0_ids = torch.topk(
        torch.matmul(hidden, full_weight[:6].T), k, dim=-1
    )

    def fake_all_gather(x, dim):
        if x.is_floating_point():
            assert x.dtype == torch.float32
            return torch.cat([rank0_vals.float(), x], dim=dim)
        return torch.cat([rank0_ids.long(), x], dim=dim)

    monkeypatch.setattr(
        "sglang.srt.models.dflash.get_parallel",
        lambda: SimpleNamespace(tp_size=2),
    )
    monkeypatch.setattr(
        "sglang.srt.models.dflash.tensor_model_parallel_all_gather", fake_all_gather
    )
    monkeypatch.setattr(
        "sglang.srt.models.dflash._flashinfer_top_k", _flashinfer_contract_topk
    )

    candidate_ids, unary_logits = DFlash2DraftModel.compute_candidates(model, hidden)

    expected_logits, expected_ids = torch.topk(
        torch.matmul(hidden, full_weight.T), k, dim=-1
    )
    torch.testing.assert_close(candidate_ids, expected_ids)
    torch.testing.assert_close(unary_logits, expected_logits.float())


def _use_cpu_top_k(monkeypatch):
    """Keep every candidate-projection fixture off the device-bound top-k.

    ``_radix_topk`` prefers an installed FlashInfer kernel, and that kernel asks
    for the compute capability of the tensor's device, so on a runner with
    FlashInfer importable a CPU fixture fails with "device must be a cuda device"
    before anything about the projection is checked. The contract-pinning
    stand-in below is a CPU implementation of the same call, so the rowwise FP8
    numerics, the padded-vocab masking and the global-index assertions are what
    these tests exercise on every runner. No CUDA parity is claimed here.
    """
    monkeypatch.setattr(
        "sglang.srt.models.dflash._flashinfer_top_k", _flashinfer_contract_topk
    )


def _rowwise_fp8_head(dense: torch.Tensor):
    """A target head in the resident Flash-Next representation: BF16 checkpoint
    values quantized once to E4M3 with one FP32 scale per output channel, behind
    the unquantized module method the NVFP4 checkpoint leaves in place."""
    head = nn.Linear(dense.shape[1], dense.shape[0], bias=False, dtype=torch.bfloat16)
    head.weight.data.copy_(dense)
    replace_linear_weight_rowwise_fp8(head)
    head.quant_method = UnquantizedLinearMethod()
    return head


def _dominant_pad(dense: torch.Tensor, org_rows: int) -> torch.Tensor:
    """Padded vocab rows that outscore their org-vocab neighbours, so a projection
    that forgets to restrict the top-k surfaces them as candidates."""
    dense[org_rows:] *= 50.0
    return dense


def _rowwise_reference(hidden, head):
    """The same dequantized matmul the rowwise kernel falls back to, with the
    scale it carries rather than a copy of the original weight."""
    dense = dequantize_rowwise_weight(head.weight, torch.float32).to(torch.bfloat16)
    return (hidden.bfloat16() @ dense.T).float()


def test_candidate_fixtures_pin_their_top_k_against_an_installed_flashinfer(
    monkeypatch,
):
    """The pin above is load-bearing, not decoration.

    On thegrid FlashInfer is installed, so ``_radix_topk`` reaches its kernel and
    that kernel refuses CPU tensors with "device must be a cuda device" — a
    fixture that stopped pinning its top-k would fail there for a reason that has
    nothing to do with the projection under test. This checks both halves: the
    device-bound call really does refuse these tensors, and the helper the
    fixtures use replaces it with a CPU implementation.
    """

    def _device_bound_top_k(scores, k, *, sorted=False, deterministic=False):
        assert sorted and deterministic
        raise ValueError("device must be a cuda device")

    monkeypatch.setattr(
        "sglang.srt.models.dflash._flashinfer_top_k", _device_bound_top_k
    )
    with pytest.raises(ValueError, match="device must be a cuda device"):
        _radix_topk(torch.randn(4, 6), 2)

    _use_cpu_top_k(monkeypatch)
    _radix_topk(torch.randn(4, 6), 2)


def test_selector_projects_the_scaled_rowwise_fp8_target_head(monkeypatch):
    """The accepted Flash-Next head is resident E4M3 plus a per-channel scale, so
    the candidate projection has to run the target's own rowwise kernel on it:
    the scale must stay attached to the weight, the padded tail must stay out of
    the top-k, and no BF16 copy of the head may appear."""
    _use_cpu_top_k(monkeypatch)
    hidden = torch.randn(40, 4, dtype=torch.bfloat16)
    torch.manual_seed(0)
    head = _rowwise_fp8_head(
        _dominant_pad(torch.randn(8, 4, dtype=torch.bfloat16), org_rows=6)
    )
    head.org_vocab_size = 6
    scale_before = rowwise_scale_of(head.weight)
    assert scale_before is not None
    model = SimpleNamespace(
        lm_head=head,
        candidate_selector=SimpleNamespace(top_k=4),
        _transform_unary_logits=lambda logits: logits.float(),
    )
    monkeypatch.setattr(
        "sglang.srt.models.dflash.get_parallel", lambda: SimpleNamespace(tp_size=1)
    )

    candidate_ids, unary_logits = DFlash2DraftModel.compute_candidates(model, hidden)

    expected_logits, expected_ids = torch.topk(
        _rowwise_reference(hidden, head)[:, :6], 4, dim=-1
    )
    torch.testing.assert_close(candidate_ids, expected_ids)
    torch.testing.assert_close(unary_logits, expected_logits, rtol=1e-2, atol=1e-2)
    # The head is still the resident quantized tensor with its own scale: the
    # projection neither dequantized it in place nor swapped in a BF16 copy.
    assert head.weight.dtype is torch.float8_e4m3fn
    assert rowwise_scale_of(head.weight) is scale_before


def test_selector_gathers_global_candidates_across_rowwise_fp8_shards(monkeypatch):
    """Same global-index contract on the rowwise head as on the quantized one:
    per-shard org restriction, shard offset before the gather, fp32 upcast."""
    _use_cpu_top_k(monkeypatch)
    k = 4
    hidden = torch.randn(40, 4, dtype=torch.bfloat16)
    torch.manual_seed(3)
    # This process plays rank 1 of tp=2: org rows 6..12 as local rows 0..6, plus
    # two dominant padded vocabulary rows.
    full = _dominant_pad(torch.randn(14, 4, dtype=torch.bfloat16), org_rows=12)
    rank1_head = _rowwise_fp8_head(full[6:])
    rank0_head = _rowwise_fp8_head(full[:8])
    rank1_head.shard_indices = SimpleNamespace(
        num_org_elements=6, org_vocab_start_index=6
    )
    model = SimpleNamespace(
        lm_head=rank1_head,
        candidate_selector=SimpleNamespace(top_k=k),
        _transform_unary_logits=lambda logits: logits.float(),
    )
    rank0_logits, rank0_ids = torch.topk(
        _rowwise_reference(hidden, rank0_head)[:, :6], k, dim=-1
    )

    def fake_all_gather(x, dim):
        if x.is_floating_point():
            assert x.dtype == torch.float32
            return torch.cat([rank0_logits.float(), x], dim=dim)
        return torch.cat([rank0_ids.long(), x], dim=dim)

    monkeypatch.setattr(
        "sglang.srt.models.dflash.get_parallel", lambda: SimpleNamespace(tp_size=2)
    )
    monkeypatch.setattr(
        "sglang.srt.models.dflash.tensor_model_parallel_all_gather", fake_all_gather
    )

    candidate_ids, unary_logits = DFlash2DraftModel.compute_candidates(model, hidden)

    expected_logits, expected_ids = torch.topk(
        torch.cat(
            [
                _rowwise_reference(hidden, rank0_head)[:, :6],
                _rowwise_reference(hidden, rank1_head)[:, :6],
            ],
            dim=-1,
        ),
        k,
        dim=-1,
    )
    torch.testing.assert_close(candidate_ids, expected_ids)
    torch.testing.assert_close(unary_logits, expected_logits, rtol=1e-2, atol=1e-2)


def test_selector_keeps_the_dense_head_on_the_plain_matmul(monkeypatch):
    """Preservation: an untouched BF16 head must not be routed through the FP8
    kernel, and its candidates keep coming from the row-sliced dense weight."""

    def _no_fp8_kernel(*args, **kwargs):
        raise AssertionError("rowwise FP8 kernel used for a dense head")

    monkeypatch.setattr(
        "sglang.srt.models.dflash.rowwise_fp8_lm_head_logits", _no_fp8_kernel
    )
    _use_cpu_top_k(monkeypatch)
    monkeypatch.setattr(
        "sglang.srt.models.dflash.get_parallel", lambda: SimpleNamespace(tp_size=1)
    )
    torch.manual_seed(1)
    weight = torch.randn(6, 4, dtype=torch.bfloat16)
    hidden = torch.randn(4, 4, dtype=torch.bfloat16)
    model = SimpleNamespace(
        lm_head=SimpleNamespace(
            weight=weight, org_vocab_size=6, quant_method=UnquantizedLinearMethod()
        ),
        candidate_selector=SimpleNamespace(top_k=3),
        _transform_unary_logits=lambda logits: logits.float(),
    )

    candidate_ids, unary_logits = DFlash2DraftModel.compute_candidates(model, hidden)

    expected_logits, expected_ids = torch.topk(
        torch.matmul(hidden, weight.T).float(), 3, dim=-1
    )
    torch.testing.assert_close(candidate_ids, expected_ids)
    torch.testing.assert_close(unary_logits, expected_logits)


def test_worker_folds_a_gate_admitted_quantized_selector_head(monkeypatch):
    """The pre-capture screen decides whether a quantized head reaches the
    graph-folded selector sampler or silently degrades to the eager per-round
    fallback -- a revert there keeps every compute_candidates test green, so
    the admission (and the rejection of an unsupported packed head) needs its
    own guard."""
    from sglang.srt.speculative import dflash_worker_v2 as worker_mod

    built = {}
    monkeypatch.setattr(
        worker_mod,
        "_SelectorDraftSampler",
        lambda **kwargs: built.setdefault("sampler", object()),
    )
    monkeypatch.setattr(
        worker_mod,
        "get_exec",
        lambda: SimpleNamespace(
            graph=SimpleNamespace(
                cuda_graph_config=SimpleNamespace(decode=SimpleNamespace(bs=[1]))
            )
        ),
    )
    quant_head = SimpleNamespace(
        weight=torch.empty(8, 2, dtype=torch.int8),
        quant_method=_FakeQuantMethod(torch.randn(6, 4), num_padded=2),
    )
    worker = SimpleNamespace(
        block_size=8,
        selector=object(),
        ps=SimpleNamespace(tp_rank=0),
        draft_model=SimpleNamespace(lm_head=None),
        device="cpu",
        _target_worker=SimpleNamespace(
            model_runner=SimpleNamespace(model=SimpleNamespace(lm_head=quant_head))
        ),
    )

    sampler = worker_mod.DFlashWorkerV2._maybe_build_draft_sampler(worker)
    assert sampler is built["sampler"]
    assert worker.draft_model.lm_head is quant_head

    # A packed head without an applicable quant method must stay eager.
    worker._target_worker.model_runner.model.lm_head = SimpleNamespace(
        weight=torch.empty(8, 2, dtype=torch.int8)
    )
    worker.draft_model.lm_head = None
    assert worker_mod.DFlashWorkerV2._maybe_build_draft_sampler(worker) is None
    assert worker.draft_model.lm_head is None


def test_worker_folds_the_scaled_rowwise_fp8_selector_head(monkeypatch):
    """Graph admission for the Flash-Next head: the target's own logits path already
    runs the rowwise FP8 kernel under CUDA graphs, so the selector sampler must fold
    it in rather than degrade to the eager per-round fallback -- and an FP8 tensor
    that does not carry its rowwise scale is still refused."""
    from sglang.srt.speculative import dflash_worker_v2 as worker_mod

    built = {}
    monkeypatch.setattr(
        worker_mod,
        "_SelectorDraftSampler",
        lambda **kwargs: built.setdefault("sampler", object()),
    )
    monkeypatch.setattr(
        worker_mod,
        "get_exec",
        lambda: SimpleNamespace(
            graph=SimpleNamespace(
                cuda_graph_config=SimpleNamespace(decode=SimpleNamespace(bs=[1]))
            )
        ),
    )
    head = _rowwise_fp8_head(
        _dominant_pad(torch.randn(8, 4, dtype=torch.bfloat16), org_rows=6)
    )
    worker = SimpleNamespace(
        block_size=8,
        selector=object(),
        ps=SimpleNamespace(tp_rank=0),
        draft_model=SimpleNamespace(lm_head=None),
        device="cpu",
        _target_worker=SimpleNamespace(
            model_runner=SimpleNamespace(model=SimpleNamespace(lm_head=head))
        ),
    )

    sampler = worker_mod.DFlashWorkerV2._maybe_build_draft_sampler(worker)

    assert sampler is built["sampler"]
    assert worker.draft_model.lm_head is head

    # A scale-less FP8 tensor is a different representation, and the dense static
    # matmul would read its codes as weights.
    built.pop("sampler")
    worker.draft_model.lm_head = None
    worker._target_worker.model_runner.model.lm_head = SimpleNamespace(
        weight=torch.empty(8, 4, dtype=torch.float8_e4m3fn)
    )
    assert worker_mod.DFlashWorkerV2._maybe_build_draft_sampler(worker) is None
    assert worker.draft_model.lm_head is None


def test_grouped_conv_supports_runtime_block_sizes():
    """The conv indexes a position inside the block, so it must follow whatever
    block size the worker resolved -- including one that is not a power of two."""
    torch.manual_seed(0)
    groups, group_size, taps = 3, 2, 2
    hidden_size = groups * group_size
    batch_size = 2

    for block_size in (5, 8, 16):
        hidden = torch.randn(batch_size * block_size, hidden_size)
        delta = torch.randn(batch_size * block_size, taps, groups)
        base = torch.randn(taps, hidden_size)

        actual = _grouped_conv(
            hidden, delta, base, block_size, groups, group_size, taps
        )

        expected = torch.empty_like(hidden)
        hidden_3d = hidden.view(batch_size, block_size, groups, group_size)
        delta_4d = delta.view(batch_size, block_size, taps, groups)
        base_3d = base.view(taps, groups, group_size)
        for batch in range(batch_size):
            for position in range(block_size):
                value = torch.zeros(groups, group_size)
                for tap in range(min(taps, position + 1)):
                    coefficient = base_3d[tap] + delta_4d[batch, position, tap, :, None]
                    value += coefficient * hidden_3d[batch, position - tap]
                expected[batch * block_size + position] = value.flatten()
        torch.testing.assert_close(actual, expected)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
