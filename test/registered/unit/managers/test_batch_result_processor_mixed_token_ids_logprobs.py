"""Mixed batches where only some requests ask for token_ids_logprob.

RC repro (Flash-Next DFlash2, /generate score request with
``token_ids_logprob=[28, 29, 12766]`` batched with ordinary chat requests):
``get_token_ids_logprobs_raw`` at DECODE under ``no_copy_to_cpu=True`` returns
a Tensor row per requested id for requesters but a bare ``[]`` for every
non-requester. Both normalization passes --
``SchedulerBatchResultProcessor._normalize_decode_outputs`` (decode) and
``.move_logprobs_to_cpu`` (prefill result handling) -- convert every row with
``v.tolist()``, so one ordinary request sharing the batch raised
``AttributeError: 'list' object has no attribute 'tolist'`` and killed the
scheduler for every request in the batch. The producer must emit
``logprobs.new_empty(0)`` for non-requester rows in tensor mode, the same
contract ``get_token_ids_logprobs_batch_optimized`` (prefill-only scoring) and
the MLX ``_assemble_logprob_output`` already honor; the plain-list behavior
when ``no_copy_to_cpu=False`` is unchanged.
"""

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.layers.logprob_processor import (
    LogprobStage,
    OutputLogprobProcessor,
    compute_spec_logprobs,
    get_token_ids_logprobs_raw,
)
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.managers.scheduler_components.batch_result_processor import (
    SchedulerBatchResultProcessor,
)
from sglang.srt.managers.utils import GenerationBatchResult
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

VOCAB = 40
# Unsorted, distinct ids: any value/id misordering shows up immediately.
SCORE_IDS = [3, 17, 39]
OTHER_IDS = [31, 0]


class _SpecAlgorithm:
    def __init__(self, is_none: bool):
        self._is_none = is_none

    def is_none(self) -> bool:
        return self._is_none


class _DecodeForwardMode:
    def is_decode(self) -> bool:
        return True

    def is_extend(self) -> bool:
        return False


def _make_batch(reqs, token_ids_logprobs, top_logprobs_nums, spec=True):
    # Only the attributes the logprob producers/consumers touch.
    return SimpleNamespace(
        reqs=reqs,
        seq_lens=[1] * len(reqs),
        return_logprob=True,
        has_grammar=False,
        forward_mode=_DecodeForwardMode(),
        spec_algorithm=_SpecAlgorithm(not spec),
        top_logprobs_nums=top_logprobs_nums,
        token_ids_logprobs=token_ids_logprobs,
        sampling_info=SimpleNamespace(is_all_greedy=True, temperatures=None),
    )


def _make_req(rid, token_ids_logprob, top_logprobs_num=0):
    sp = SamplingParams(max_new_tokens=256, temperature=0)
    sp.normalize(None)
    req = Req(
        rid=rid,
        origin_input_text="",
        origin_input_ids=[1, 2, 3],
        sampling_params=sp,
        return_logprob=True,
        token_ids_logprob=token_ids_logprob,
        top_logprobs_num=top_logprobs_num,
    )
    req.kv_committed_len = 0
    return req


def _make_processor():
    return SchedulerBatchResultProcessor(
        is_generation=True,
        disaggregation_mode=None,
        enable_overlap=False,
        enable_overlap_mlx=False,
        model_config=SimpleNamespace(think_end_ids=None),
        token_to_kv_pool_allocator=None,
        tree_cache=None,
        hisparse_coordinator=None,
        req_to_token_pool=None,
        decode_offload_manager=None,
        metrics_collector=None,
        metrics_reporter=SimpleNamespace(),
        draft_worker=None,
        model_worker=SimpleNamespace(on_verify_complete_cpu=lambda *a, **k: None),
        logprob_result_processor=None,
        output_streamer=SimpleNamespace(),
        abort_request=lambda *a, **k: None,
    )


def _reference(logits):
    return torch.log_softmax(logits.float(), dim=-1)


class _DummyCopyDone:
    """Stands in for the CUDA event overlap scheduling records around the
    D2H copy; a CPU-only box has no device to record on. Covers only the
    transport bookkeeping, never the conversions under test."""

    def record(self) -> None:
        pass

    def synchronize(self) -> None:
        pass


class TestDecodeProducerTensorRows(CustomTestCase):
    """Direct producer contract: in tensor mode every DECODE row is tolist-
    able, including non-requester and explicitly-empty-id rows."""

    def _check_tensor_mode(self, logprobs, token_ids_logprobs_list):
        ref = _reference(logprobs)
        vals, idxs = get_token_ids_logprobs_raw(
            logprobs,
            token_ids_logprobs_list,
            stage=LogprobStage.DECODE,
            no_copy_to_cpu=True,
        )
        self.assertEqual(len(vals), len(token_ids_logprobs_list))
        for i, token_ids in enumerate(token_ids_logprobs_list):
            self.assertIsInstance(
                vals[i],
                torch.Tensor,
                f"row {i} of a no_copy_to_cpu batch must be a tensor, got "
                f"{type(vals[i]).__name__}; consumers call .tolist() on it",
            )
            if token_ids:
                self.assertEqual(idxs[i], list(token_ids))
                self.assertEqual(vals[i].tolist(), ref[i, token_ids].tolist())
            else:
                self.assertEqual(idxs[i], [])
                self.assertEqual(vals[i].numel(), 0)
        return vals

    def test_non_requester_and_empty_rows_are_tensors(self):
        torch.manual_seed(0)
        logprobs = _reference(torch.randn(3, VOCAB))
        # Non-requester first, requester last.
        self._check_tensor_mode(logprobs, [None, SCORE_IDS, None])
        # Requester first, non-requester last.
        self._check_tensor_mode(logprobs, [SCORE_IDS, None, OTHER_IDS])
        # Explicitly-empty request: opt-in with no ids -> empty tensor row.
        self._check_tensor_mode(logprobs, [[], SCORE_IDS, OTHER_IDS])
        self._check_tensor_mode(logprobs, [SCORE_IDS, [], None])

    def test_plain_mode_keeps_empty_lists(self):
        """no_copy_to_cpu=False callers keep the old list-shaped rows."""
        torch.manual_seed(0)
        logprobs = _reference(torch.randn(2, VOCAB))
        vals, idxs = get_token_ids_logprobs_raw(
            logprobs, [None, SCORE_IDS], stage=LogprobStage.DECODE
        )
        self.assertEqual(vals[0], [])
        self.assertEqual(idxs[0], [])
        self.assertEqual(vals[1], logprobs[1, SCORE_IDS].tolist())
        self.assertEqual(idxs[1], SCORE_IDS)


class TestSpecDecodeMixedBatch(CustomTestCase):
    """Full overlap-shaped spec-v2 chain: real producer -> real
    GenerationBatchResult.copy_to_cpu -> real _normalize_decode_outputs ->
    real _apply_decode_logprobs, for DFlash block4 and spec-v2 block8."""

    def _run(self, stride, requester_first):
        bs = 2
        torch.manual_seed(11)
        logits = torch.randn(bs * stride, VOCAB)
        ref = _reference(logits)
        # The score request accepts 2 tokens, the plain chat request 1.
        token_ids_per_req = [SCORE_IDS, None] if requester_first else [None, SCORE_IDS]
        accept_lens = [2, 1] if requester_first else [1, 2]
        reqs = [_make_req(f"r{i}", ids) for i, ids in enumerate(token_ids_per_req)]
        batch = _make_batch(reqs, list(token_ids_per_req), [0, 0], spec=True)
        logits_output = LogitsProcessorOutput(next_token_logits=logits)
        predict = torch.arange(bs * stride, dtype=torch.long) % VOCAB

        compute_spec_logprobs(batch, logits_output, predict, chain_stride=stride)

        result = GenerationBatchResult(
            logits_output=logits_output,
            next_token_ids=predict,
            accept_lens=torch.tensor(accept_lens, dtype=torch.long),
            speculative_num_draft_tokens=stride,
        )
        result.copy_done = _DummyCopyDone()

        # Overlap scheduling moves the per-row tensors to CPU one by one and
        # leaves the bare-list non-requester rows untouched -> the crash site.
        result.copy_to_cpu(return_logprob=True, return_hidden_states=False)

        proc = _make_processor()
        next_token_ids, next_token_logprobs = proc._normalize_decode_outputs(
            batch=batch,
            result=result,
            logits_output=logits_output,
            next_token_ids=result.next_token_ids,
        )
        for i, req in enumerate(reqs):
            proc._apply_decode_logprobs(
                req=req,
                i=i,
                batch=batch,
                next_token_id=next_token_ids[i],
                next_token_logprobs=next_token_logprobs,
                logits_output=logits_output,
            )

        requester = 0 if requester_first else 1
        plain = 1 - requester
        # Requested values/ids preserved, in request order, on the requester's
        # own rows (row i * stride + j = accepted token j of req i).
        self.assertEqual(
            reqs[requester].logprob.output_token_ids_logprobs_val,
            [ref[requester * stride + j, SCORE_IDS].tolist() for j in range(2)],
        )
        self.assertEqual(
            reqs[requester].logprob.output_token_ids_logprobs_idx,
            [list(SCORE_IDS), list(SCORE_IDS)],
        )
        # Non-requester: empty rows survived normalization as empty lists and
        # are not attached to the request that never asked for them.
        self.assertEqual(reqs[plain].logprob.output_token_ids_logprobs_val, [])
        for j in range(stride):
            row = logits_output.next_token_token_ids_logprobs_val[plain * stride + j]
            self.assertIsInstance(row, list)
            self.assertEqual(row, [])
        # Accepted-token logprobs/ids untouched for both reqs.
        for i in range(bs):
            self.assertEqual(
                reqs[i].logprob.output_token_logprobs_idx,
                predict[i * stride : i * stride + accept_lens[i]].tolist(),
            )
            self.assertEqual(
                reqs[i].logprob.output_token_logprobs_val,
                [
                    ref[i * stride + j, predict[i * stride + j]].item()
                    for j in range(accept_lens[i])
                ],
            )

    def test_block4_requester_first(self):
        self._run(stride=4, requester_first=True)

    def test_block4_plain_request_first(self):
        self._run(stride=4, requester_first=False)

    def test_block8_requester_first(self):
        self._run(stride=8, requester_first=True)

    def test_block8_plain_request_first(self):
        self._run(stride=8, requester_first=False)

    def test_all_requests_requesting_stay_aligned(self):
        stride, bs = 4, 2
        torch.manual_seed(3)
        logits = torch.randn(bs * stride, VOCAB)
        ref = _reference(logits)
        reqs = [_make_req("r0", SCORE_IDS), _make_req("r1", OTHER_IDS)]
        batch = _make_batch(reqs, [SCORE_IDS, OTHER_IDS], [0, 0], spec=True)
        logits_output = LogitsProcessorOutput(next_token_logits=logits)
        predict = torch.arange(bs * stride, dtype=torch.long) % VOCAB
        compute_spec_logprobs(batch, logits_output, predict, chain_stride=stride)
        result = GenerationBatchResult(
            logits_output=logits_output,
            next_token_ids=predict,
            accept_lens=torch.tensor([1, 1], dtype=torch.long),
            speculative_num_draft_tokens=stride,
        )
        result.copy_done = _DummyCopyDone()
        result.copy_to_cpu(return_logprob=True, return_hidden_states=False)
        proc = _make_processor()
        next_token_ids, next_token_logprobs = proc._normalize_decode_outputs(
            batch=batch,
            result=result,
            logits_output=logits_output,
            next_token_ids=result.next_token_ids,
        )
        for i, req in enumerate(reqs):
            proc._apply_decode_logprobs(
                req=req,
                i=i,
                batch=batch,
                next_token_id=next_token_ids[i],
                next_token_logprobs=next_token_logprobs,
                logits_output=logits_output,
            )
        self.assertEqual(
            reqs[0].logprob.output_token_ids_logprobs_val,
            [ref[0, SCORE_IDS].tolist()],
        )
        self.assertEqual(
            reqs[1].logprob.output_token_ids_logprobs_val,
            [ref[stride, OTHER_IDS].tolist()],
        )

    def test_no_requester_leaves_fields_unset(self):
        stride, bs = 4, 2
        torch.manual_seed(5)
        logits = torch.randn(bs * stride, VOCAB)
        reqs = [_make_req("r0", None), _make_req("r1", None)]
        batch = _make_batch(reqs, [None, None], [0, 0], spec=True)
        logits_output = LogitsProcessorOutput(next_token_logits=logits)
        predict = torch.arange(bs * stride, dtype=torch.long) % VOCAB
        compute_spec_logprobs(batch, logits_output, predict, chain_stride=stride)
        self.assertIsNone(logits_output.next_token_token_ids_logprobs_val)
        self.assertIsNone(logits_output.next_token_token_ids_logprobs_idx)
        result = GenerationBatchResult(
            logits_output=logits_output,
            next_token_ids=predict,
            accept_lens=torch.tensor([1, 1], dtype=torch.long),
            speculative_num_draft_tokens=stride,
        )
        result.copy_done = _DummyCopyDone()
        result.copy_to_cpu(return_logprob=True, return_hidden_states=False)
        proc = _make_processor()
        proc._normalize_decode_outputs(
            batch=batch,
            result=result,
            logits_output=logits_output,
            next_token_ids=result.next_token_ids,
        )
        self.assertIsNone(logits_output.next_token_token_ids_logprobs_val)


class TestNonSpecDecodeAndPrefillConversion(CustomTestCase):
    """The sampler's DECODE producer also feeds the non-spec decode
    normalization and the prefill-result move_logprobs_to_cpu; a mixed batch
    must survive both."""

    def _mixed_output(self, requester_first, top_logprobs_nums):
        torch.manual_seed(23)
        logprobs = _reference(torch.randn(2, VOCAB))
        token_ids = [SCORE_IDS, None] if requester_first else [None, SCORE_IDS]
        reqs = [
            _make_req(f"r{i}", ids, top_logprobs_num=k)
            for i, (ids, k) in enumerate(zip(token_ids, top_logprobs_nums))
        ]
        processor = OutputLogprobProcessor()
        result = processor.compute_logprobs(
            logprobs.clone(),
            list(top_logprobs_nums),
            list(token_ids),
            torch.tensor([7, 9], dtype=torch.long),
        )
        logits_output = LogitsProcessorOutput(next_token_logits=torch.randn(2, VOCAB))
        result.write_output_to(logits_output)
        return logprobs, reqs, token_ids, logits_output

    def test_decode_normalization_mixed_both_orders(self):
        for requester_first in (True, False):
            with self.subTest(requester_first=requester_first):
                logprobs, reqs, token_ids, logits_output = self._mixed_output(
                    requester_first, [0, 0]
                )
                batch = _make_batch(reqs, list(token_ids), [0, 0], spec=False)
                proc = _make_processor()
                next_token_ids, next_token_logprobs = proc._normalize_decode_outputs(
                    batch=batch,
                    result=SimpleNamespace(),
                    logits_output=logits_output,
                    next_token_ids=torch.tensor([7, 9], dtype=torch.long),
                )
                for i, req in enumerate(reqs):
                    proc._apply_decode_logprobs(
                        req=req,
                        i=i,
                        batch=batch,
                        next_token_id=next_token_ids[i],
                        next_token_logprobs=next_token_logprobs,
                        logits_output=logits_output,
                    )
                requester = 0 if requester_first else 1
                self.assertEqual(
                    reqs[requester].logprob.output_token_ids_logprobs_val,
                    [logprobs[requester, SCORE_IDS].tolist()],
                )
                self.assertEqual(
                    reqs[requester].logprob.output_token_ids_logprobs_idx,
                    [list(SCORE_IDS)],
                )
                self.assertEqual(
                    reqs[1 - requester].logprob.output_token_ids_logprobs_val,
                    [],
                )

    def test_prefill_move_logprobs_mixed(self):
        # Prefill batches sample their first token through the same sampler
        # path; process_batch_result_prefill then converts the same fields.
        for requester_first in (True, False):
            with self.subTest(requester_first=requester_first):
                logprobs, reqs, token_ids, logits_output = self._mixed_output(
                    requester_first, [2, 0]
                )
                batch = _make_batch(reqs, list(token_ids), [2, 0], spec=False)
                proc = _make_processor()
                proc.move_logprobs_to_cpu(batch=batch, logits_output=logits_output)
                requester = 0 if requester_first else 1
                self.assertEqual(
                    logits_output.next_token_token_ids_logprobs_val[requester],
                    logprobs[requester, SCORE_IDS].tolist(),
                )
                self.assertEqual(
                    logits_output.next_token_token_ids_logprobs_val[1 - requester],
                    [],
                )
                # Mixed top-k rows were already tensor-shaped (k=0 rows
                # sliced to empty tensors) even before the fix; pin that.
                # top_logprobs_nums is positional: row 0 asks k=2, row 1 k=0.
                self.assertEqual(logits_output.next_token_top_logprobs_val[1], [])
                self.assertEqual(len(logits_output.next_token_top_logprobs_val[0]), 2)

    def test_scoring_producer_contract_pinned(self):
        """compute_logprobs_only (prefill-only scoring) already returned
        tolist-able rows for every entry; pin it next to the fixed producer."""
        torch.manual_seed(31)
        logits = torch.randn(2, VOCAB)
        ref = _reference(logits)
        processor = OutputLogprobProcessor()
        result = processor.compute_logprobs_only(
            next_token_logits=logits,
            top_logprobs_nums=[0, 1],
            token_ids_logprobs=[[], SCORE_IDS],
            preprocess_fn=lambda x: x,
        )
        logits_output = LogitsProcessorOutput(next_token_logits=logits)
        result.write_output_to(logits_output)
        vals = logits_output.next_token_token_ids_logprobs_val
        self.assertEqual([v.tolist() for v in vals], [[], ref[1, SCORE_IDS].tolist()])
        self.assertEqual(
            logits_output.next_token_token_ids_logprobs_idx, [[], list(SCORE_IDS)]
        )


if __name__ == "__main__":
    unittest.main()
