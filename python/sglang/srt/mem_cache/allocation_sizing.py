from __future__ import annotations

from sglang.srt.runtime_context import (
    get_parallel,
    get_schedule,
    get_spec,
    max_speculative_num_draft_tokens,
)


def get_alloc_page_size() -> int:
    # Mirrors _build_token_to_kv_pool_allocator's DCP branch; the platform
    # allocators that skip it page smaller, so this is an upper bound for them.
    return get_schedule().page_size * get_parallel().attn_dcp_size


def get_alloc_len_per_decode() -> int:
    """KV length one request may allocate in a single decode step.

    Reads the bags: adaptive speculative decoding moves the step count and the
    draft-token bound after publish, and this runs per decode batch.
    """
    spec = get_spec()
    if spec.speculative_algorithm is None:
        return 1

    # Spec decoding allocates max(topk * num_steps, num_draft_tokens) per decode step.
    spec_steps = spec.speculative_num_steps or 1
    spec_topk = spec.speculative_eagle_topk or 1
    spec_tokens = max_speculative_num_draft_tokens()
    page_size = get_alloc_page_size()

    from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

    spec_algo = SpeculativeAlgorithm.from_string(spec.speculative_algorithm)
    if page_size == 1 or spec_topk == 1 or not spec_algo.has_draft_kv():
        return max(spec_steps * spec_topk, spec_tokens)
    else:
        # spec v2 tree (page>1, topk>1): worst-case page-aligned footprint per
        # topk branch is ceil((page_size-1 + num_steps) / page) pages, each branch
        # duplicated -- reserve for all topk branches.
        num_new_pages_per_topk = (
            (page_size - 1) + spec_steps + page_size - 1
        ) // page_size
        return max(num_new_pages_per_topk * page_size * spec_topk, spec_tokens)


def get_alloc_reserve_per_decode() -> int:
    """KV length reserved per request at each decode step.

    The 2x is a double-buffer that absorbs the kv_committed_len lag in overlap
    mode; see eagle_utils.eagle_prepare_for_decode.
    """
    return 2 * get_alloc_len_per_decode()


def get_step_alloc_per_request(
    page_size: int, *, disable_overlap_schedule: bool
) -> int:
    """KV tokens one request may hold for a single decode step.

    Without speculation, one page. Under spec-v1, the per-request decode alloc.
    Under spec-v2, the overlap allocator keeps a double buffer outstanding
    (kv_committed_len + 2 * alloc_len), so both steps are resident at once.
    Shared by every windowed sub-pool's cap, which bounds its out-of-window
    lag by exactly this much.
    """
    spec = get_spec()
    if spec.speculative_algorithm is None:
        return page_size
    if disable_overlap_schedule:
        from sglang.srt.utils.common import spec_decode_alloc_len_per_request

        return spec_decode_alloc_len_per_request(
            page_size=page_size,
            speculative_num_steps=spec.speculative_num_steps,
            speculative_eagle_topk=spec.speculative_eagle_topk,
            speculative_num_draft_tokens=spec.speculative_num_draft_tokens,
        )
    return get_alloc_reserve_per_decode()


def windowed_token_cap(
    *,
    sliding_window_size: int,
    page_size: int,
    eviction_interval: int,
    draft_tokens: int,
    decode_alloc: int,
    num_reqs: int,
    prefill_headroom: int = 0,
) -> int:
    """Tokens a bounded windowed KV sub-pool must hold.

    ``__________[padding][eviction_interval][window]`` plus the decode step in
    flight: one windowed tail per resident request, plus whatever the sub-pool
    cannot have evicted yet (in-flight chunked prefill, or the slots a
    disaggregation decode role pre-allocates). The eviction interval is charged
    in draft tokens because that is the granularity at which the window
    frontier moves.

    Hoisted out of ``SWAChunkCapPoolConfigurator`` so every windowed sub-pool
    (a hybrid-SWA target's SWA pool, a windowed DFLASH draft pool) is sized by
    one formula instead of a per-architecture copy.
    """
    trailing_tokens = sliding_window_size + eviction_interval * draft_tokens + page_size
    return (trailing_tokens + decode_alloc) * num_reqs + prefill_headroom


def page_aligned_decode_alloc_lens(
    reqs,
    *,
    reserve: int,
    page_size: int,
):
    """Whole-page decode alloc lens: nxt rounds committed up to page so allocated
    == recorded (unaligned tails leak at ps>1)."""
    cur_kv_lens = [0] * len(reqs)
    nxt_kv_lens = [0] * len(reqs)
    num_needed_tokens = 0
    for i, r in enumerate(reqs):
        cur = r.kv.kv_allocated_len
        nxt = max(
            cur,
            (r.kv_committed_len + reserve + page_size - 1) // page_size * page_size,
        )
        cur_kv_lens[i] = cur
        nxt_kv_lens[i] = nxt
        num_needed_tokens += nxt - cur
    return cur_kv_lens, nxt_kv_lens, num_needed_tokens


def get_req_to_token_extra_context_len() -> int:
    """req_to_token row headroom beyond the model context length.

    Sized to hold the decode over-allocation; the spec v2 page>1 topk>1 holey
    draft footprint can outgrow the default num_draft_tokens headroom. The row
    headroom and the pools it sits next to derive from the same bag leaves, so
    they cannot disagree after a post-publish override.
    """
    # FIXME(lsyin): temporary fix for the context length issue under spec decoding
    extra = 4 + (max_speculative_num_draft_tokens() or 0)
    page_size = get_alloc_page_size()
    if get_spec().speculative_algorithm is not None and page_size > 1:
        # kv_allocated_len is page-aligned (eagle_prepare_for_decode), so near
        # the context limit the aligned reserve can overshoot by page_size - 1;
        # without the headroom the row write silently lands in the neighbor row.
        extra = max(extra, get_alloc_reserve_per_decode() + page_size - 1)
    return extra
