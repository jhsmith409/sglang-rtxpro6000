"""CPU contract checks for Flash-Next (qwen4_exp) DFlash2 target-feature capture.

DFlash2 drafts from *completed* target-layer outputs, averaged over the target's
hyper-connection streams and concatenated into one feature row for the draft's
``fc``.  The inherited Qwen3-VL hook instead re-expresses the requested ids as
before-layer indices (+1) for a plain residual stream, and the hyper-connection
backbone does not implement that capture hook at all: its ``forward`` returns
``(hidden, hc_hidden)``, which the VL adapter reads as HC state, so the draft
would receive the packed 4-stream tensor instead of the trained features.

These checks drive the real adapter/backbone ``forward``, the real aux packer and
the real draft ``fc`` projection.  Only the CUDA-bound parts (the final HC mixer
and the decoder layers) are fixtures, and they carry exactly known values.
"""

import sys
from types import SimpleNamespace

import pytest
import torch

from sglang.srt.layers.aux_hidden_states import pack_aux_hidden_states
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.models.dflash import DFlash2DraftModel
from sglang.srt.models.qwen3_5 import Qwen3_5ForCausalLM
from sglang.srt.models.qwen4_exp import (
    Qwen4ExpForConditionalGeneration,
    Qwen4ExpVLModel,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

QWEN4_EXP = "sglang.srt.models.qwen4_exp"
HC_COUNT = 4
HIDDEN = 8
FEATURES = HC_COUNT * HIDDEN
LAYER_IDS = [5, 15, 25, 35, 45]
ROWS = 3
# One exact, distinct value per (layer, stream, row, column) of the fixture, so a
# wrong layer, a wrong stream axis, or a shuffled token axis all move the numbers.
STREAM_STEP, ROW_STEP, COLUMN_STEP = 100.0, 10.0, 0.5


def _fixture_packed_state(layer_id: int) -> torch.Tensor:
    streams = torch.arange(HC_COUNT).view(HC_COUNT, 1, 1) * STREAM_STEP
    rows = torch.arange(ROWS).view(1, ROWS, 1) * ROW_STEP
    columns = torch.arange(HIDDEN).view(1, 1, HIDDEN) * COLUMN_STEP
    value = layer_id * 1000.0 + streams + rows + columns
    # Packed the way the backbone carries it: one hidden_size block per stream.
    return value.permute(1, 0, 2).reshape(ROWS, FEATURES)


def _expected_feature(layer_id: int) -> torch.Tensor:
    stream_mean = STREAM_STEP * sum(range(HC_COUNT)) / HC_COUNT
    rows = torch.arange(ROWS).view(ROWS, 1) * ROW_STEP
    columns = torch.arange(HIDDEN).view(1, HIDDEN) * COLUMN_STEP
    return layer_id * 1000.0 + stream_mean + rows + columns


class _FixtureLayer:
    """Stand-in decoder layer: emits the packed post-layer HC state whose streams
    are individually known, and records whether the backbone offered it the
    inherited before-layer capture accumulator."""

    def __init__(self, layer_id: int):
        self.layer_id = layer_id
        self.offered_capture = False

    def __call__(self, **kwargs):
        self.offered_capture = kwargs["captured_last_layer_outputs"] is not None
        return _fixture_packed_state(self.layer_id), None


class _FixtureMixer:
    """The final HC mix needs CUDA; captured features never pass through it."""

    def mix(self, hidden_states):
        return hidden_states[..., :HIDDEN], None


def _backbone(*, pp_rank=0, pp_size=1, **overrides):
    model = object.__new__(Qwen4ExpVLModel)
    torch.nn.Module.__init__(model)
    model.pp_group = SimpleNamespace(
        rank_in_group=pp_rank,
        world_size=pp_size,
        is_last_rank=pp_rank == pp_size - 1,
        is_first_rank=pp_rank == 0,
    )
    model.hc_count = HC_COUNT
    model.hidden_size = HIDDEN
    model.has_ple = False
    model.ple_ngram_size = None
    model.ple_ngram_eos_token_id = None
    model._start_layer = 0
    model._end_layer = 48
    model.layers = [_FixtureLayer(i) for i in range(48)]
    model.layers_to_capture = []
    model._dflash_completed_layer_ids = ()
    model.embed_tokens = lambda input_ids: torch.zeros(ROWS, HIDDEN)
    model.hyper_connection_mixer = _FixtureMixer()
    model.last_hc_hidden_states = None
    for name, value in overrides.items():
        setattr(model, name, value)
    return model


def _top_level(backbone):
    return SimpleNamespace(
        model=backbone,
        pp_group=backbone.pp_group,
        capture_aux_hidden_states=False,
    )


def _forward(backbone, mode=ForwardMode.EXTEND):
    return Qwen4ExpVLModel.forward(
        backbone,
        input_ids=torch.arange(ROWS),
        positions=torch.arange(ROWS),
        forward_batch=SimpleNamespace(forward_mode=mode),
    )


def _project(aux, wrong_width_state):
    """Pack the captured list the way the logits path does, then run the real
    DFlash2 projection that enforces the trained ``fc`` width."""
    draft = SimpleNamespace(
        fc=torch.nn.Linear(HIDDEN * len(LAYER_IDS), HIDDEN, bias=False),
        is_nemotron_35_draft=False,
        num_context_features=len(LAYER_IDS),
        hidden_norm=RMSNorm(HIDDEN, eps=1e-6),
        config=SimpleNamespace(hidden_size=HIDDEN),
    )
    projected = DFlash2DraftModel.project_target_hidden(
        draft, pack_aux_hidden_states(aux)
    )
    assert projected.shape == (ROWS, HIDDEN)
    # A capture that emitted the packed HC state instead of the trained features
    # is the wrong width, and the draft must say so rather than misproject it.
    with pytest.raises(ValueError, match="feature dim mismatch"):
        DFlash2DraftModel.project_target_hidden(
            draft, pack_aux_hidden_states([wrong_width_state])
        )
    return projected


@pytest.mark.parametrize(
    "mode", [ForwardMode.EXTEND, ForwardMode.DECODE, ForwardMode.IDLE]
)
def test_capture_emits_completed_layer_hyper_connection_means(monkeypatch, mode):
    """One ``hidden_size`` feature per requested layer: the mean of that layer's
    completed hyper-connection streams, in requested-layer order, at every forward
    mode a captured graph may replay."""
    monkeypatch.setattr(f"{QWEN4_EXP}.get_parallel", lambda: SimpleNamespace(tp_size=1))
    top = _top_level(_backbone())
    Qwen4ExpForConditionalGeneration.set_dflash_layers_to_capture(top, LAYER_IDS)
    assert top.capture_aux_hidden_states and top.model.last_hc_hidden_states is None
    packed_state = _fixture_packed_state(LAYER_IDS[-1])

    _, aux = _forward(top.model, mode)

    # The adapter must leave the HC slot empty: the conditional-generation forward
    # copies it over ``output.hidden_states`` when it is set, which would erase the
    # very features the draft consumes.
    assert top.model.last_hc_hidden_states is None
    assert len(aux) == len(LAYER_IDS)
    for feature, layer_id in zip(aux, LAYER_IDS):
        assert feature.shape == (ROWS, HIDDEN)
        torch.testing.assert_close(feature, _expected_feature(layer_id))
    _project(aux, packed_state)


def test_capture_does_not_ask_the_layers_for_before_layer_state(monkeypatch):
    """The backbone captures at the completed-layer boundary itself, so the
    inherited per-layer hook stays unused on this backbone."""
    monkeypatch.setattr(f"{QWEN4_EXP}.get_parallel", lambda: SimpleNamespace(tp_size=1))
    top = _top_level(_backbone())
    Qwen4ExpForConditionalGeneration.set_dflash_layers_to_capture(top, LAYER_IDS)
    _forward(top.model)
    assert not any(layer.offered_capture for layer in top.model.layers)


def test_capture_keeps_the_shared_ple_batch_commit(monkeypatch):
    """The capture branch must not skip the n-gram commit every other Flash-Next
    forward performs: PLE is a sibling of the captured layers, so the commit has to
    stay on the path ahead of the DFlash return."""
    monkeypatch.setattr(f"{QWEN4_EXP}.get_parallel", lambda: SimpleNamespace(tp_size=1))
    commits = []
    monkeypatch.setattr(
        f"{QWEN4_EXP}._commit_ple_batch",
        lambda batch, forward_batch: commits.append(batch),
    )
    top = _top_level(_backbone())
    Qwen4ExpForConditionalGeneration.set_dflash_layers_to_capture(top, LAYER_IDS)
    output = _forward(top.model)
    assert isinstance(output, tuple) and len(output[1]) == len(LAYER_IDS)
    assert commits == [None]


def test_capture_rejects_a_topology_it_was_not_validated_on(monkeypatch):
    """The stream mean is validated at TP1 only: sharded or replicated HC streams
    would average different values per rank, so boot must stop, not drift."""
    monkeypatch.setattr(f"{QWEN4_EXP}.get_parallel", lambda: SimpleNamespace(tp_size=2))
    top = _top_level(_backbone())
    with pytest.raises(ValueError, match="tp_size=2"):
        Qwen4ExpForConditionalGeneration.set_dflash_layers_to_capture(top, LAYER_IDS)


@pytest.mark.parametrize(
    "layer_ids,pp_rank,pp_size,message",
    [
        # Pipeline split: layer 5 lives on the first stage, while only the last
        # stage's hidden states can reach the draft.
        (LAYER_IDS, 1, 2, "outside this rank"),
        ([5, 15, 25, 35, 35], 0, 1, "duplicates"),
        ([], 0, 1, "carried none"),
    ],
)
def test_capture_fails_loud_on_unusable_layer_ids(
    monkeypatch, layer_ids, pp_rank, pp_size, message
):
    monkeypatch.setattr(f"{QWEN4_EXP}.get_parallel", lambda: SimpleNamespace(tp_size=1))
    backbone = _backbone(pp_rank=pp_rank, pp_size=pp_size)
    if pp_size > 1:
        backbone._start_layer, backbone._end_layer = 24, 48
    with pytest.raises(ValueError, match=message):
        Qwen4ExpForConditionalGeneration.set_dflash_layers_to_capture(
            _top_level(backbone), layer_ids
        )


def test_non_dflash_forward_still_returns_hyper_connection_state(monkeypatch):
    """NEXTN/MTP preservation: with no DFlash capture configured the backbone
    still hands the packed pre-mix HC state to the adapter, which still keeps it on
    ``last_hc_hidden_states`` for the target's own draft instead of reading it as
    auxiliary features."""
    monkeypatch.setattr(f"{QWEN4_EXP}.get_parallel", lambda: SimpleNamespace(tp_size=1))
    backbone = _backbone()
    hidden = _forward(backbone)
    assert torch.is_tensor(hidden) and hidden.shape == (ROWS, HIDDEN)
    assert torch.equal(backbone.last_hc_hidden_states, _fixture_packed_state(47))
    with pytest.raises(ValueError, match="feature dim mismatch"):
        _project([backbone.last_hc_hidden_states], _fixture_packed_state(5))


def test_27b_capture_keeps_its_before_layer_indexing():
    """27B preservation: the plain-residual backbone keeps translating DFlash
    layer ids into before-layer capture indices, one flag per requested id."""
    backbone = object.__new__(Qwen3_5ForCausalLM)
    torch.nn.Module.__init__(backbone)
    backbone.layers = [_FixtureLayer(i) for i in range(64)]
    backbone.layers_to_capture = []
    Qwen3_5ForCausalLM.set_dflash_layers_to_capture(
        backbone, [val + 1 for val in LAYER_IDS]
    )
    assert backbone.layers_to_capture == [6, 16, 26, 36, 46]
    assert [
        layer.layer_id
        for layer in backbone.layers
        if getattr(layer, "_is_layer_to_capture", False)
    ] == [6, 16, 26, 36, 46]


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
