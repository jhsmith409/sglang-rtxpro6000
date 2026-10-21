# Penny Royal 3.1

Penny Royal 3.1 keeps native NEXTN MTP with FR-Spec as the maintained default
for Qwen3.8 Flash-Next, with its verify width fixed by active concurrency
(block 8 for one running request, block 4 for two or more). Flash-Next DFlash2
with FR-Spec is fully available as an optional profile — same weights, same
cache stack, its own draft model — now verifying draft tokens with
FlashInfer's SM120 kernel. The release also replaces the preset KV-token cap
with automatic pool sizing. Qwen3.8-27B FP8 with DFlash2 remains the second
supported profile, unchanged. The target is RTX PRO 6000 Blackwell with one
card (TP1) or two (TP2).

## Flash-Next keeps native MTP as the maintained default

The `next` profile (FR-Spec) and `next-plain` (without FR-Spec) keep their
existing recipes — `.97` and `.981` static memory fraction, 4 requests /
24 Mamba slots unless you override them — and keep receiving updates and
testing. New setups start on `next`, both native profiles remain maintained,
and a saved explicit choice is always preserved.

**The native verify width is fixed by active concurrency:** block 8 for a
single running request, block 4 for every batch of two or more, read from the
bundled policy `configs/pennyroyal/adaptive-next.json`. `next-plain` uses the
same widths without the FR-Spec map. A single request stays at width 8
instead of switching widths with acceptance feedback.

## Flash-Next gets its own DFlash2 draft

The optional profile (`PENNYROYAL_PROFILE=next-dflash2`) runs the accepted
OrcaRouter ModelOpt NVFP4 Flash-Next target — same weights, same precision,
same cache stack — against the separate
[`incoai/Qwen3.8-Flash-Next-DFlash2`](https://huggingface.co/incoai/Qwen3.8-Flash-Next-DFlash2)
draft at full 524,288-token context. That draft is the one new download this
release, needed only for the optional profile. The profile ships with the 65,536-ID FR-Spec map, the resident
rowwise-FP8 output head shared with the target (PR90), candidate projection
through the FR-Spec map (PR93), a `.95` static memory fraction for
image-workload headroom, 6 requests / 36 Mamba slots, a 2,048-token draft
window, and draft GPU KV bounded to the sliding window without taking host
HiCache capacity (PR91, PR92).

**The profile now uses FlashInfer for target linear-attention verification**
(`--linear-attn-verify-backend flashinfer`, PR104). Both the native launch
files and the container startup files select it automatically. On the final
candidate, matched screens show single-request decode at block 6 rising from
227.37 to 241.49 tok/s mean (sampled peak 339.53) and the full six-request
batch at block 4 from 503.41 to 504.80 tok/s aggregate (peak 568.90); the
completed ten-slide presentation reached 217.19 tok/s mean (peak 448.70).

**Verification widths are fixed by active concurrency:** block 6 for a single
running request, block 4 for every batch of two or more, read from the bundled
policy `configs/pennyroyal/adaptive-dflash2.json`.

DFlash2 offers more token-pool capacity; we are still evaluating performance
across concurrency levels. TP2 is expected to work for this profile; it has
not been tested on two-GPU hardware.

## KV pool sizing is automatic

The launch recipes no longer request a preset KV-token cap (PR86). SGLang
sizes the shared pool from the GPU memory it profiles at boot, inside the
recipe's static memory fraction, and reports the resolved number in the startup
output; this system resolved a 1,210,944-token target KV pool for the optional
DFlash2 profile and a 1,081,792-token pool for the final native-MTP
measurements. The maintained native-MTP profiles
size their pool the same way and report their own resolved number at startup.
An explicit `MAX_TOTAL_TOKENS` remains available as a deliberate,
page-aligned positive cap on every profile, and startup summaries say which of
the two paths produced the number you are reading.

## Fixes

- Finished requests are dropped from the running batch before every admission,
  and a request's held `req_to_token` row is counted once (PR87, PR88), so
  admission decisions match the batch that is actually running.
- Stop-regex matches inside accepted speculative blocks are honoured on both
  DFlash2 block sizes (PR96).
- Mixing token-probability requests with ordinary generation no longer
  triggers the identified scheduler crash (PR97).
- The beta configurator learned the `next-dflash2` profile on both the native
  and container paths (PR94): it prompts for the draft checkpoint and carries
  the fixed-width policy and FlashInfer verification from the samples. The
  external container configuration files and the manual native recipes remain
  the primary setup paths; the configurator stays an optional beta.
- The agreed upstream-fix subset (PR79) lands QSA cache-restore ordering and
  packed-scratch initialization, the shared Chat/Responses `chat_template`
  trust gate, absent `tool_calls` fields, stable stream timestamps, 400
  responses for truncated or corrupt images, collection before KV sizing,
  `ignore_eos` token reservation, and freed-response-state handling.
- The optional Swift FP8 PLE transplant/verification recipe (PR76) validates
  the whole safetensors data section, table widths, scales and payload layout
  before staging anything; its serving qualification remains contributor-owned.

## Current results

Native generation rates — mean and sampled peak over complete active-decode
intervals, reasoning included — on one RTX PRO 6000 at TP1, normal
non-greedy sampling, recorded October 21, 2026 on the merged final source
(`fe5f02c`) with the maintained default routing (block 8 at one running
request, block 4 at two or more), `.97` static memory fraction at 6 requests /
36 Mamba slots, diagnostics off, and an A4000 preprocessing sidecar. These
native conditions apply to the native rows only; every other row keeps its
recorded source and configuration label, and the full conditions for each are
in [RESULTS.md](RESULTS.md#penny-royal-31).

| Profile / workload | Mean tok/s | Sampled peak tok/s |
|---|---:|---:|
| Flash-Next native MTP — single request, C1 W8, `fe5f02c` (maintained default) | **197.87** | **345.30** |
| Flash-Next native MTP — full four-request batch, aggregate, W4, `fe5f02c` | **555.11** | **731.06** |
| Flash-Next native MTP — full six-request batch, aggregate, W4, `fe5f02c` | **472.79** | **528.49** |
| Flash-Next native MTP — coding workflow (agentic), C1 W8, medium, `fe5f02c` | **252.83** | **501.63** |
| Flash-Next native MTP — presentation workflow (agentic), C1 W8, medium, `fe5f02c` | **216.23** | **525.80** |
| Flash-Next DFlash2 + FlashInfer — single request, C1 W6, `a60f442` (optional profile) | **241.49** | **339.53** |
| Flash-Next DFlash2 + FlashInfer — full six-request batch, aggregate, W4, `a60f442` | **504.80** | **568.90** |
| Flash-Next DFlash2 + FlashInfer — presentation workflow (agentic), C1 W6, explicit medium, `a60f442` | **217.19** | **448.70** |
| 27B DFlash2 — single request, fixed W8, `27f9eaa` | 133.06 | 232.07 |
| 27B DFlash2 — full four-request batch, aggregate, `27f9eaa` | 459.74 | 570.73 |

| Cold prompt | Flash-Next DFlash2 prefill tok/s | 27B prefill tok/s |
|---|---:|---:|
| 63,906 tokens | 15,365.38 | 6,421.59 |
| 489,921 tokens | 8,715.05 | 1,677.10 |

The native coding workflow completed naturally at 27 model turns and the
agentic ten-slide presentation at 80 model turns, both at medium reasoning on
block 8. The DFlash2 reasoning qualification scored 93.46/100 for Flash-Next
and 96.94/100 for 27B, recorded on the DFlash2 profile.
[RESULTS.md](RESULTS.md#penny-royal-31) records the qualification and all
measurements.

## Updating

Native: follow [BUILD.md](BUILD.md#update-an-existing-install) with
`RELEASE_REF='pennyroyal-v3.1.0'` once the tag is published. Container:
download the release's launch-file set and run
`ghcr.io/jpezzulli/sglang-rtxpro6000:v3.1.0` with the matching startup files,
per the [container guide](docker/pennyroyal/README.md). Update any explicit
image pin — `PENNYROYAL_IMAGE=` in saved setup settings or the Compose `.env`,
or the `IMAGE=` value in an edited `run.sh` — to
`ghcr.io/jpezzulli/sglang-rtxpro6000:v3.1.0`, or pass that image name with `--image`. Existing target weights
are unchanged; the Flash-Next DFlash2 profile needs the new
`incoai/Qwen3.8-Flash-Next-DFlash2` draft download. Your saved settings, model
folders and persistent caches stay as they are, and no blanket cache deletion
is required. Regenerate setup files with the 3.1 configurator or edit the 3.1
samples; every save writes new timestamped files and preserves existing ones.
The complete settings list, defaults, interactions and restart/cache effects
are in [CONFIGURATION.md](CONFIGURATION.md).

## Thanks

- **[@jhsmith409](https://github.com/jhsmith409)** and
  **[@LocalhostRonin](https://github.com/LocalhostRonin)** — direct Penny Royal
  contributors to this release.
- **[Penny Royal](mailto:Pennyroyal@agentmail.to)** did almost all of the
  coding, including the ports and integration work.
- [John](https://github.com/jpezzulli), Penny and Codex worked through the
  architecture, tuning and debugging together — including the width-routing
  decision, the memory-fraction measurements and the verification-backend
  choice.

This release is in preparation with the final artifact pending: the tag
`pennyroyal-v3.1.0` and the `v3.1.0` image are planned, and the published
revision and image digest will be recorded when they exist (see
[release-3.1.json](release-3.1.json)).
