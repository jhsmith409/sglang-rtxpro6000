# Penny Royal 3.1

Penny Royal 3.1 makes DFlash2 with FR-Spec the default for Qwen3.8 Flash-Next
and replaces the preset KV-token cap with automatic pool sizing. Qwen3.8-27B FP8 with DFlash2 remains the second
supported profile. The target is RTX PRO 6000 Blackwell with one card (TP1) or
two (TP2).

## Flash-Next gets its own DFlash2 draft

The new default profile (`PENNYROYAL_PROFILE=next-dflash2`) runs the accepted
OrcaRouter ModelOpt NVFP4 Flash-Next target — same weights, same precision,
same cache stack — against the separate
[`incoai/Qwen3.8-Flash-Next-DFlash2`](https://huggingface.co/incoai/Qwen3.8-Flash-Next-DFlash2)
draft at full 524,288-token context. That draft is the one new download this
release. The profile ships with the 65,536-ID FR-Spec map, the resident
rowwise-FP8 output head shared with the target (PR90), candidate projection
through the FR-Spec map (PR93), a `.95` static memory fraction for
image-workload headroom, 6 requests / 36 Mamba slots, a 2,048-token draft
window, and draft GPU KV bounded to the sliding window without taking host
HiCache capacity (PR91, PR92).

TP2 is expected to work; it has not been tested on two-GPU hardware.

**Verification widths are fixed by active concurrency:** block 6 for a single
running request, block 4 for every batch of two or more, read from the bundled
policy `configs/pennyroyal/adaptive-dflash2.json`.

**Native NEXTN MTP is deprecated.** The `next` and `next-plain` profiles
remain available, with and without FR-Spec respectively. They receive no
further updates or testing. New setups use the DFlash2 profile.

## KV pool sizing is automatic

The launch recipes no longer request a preset KV-token cap (PR86). SGLang
sizes the shared pool from the GPU memory it profiles at boot, inside the
recipe's static memory fraction, and reports the resolved number in the startup
output; the 3.1 Flash-Next candidate boot resolved a 1,210,944-token pool this
way. An explicit `MAX_TOTAL_TOKENS` remains available as a deliberate,
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
intervals, reasoning included — on one RTX PRO 6000 at TP1, normal non-greedy
sampling. Full conditions and source labels are in
[RESULTS.md](RESULTS.md#penny-royal-31).

| Profile / workload | Mean tok/s | Sampled peak tok/s |
|---|---:|---:|
| Flash-Next DFlash2 — presentation workflow (agentic), C1 W6 (final candidate source `a60f442`) | **194.56** | **421.35** |
| Flash-Next — coding flow, C1 W6 (`5ca3f39`) | 235.60 | 417.50 |
| Flash-Next — single request, C1 W4 (`5ca3f39`) | 214.18 | 306.44 |
| Flash-Next — full six-request batch, aggregate, W4 (`5ca3f39`) | 566.14 | 657.52 |
| 27B DFlash2 — single request, fixed W8 (`27f9eaa`) | 133.06 | 232.07 |
| 27B DFlash2 — full four-request batch, aggregate (`27f9eaa`) | 459.74 | 570.73 |

| Cold prompt | Flash-Next prefill tok/s | 27B prefill tok/s |
|---|---:|---:|
| 63,906 tokens | 15,365.38 | 6,421.59 |
| 489,921 tokens | 8,715.05 | 1,677.10 |

Both profiles completed the recorded reasoning, tool, vision, long-context and
RAM/NIXL restore checks on the qualified sources. The presentation workflow
(agentic) ran end to end at block 6: 121 model turns, all matched one-to-one
against the retained timestamps, 192,349 output tokens including reasoning,
prompt growth 42,637 to 253,267 tokens. Reasoning-suite scores: 93.46/100
(Flash-Next) and 96.94/100 (27B) on actual answer grading; recorded deviations are listed with
the measurements.

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
  decision and the memory-fraction measurements.

The machine-readable record, including source identities and validation scope,
is [release-3.1.json](release-3.1.json).
