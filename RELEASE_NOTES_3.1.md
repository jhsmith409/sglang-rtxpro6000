# Penny Royal 3.1

Release preparation state: the source is integrated and documentation is
final; the public merge, immutable tag `pennyroyal-v3.1.0`, image build, and
release page are still pending and will be recorded on the GitHub release.

Penny Royal 3.1 moves Qwen3.8 Flash-Next onto DFlash2 speculation with FR-Spec
and makes that the default profile, replaces the preset KV-token cap with
automatic pool sizing, and deprecates the native NEXTN MTP Flash-Next profiles.
Qwen3.8-27B FP8 with DFlash2 remains the second supported profile. The target
is RTX PRO 6000 Blackwell with one card (TP1) or two (TP2).

## Flash-Next gets its own DFlash2 draft

- **A separate next-draft model.** The new default profile
  (`PENNYROYAL_PROFILE=next-dflash2`) runs the accepted OrcaRouter ModelOpt
  NVFP4 Flash-Next target against the separate
  [`incoai/Qwen3.8-Flash-Next-DFlash2`](https://huggingface.co/incoai/Qwen3.8-Flash-Next-DFlash2)
  draft at full 524,288-token context. The draft is a distinct download; the
  target, precision, and cache stack are unchanged. If you are coming from
  3.0.x, this is the one new model to fetch.
- **Fixed concurrency width routing, not acceptance tuning.** The draft block
  is a lookup on the active batch size from the bundled
  `configs/pennyroyal/adaptive-dflash2.json`: block 6 for a single running
  request, block 4 for every batch of two or more. One candidate per tier
  means acceptance feedback can never move the width, so behaviour is
  predictable and the old `SPECULATIVE_NUM_DRAFT_TOKENS` knob is retired on
  this profile rather than silently ignored. FR-Spec rides along on the same
  bundled 65,536-ID token map; the target keeps its full vocabulary.
- **Resident rowwise-FP8 output head.** The draft reuses the target's own
  online-FP8 rowwise output head instead of carrying a second copy (PR90), and
  the candidate projection with the FR-Spec map reduces the draft's search work
  (PR93).

## KV pool sizing is automatic

The launch recipes no longer request a preset KV-token cap (PR86). SGLang
sizes the shared pool from the GPU memory it profiles at boot, inside the
recipe's static memory fraction, and reports the resolved number in the
startup output. An explicit `MAX_TOTAL_TOKENS` remains available as a
deliberate positive, page-64-aligned cap on every profile. Startup summaries
say which of the two paths produced the number you are reading.

## Bounded draft cache, ordinary host cache

Draft GPU KV allocation is now bounded to what the draft's sliding window
actually requires (PR91) and no longer takes host HiCache capacity with it
(PR92), so the qualified 32 GB (Flash-Next) and 96 GB (27B) host-cache
behaviour is preserved while the draft keeps its own small pool. Scheduler
admission accounting was corrected in two places: finished requests are dropped
from the running batch before every admission (PR87) and a request's held
`req_to_token` row is counted once (PR88). Stop-regex matches inside accepted
speculative blocks are honoured again (PR96), and mixing token-probability
requests with ordinary generation no longer triggers the identified scheduler
crash (PR97).

## Native MTP is deprecated, not removed

The two Flash-Next native-MTP profiles (`next`, `next-plain`) still launch
with their existing recipes, including FR-Spec and the C1-only adaptive 4/8
selector, and the FR-Spec recipe keeps its separately measured `.97` memory
fraction (PR95). They are now deprecated: they receive no further updates or
testing. New setups should start on the DFlash2 profile.

## TP2

The DFlash2 capture path previously rejected `tp_size > 1` outright, which was
wrong: ordinary TP2 already produces complete replicated hyper-connection
state at the capture point (PR89). TP2 is expected to work; it has not been
tested on two-GPU hardware. The launcher still refuses a TP size the visible
GPUs cannot serve, and a sidecar preprocessing GPU stays outside the TP pair.

## Also included

The agreed upstream-fix subset (PR79) lands QSA cache-restore ordering and
packed-scratch initialization, the shared Chat/Responses `chat_template` trust
gate, absent `tool_calls` fields, stable stream timestamps, 400 responses for
truncated or corrupt images, collection before KV sizing, `ignore_eos` token
reservation, and freed-response-state handling. The optional Swift FP8 PLE
transplant and verification recipe (PR76) validates the whole safetensors data
section, table widths, scales, and payload layout before staging anything; its
serving qualification remains contributor-owned.

## Measured on one RTX PRO 6000

Native generation rates, mean and sampled peak, reasoning included, on a
single-card TP1 boot; the 3.0-era W4 screens keep their W4 labels and the
completed workflows below ran the routed block-6 width at C1. Full conditions
are in [RESULTS.md](RESULTS.md#penny-royal-31).

| Profile / workload | Mean tok/s | Sampled peak tok/s |
|---|---:|---:|
| Flash-Next DFlash2 — presentation workflow, C1 W6 (final source) | **194.56** | **421.35** |
| Flash-Next — coding workflow, C1 W6 | 235.60 | 417.50 |
| Flash-Next — single request, W4 | 214.18 | 306.44 |
| Flash-Next — full six-request batch, aggregate, W4 | 566.14 | 657.52 |
| Flash-Next — reasoning request, W4 | 210.24 | 300.79 |
| 27B DFlash2 — single request, fixed W8 | 133.06 | 232.07 |
| 27B DFlash2 — full four-request batch, aggregate | 459.74 | 570.73 |
| 27B DFlash2 — reasoning request, fixed W8 | 125.88 | 215.93 |

| Cold prompt | Flash-Next prefill tok/s | 27B prefill tok/s |
|---|---:|---:|
| 63,906 tokens | 15,365.38 | 6,421.59 |
| 489,921 tokens | 8,715.05 | 1,677.10 |

Both profiles completed the recorded reasoning, tool, vision, long-context,
and RAM/NIXL restore checks. Current reasoning-suite scores are 93.46
(Flash-Next) and 96.94 (27B) on actual answer grading; the suites are not
presented as perfect, and recorded model and tool deviations are listed in
[RESULTS.md](RESULTS.md#penny-royal-31).

## Updating

Native: follow [BUILD.md](BUILD.md#update-an-existing-install) with
`RELEASE_REF='pennyroyal-v3.1.0'` once the tag is published. Container:
download the release's launch-file set and run
`ghcr.io/jpezzulli/sglang-rtxpro6000:v3.1.0` with the matching startup files,
per the [container guide](docker/pennyroyal/README.md). If an explicit
`PENNYROYAL_IMAGE=` or `IMAGE=` pin names an older image, update or delete it
to move to the new default; your saved settings, model folders, and persistent
caches stay as they are, and no blanket cache deletion or weight re-download
is required — except that the Flash-Next DFlash2 profile needs the new
`incoai/Qwen3.8-Flash-Next-DFlash2` draft download. Regenerate setup files
with the 3.1 configurator, or edit the 3.1 samples; every save writes new
timestamped files and preserves existing ones. The complete settings list,
defaults, interactions, and restart/cache effects are in
[CONFIGURATION.md](CONFIGURATION.md).

## Thanks

- **[@jhsmith409](https://github.com/jhsmith409)** and
  **[@LocalhostRonin](https://github.com/LocalhostRonin)** — direct Penny Royal
  contributors to this release.
- **[Penny Royal](mailto:Pennyroyal@agentmail.to)** did almost all of the
  coding, including the ports and integration work.
- **[John](https://github.com/jpezzulli)** worked through the architecture,
  source investigation, tuning, and debugging together with Penny — including
  the width-routing decision and the memory-fraction measurements — not just
  the direction.

The machine-readable record, including source identities and validation scope,
is [release-3.1.json](release-3.1.json).
