<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/brand/penny-royal/logos/logo-horizontal-dark.svg">
    <img src="assets/brand/penny-royal/logos/logo-horizontal-light.svg" alt="Penny Royal" width="480">
  </picture>
</p>

# Penny Royal — Qwen3.8 on RTX PRO 6000

Run long conversations, coding agents, and tool-heavy workflows on your own
hardware. Penny Royal is an SGLang-derived runtime tuned for **NVIDIA RTX PRO
6000 Blackwell (SM120), with one card (TP1) or two (TP2)**. TP2 is expected to
work; it has not been tested on two-GPU hardware.

**Persistent conversation caching is at the heart of Penny.** HiCache and NIXL
keep reusable state in system RAM and on NVMe, beyond what fits on the GPU,
so long-running sessions can pick up from cached history.

**524K context · HiCache + NIXL · DFlash2 + FR-Spec**

[Get started](#build-and-launch) · [Conversation cache](#hicache-and-nixl) ·
[Performance](#performance) ·
[3.1 release notes](RELEASE_NOTES_3.1.md) · [Community results](COMMUNITY-RESULTS.md)

**Penny Royal 3.1:** Flash-Next now runs DFlash2 with
FR-Spec and a separate draft model as the default profile, with fixed
concurrency width routing (block 6 for one request, block 4 for two or more),
automatic KV pool sizing replacing the preset cap, and the native NEXTN MTP
recipes deprecated but still launchable.

<a id="hicache-and-nixl"></a>

## Built for long conversations and agents

Every tool call adds another turn. A coding session can build up hundreds of
thousands of tokens of history, and processing that history again costs time.

Penny keeps reusable conversation state across three layers:

| Where | What it does |
|---|---|
| **GPU memory** | Keeps active conversations ready for the next turn. |
| **System RAM — HiCache** | Holds reusable state beyond what fits on the GPU and brings it back when needed. |
| **NVMe storage — NIXL** | Keeps cached state on disk so matching conversation history can be restored, including after a server restart with the same configuration. |

When your client sends the next turn, Penny reuses the matching cached history
and processes the new material.

Qwen's hybrid models need their recurrent state restored alongside the
attention cache. Penny handles that complete state, including Flash-Next's
additional PLE and QSA state, so restored conversations can continue correctly.

Penny restored a **490K-token conversation** from cache and processed just
**five new tokens**, skipping almost half a million tokens of repeated prefill.

Both model profiles use this hierarchy. **NIXL disk caching is on by default**;
you can disable it while keeping GPU and RAM caching. The
[run guide](RUN.md#choose-hicache-ram-size) covers sizing and setup, and
[memory and persistence](MEMORY-AND-PERSISTENCE.md#hicache-and-nixl) explains
how the state is stored and restored.

## What's new in 3.1

Flash-Next gets **DFlash2 with FR-Spec** and its own draft model,
`incoai/Qwen3.8-Flash-Next-DFlash2`, and that profile is the default for new
setups. The draft block is **fixed by active concurrency** — block 6 for a
single running request, block 4 for two or more. The **KV pool is sized
automatically** from profiled memory instead of a preset cap; an explicit
`MAX_TOTAL_TOKENS` remains available. The native NEXTN MTP recipes still
launch but are **deprecated**: no further updates or testing. Scheduler
admission, stop-regex inside speculative blocks and mixed-probability requests
are fixed. TP2 is expected to work; it has not been tested on two-GPU
hardware. See the
[3.1 release notes](RELEASE_NOTES_3.1.md) for the full change list and the
[configuration reference](CONFIGURATION.md) for every supported setting.

<a id="models-and-launch-recipes"></a>
<a id="qualified-model-profiles-and-launch-recipes"></a>

## Two model profiles

Both profiles support **524,288-token context**, tool calling, reasoning,
images, HiCache RAM prefix caching, and optional NIXL disk caching.

| Model | Weights | Speculative decoding | Checkpoints |
|---|---|---|---|
| **Qwen3.8 Flash-Next** | NVFP4 experts, automatic FP8 dense layers on SM120 | DFlash2 with a separate draft model and FR-Spec; width routed by batch size (block 6 at C1, block 4 at C2+); native NEXTN MTP deprecated | [Our OrcaRouter ModelOpt conversion](https://huggingface.co/jpezzulli/OrcaRouter-Qwen3.8-Flash-Next-Uncensored-ModelOpt-NVFP4) · [RadixArk](https://huggingface.co/RadixArk/Qwen3.8-Flash-Next-NVFP4) · [Flash-Next DFlash2 draft](https://huggingface.co/incoai/Qwen3.8-Flash-Next-DFlash2) |
| **Qwen3.8-27B** | FP8 | DFlash2, with a separate draft model at fixed block 8 | [Qwen FP8](https://huggingface.co/Qwen/Qwen3.8-27B-FP8) · [DFlash2 draft](https://huggingface.co/incoai/Qwen3.8-27B-DFlash2) |

I personally run Flash-Next on **one RTX PRO 6000 at C=6**, with **524,288-token
context** and the KV pool sized automatically from profiled memory; the final
3.1 candidate boot resolved a **1,210,944-token** shared GPU pool. That pool is
shared across active requests and cached prefixes.

### How draft width is routed

The Flash-Next DFlash2 profile pairs the NVFP4 target with the separate
`incoai/Qwen3.8-Flash-Next-DFlash2` draft and FR-Spec. The verification block
is fixed by active concurrency, read from the bundled policy file
`configs/pennyroyal/adaptive-dflash2.json`:

| Requests generating together | Draft behavior |
|---|---|
| **C=1** | Fixed six-token blocks. |
| **C=2 or more** | Fixed four-token blocks; speculative decoding stays on. |

One candidate width per tier means the width is a lookup, so it is identical
between boots, and a single request gets the wider draft without making every
request in a busy batch pay that extra draft work. **C is request concurrency;
TP is the number of inference GPUs.**

**FR-Spec rides along at every concurrency.** FR-Spec narrows the vocabulary
searched by the draft head; routing selects how many tokens it drafts. The
target model keeps its full vocabulary. The qualified launch is the
[DFlash2 recipe](RUN.md#launch-flash-next-with-dflash2); the native-MTP
[FR-Spec](RUN.md#launch-flash-next-with-fr-spec) and
[non-FR recipes](RUN.md#launch-flash-next-without-fr-spec-alternative) remain
available as deprecated options.

For 27B alternatives, see the [Swift 27B results](SWIFT-27B.md) and the
[OrcaRouter FP8 checkpoint](https://huggingface.co/orcarouter/Qwen3.8-27B-Uncensored-FP8).

<a id="performance-and-qualification"></a>
<a id="performance-and-testing"></a>
<a id="flash-next-v250-online-fp8-performance-at-a-glance"></a>

## Performance

### Flash-Next 3.1

**Presentation workflow (agentic): 194.56 tok/s mean, 421.35 peak (C1, block 6). Coding flow: 235.60 mean, 417.50 peak (C1, block 6).**

Measured on one RTX PRO 6000 Blackwell, using our OrcaRouter ModelOpt NVFP4
checkpoint, FR-Spec with DFlash2, a 524K context limit, and normal non-greedy
sampling. The completed presentation ran on the final 3.1 candidate source; the
other rows keep the source and width of the screen that recorded them.

| Workload | Mean tok/s | Sampled peak tok/s |
|---|---:|---:|
| Presentation workflow (agentic), C1 W6 | **194.56** | **421.35** |
| Coding flow, C1 W6 | **235.60** | **417.50** |
| Single request, C1 W6 (current default) | **228.44** | **336.20** |
| Single request, C1 fixed W4 | 214.18 | 306.44 |
| Six concurrent requests, aggregate, fixed W4 | 566.14 | 657.52 |

These are **SGLang's native generation rates**, including reasoning. Prefill,
tool execution, and idle time are separate. The presentation completed end to
end: 121 model turns and 192,349 output tokens at block 6 with one request
generating. [Current results](RESULTS.md#penny-royal-31) records the sources,
intervals and settings behind every number.

| Cold prompt | Mean prefill tok/s |
|---|---:|
| 63,906 tokens | **15,365.38** |
| 489,921 tokens | **8,715.05** |

The reasoning suite scored **93.46/100** on the qualified source, graded on
actual answers. See [results and measurement settings](RESULTS.md#penny-royal-31)
for the tool checks, cache restore, sampling details, and earlier results.

<a id="27b-fp8--dflash2-performance-at-a-glance"></a>

### Qwen3.8-27B FP8 with DFlash2

Current recorded performance on one RTX PRO 6000, fixed block 8:

| Workload | Mean tok/s | Sampled peak tok/s |
|---|---:|---:|
| Single request | **133.06** | **232.07** |
| Four concurrent requests, aggregate | **459.74** | **570.73** |

| Cold prompt | Mean prefill tok/s |
|---|---:|
| 63,906 tokens | 6,421.59 |
| 489,921 tokens | 1,677.10 |

The reasoning suite scored **96.94/100** on this profile.

[Benchmark history](RESULTS.md) keeps the earlier records — the October 10 3.0
screens, the 2.5.3 comparison, the public code-edit benchmark, and the dated
27B campaigns — with their original labels and timing methods.

<a id="independent-tp2-fp8-validation"></a>

Running two cards? [Community results](COMMUNITY-RESULTS.md) covers TP2 setups
and measurements from other users.

## Build and launch

Choose the installation path that suits your machine:

| Path | What you do |
|---|---|
| **[Prebuilt container](docker/pennyroyal/README.md)** | Download the image and launch files, edit your host configuration, and run it. Models and caches stay on the host. |
| **[Native installation](BUILD.md)** | Build the runtime, set your model and cache paths, and use the [native startup recipe](RUN.md). |

Both paths work with ordinary configuration files. The optional
**[beta configurator](CONFIGURE.md)** helps write those files and prints the
launch command. It is open for testing; please [report issues](https://github.com/jpezzulli/sglang-rtxpro6000/issues)
if you try it. The complete list of supported settings, defaults, interactions
and restart/cache effects is the [configuration reference](CONFIGURATION.md).

### Your container configuration stays outside the image

```text
your-host/
  run.sh                              image, GPU selection, ports and mounts
  config/start-flash-next-dflash2.sh  model and SGLang startup settings
  config/nixl-posix.toml              optional disk-cache settings
```

Prepare those files before starting. `run.sh` starts Docker and mounts your
configuration directory read-only at `/config`; the container runs the startup
file from that mount. Models and writable caches are mounted separately.
Edit the host files and restart to apply changes. You can download a newer
image while keeping your own configuration—no container rebuild required.

The [container guide](docker/pennyroyal/README.md) provides the files and
commands; the beta configurator can generate the same ordinary files for you.

### Hardware and memory

The inference target is **RTX PRO 6000 Blackwell, TP1 or TP2**. An Ampere or
Ada Lovelace GPU can be used separately for image preprocessing; it is not an
extra tensor-parallel inference card. CPU preprocessing is also available. The
build is narrowed to RTX Blackwell inference and Ampere/Ada sidecar image
processing — other GPU build targets are trimmed, which keeps the release
slimmer and easier to download.

| Profile | Default host cache | Additional model-related host memory |
|---|---|---|
| Flash-Next | 32 GB HiCache | About 48 GiB for RAM-backed PLE |
| 27B/DFlash2 | 96 GB HiCache | Loading and runtime allocations |

Leave room for the operating system and model loading. You can
[change the RAM cache size](RUN.md#choose-hicache-ram-size), or move Flash-Next's
PLE table to an SSD with [NVMe PLE](NVME-PLE.md).

NIXL disk caching is **on by default and optional**. Switching it off keeps GPU
and RAM prefix caching. NVMe PLE is a separate choice. The [launch guide](RUN.md)
explains cache sizing, GPU selection, concurrency, and startup.

The first start compiles kernels and captures CUDA graphs. Wait for the
server-ready message, then [check the API](RUN.md#smoke-through-the-normal-api).

## Built for the way we use it

Penny Royal started as my home runtime on thegrid. I wanted long conversations
and useful coding agents on the hardware I already owned. The tuning follows
that workload: keep several requests moving, make tool calls usable, and avoid
reprocessing a long conversation when its state can be restored.

The name comes from Penny Royal in Neal Asher's Polity books.

Thanks to **[aiueo52](https://github.com/aiueo52)** for the Flash-Next optimization
work and discussion, **[LandOfLemons](https://github.com/LandOfLemons)** for the
session-cache and WSL2 contributions, **[cube4elements](https://github.com/cube4elements)**
for the image-request crash fix, and **[untcoder2](https://github.com/untcoder2)**,
**[StockSpecialist1707](https://www.reddit.com/user/StockSpecialist1707/)**, and
**[palves](https://github.com/palves)** for field testing and setup feedback.
For 3.1, **[@jhsmith409](https://github.com/jhsmith409)** and
**[@LocalhostRonin](https://github.com/LocalhostRonin)** contributed directly.
[Penny Royal](mailto:Pennyroyal@agentmail.to) did almost all the coding across
these releases, including the ports and integration work. [John](https://github.com/jpezzulli),
Penny and Codex worked through the architecture, implementation, tuning and debugging
together, with plenty of back and forth over what was actually worth keeping.

## Documentation

| I want to… | Read |
|---|---|
| Install or upgrade | [Native build](BUILD.md) · [Container](docker/pennyroyal/README.md) |
| Choose launch, cache, GPU, and context settings | [Run guide](RUN.md) · [Configuration reference](CONFIGURATION.md) |
| Understand conversation caching | [HiCache and NIXL](#hicache-and-nixl) · [Memory and persistence](MEMORY-AND-PERSISTENCE.md#hicache-and-nixl) |
| Try guided setup | [Beta configurator](CONFIGURE.md) |
| Understand the FP8 and SSD options | [Online FP8](FP8.md) · [NVMe PLE](NVME-PLE.md) |
| See measurements | [Results and history](RESULTS.md) · [Community reports](COMMUNITY-RESULTS.md) |
| Read the implementation details | [Backends](BACKENDS.md) · [Memory and persistence](MEMORY-AND-PERSISTENCE.md) |
| Check changes and source lineage | [3.1 notes](RELEASE_NOTES_3.1.md) · [Machine-readable notes](release-3.1.json) · [History](CHANGES.md) · [Provenance](PROVENANCE.md) |
| Check known issues | [Limitations](LIMITATIONS.md) |
| Use an AI assistant for setup | [llms.txt](llms.txt) |

Our [validation suite](https://github.com/jpezzulli/pennyroyal-validation)
contains the reasoning, tool, image, context, and performance tests.

To hear about new releases, choose **Watch → Custom → Releases** on GitHub.
If you build on Penny Royal, please link this repository and the version you
started from. Feedback, contributions, and real workload reports are welcome.

<p align="center">
  <img src="assets/brand/penny-royal/artwork/penny-royal-crystal-transparent.png" alt="Penny Royal crystal artwork" width="240">
</p>

## License

The SGLang-derived source is Apache-2.0. Model checkpoints and third-party
components retain their own licenses.

<!-- Preserve existing README links while detailed material lives on its own page. -->
<a id="current-release--pennyroyal-v251"></a>
<a id="current-release--pennyroyal-v250"></a>
<a id="current-release--pennyroyal-v253"></a>
<a id="v250-options--online-fp8-and-nvme-ple"></a>
<a id="retained-v240-flash-next-prefill-measurements"></a>
<a id="qwen38-27b-fp8dflash2--august-24-2026-campaign"></a>
<a id="27b-real-agentic-context-behavior"></a>
<a id="earlier-releases"></a>
<a id="v23--faster-flash-next-decoding-with-fr-spec"></a>
<a id="previous-v212--maintenance-release"></a>
<a id="qualified-configuration-matrix"></a>
<a id="resolved-backends-qwen38-27bdflash2"></a>
<a id="resolved-backends-qwen38-flash-next"></a>
<a id="sm120-enablement-was-scoped-not-indiscriminate"></a>
<a id="how-pennyroyal-selects-sm120-paths"></a>
<a id="memory-recovery-and-automatic-kv-sizing"></a>
<a id="scope-and-limitations"></a>
<a id="documentation-map"></a>

[Current release notes](RELEASE_NOTES_3.1.md) · [Benchmark history](RESULTS.md) ·
[Configuration reference](CONFIGURATION.md) · [Known issues](LIMITATIONS.md)

<!-- Compatibility anchors for earlier versions of this guide. -->
<a id="broader-model-support-without-hicachenixl"></a>
<a id="building-on-pennyroyal"></a>
<a id="choosing-your-settings"></a>
<a id="flash-next-precision-and-placement"></a>
<a id="hardware-storage-and-first-start-expectations"></a>
<a id="hicachenixl-persistence"></a>
<a id="model-profiles-and-launch-recipes"></a>
<a id="prebuilt-docker-image-for-qwen38-on-rtx-pro-6000"></a>
<a id="qwen38-flash-next-nvfp4-with-online-fp8"></a>
<a id="sglang-for-qwen38-on-one-nvidia-rtx-pro-6000-blackwell-sm120"></a>
<a id="source-changes-and-upstream-work"></a>
<a id="technical-reference"></a>
<a id="validation-suite"></a>
<a id="why-this-runtime-exists"></a>
