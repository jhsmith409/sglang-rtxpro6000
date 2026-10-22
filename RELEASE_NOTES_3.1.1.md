# Penny Royal 3.1.1

This patch release fixes one startup problem reported after 3.1: explicit
NVMe PLE selection was rejected at launch.

## Fix

- **NVMe PLE startup.** The bundled NVMe reader checked whole-module hashes
  of the runtime source it registers against, and those hashes were still
  the ones taken on the v3.0.1 tree. Any later source edit — including the
  unrelated changes that shipped in 3.1.0 — made the comparison fail, so the
  reader refused to register and NVMe PLE startup was rejected before the
  model could load. The whole-module gate and its source manifest are removed
  ([#111](https://github.com/jpezzulli/sglang-rtxpro6000/issues/111)).

Everything else about the reader is unchanged: table and artifact validation,
cache identity, the requirement to select NVMe explicitly, and the failure
handling for missing required hooks all remain enforced. Model, profile and
performance defaults are unchanged from 3.1.

## Update

Native users can follow [BUILD.md](BUILD.md). Container users should pull
`ghcr.io/jpezzulli/sglang-rtxpro6000:v3.1.1` and restart with their existing
external configuration. Fresh launch files and the beta configurator select
the new image; an explicit older image pin in saved settings survives until
you change it. The [3.1 release notes](RELEASE_NOTES_3.1.md) contain the
feature list and performance results.

## Thanks

- [@desva](https://github.com/desva) — NVMe PLE startup failure report.
- [Penny Royal](mailto:Pennyroyal@agentmail.to) — fix implementation.

## Status

The fix is accepted after an independent regression check on the v3.1.0 base,
75 passing CPU tests and 9 subtests, lint, and a clean independent review. The
container build, image digest and CPU package/content verification are pending
and will be recorded in release-3.1.1.json. NVMe serving verification is
outside this release's scope.
