# Releasing

How to cut and publish a version of `@paulrobello/webllm`. Audience:
maintainers with publish rights.

## Table of Contents

- [Release flow](#release-flow)
- [Publishing surface](#publishing-surface)
- [Compatibility surfaces](#compatibility-surfaces)
- [Related Documentation](#related-documentation)

## Release flow

Releases go through CI/CD only. **Never run `npm publish` (or
`make publish`) locally** — per repo policy, registry publishes go through
CI.

1. **Decide the semver bump.** This project follows [Semantic
   Versioning](https://semver.org/). Note that the root barrel
   (`src/index.ts`) is the consumer API; deep internals live under the
   `./internal` subpath with no semver guarantees, and `./persistence` is
   versioned separately from the root.
2. **Update `CHANGELOG.md`.** Move entries from `## [Unreleased]` into a
   new `## [<new-version>] - <YYYY-MM-DD>` section. Any change to the
   conversation-persistence `schemaVersion` is a breaking change and must
   be recorded here (see [Compatibility surfaces](#compatibility-surfaces)).
3. **Bump `version` in `package.json`** to the new semver value.
4. **Tag the release** with an annotated tag in the form `v<semver>`
   (e.g. `v0.2.0`). Tags must be semver-shaped — the historical
   `p1-baseline-D` tag is not semver and is not a release.
5. **Push the tag.** The CI publish job (when wired — see ARC-005) builds
   the package and publishes to the registry. Until that job exists,
   publishing is blocked; do not work around it with a local publish.

> **Note:** The package ships zero runtime dependencies and includes only
> `dist/` (per `"files": ["dist"]` in `package.json`). The WASM artifacts
> (`webllm-wasm*.{js,wasm}`) are produced by `make wasm-build` and must be
> present in `dist/` before the publish job runs; `scripts/build-package.ts`
> fails cleanly if they are missing.

## Publishing surface

The npm package exports three subpaths (see `package.json` `exports`):

- `.` — the consumer API (`WebLLM`, types, errors, sampling profiles,
  persistence re-exports, evaluation helpers). This is the semver-stable
  surface.
- `./persistence` — `IndexedDBConversationStore` and persistence helpers.
- `./internal` — unstable internals for the smoke harness and power users.
  **No semver guarantees.** Consumers who import from `./internal` opt out
  of stability.

## Compatibility surfaces

The following are tracked compatibility surfaces — changes to them require
a CHANGELOG entry and, for breaking changes, a semver major bump:

- **Root barrel exports** (`src/index.ts`) — removing or renaming a
  value/type export from the `.` subpath is a breaking change.
- **`WebLLMConfig` / `ModelLoadOptions` / public method signatures** on
  `WebLLM` — additive optional fields are minor; removing or re-typing a
  field is breaking.
- **Conversation persistence `schemaVersion`** — a bump is breaking for
  previously-saved blobs. `importConversation` refuses mismatched schemas.
- **`ScoringMethod` / `EvalDimension` unions** (`src/evaluation/types.ts`)
  — additive; removing an arm is breaking for eval consumers.
- **Wire-format magic bytes (`WLKV`)** and the `ModelFingerprint` shape —
  changes invalidate saved blobs.

## Related Documentation

- [`CHANGELOG.md`](../CHANGELOG.md) — versioned change history
- [`CLAUDE.md`](../CLAUDE.md) — publishing policy (CI/CD only)
- [`package.json`](../package.json) — `exports` map and `files` include
