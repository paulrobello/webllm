#!/usr/bin/env python3
"""Patch the Emscripten-generated wasm64 bind-group shim.

Emscripten 5.0.6 (commit 6ea9c28c) emits a `_wgpuDeviceCreateBindGroup` shim
whose `makeEntry` reads the `WGPUBindGroupEntry::buffer` / `::sampler` /
`::textureView` pointer fields with `HEAPU32[(entryPtr+OFF)/4]` — i.e. only
the LOW 4 bytes of each 8-byte pointer. Under MEMORY64, when one of those
opaque handles is allocated above the 2^32 mark, the high 32 bits are lost
and `WebGPU.Internals.getJsObject()` returns undefined, which surfaces as:

    TypeError: Failed to read the 'buffer' property from 'GPUBufferBinding':
    Required member is undefined.

Concretely: a registered handle key like 0x1_a96cd6c8 (= 7,137,449,672) is
queried as 0xa96cd6c8 (= 2,842,482,376) and misses the table.

This script rewrites the three reads to use `HEAPU64[...]/8` followed by
`Number(...)` so the full 64-bit handle reaches `getJsObject`. Idempotent;
exits non-zero only if the expected pre-patch text isn't found exactly once
(meaning Emscripten's codegen layout drifted and the patch needs review).

Run after every `make wasm-build-mem64` — wired into the Makefile's
wasm-build-mem64 target so a fresh build always has the fix.

Tracking: PHASE-7-VALIDATION + task #543 in TODO.md.
"""

from __future__ import annotations

import sys
from pathlib import Path

PATCHES = [
    (
        "var bufferPtr=HEAPU32[(entryPtr+16)/4]",
        "var bufferPtr=Number(HEAPU64[(entryPtr+16)/8])",
    ),
    (
        "var samplerPtr=HEAPU32[(entryPtr+40)/4]",
        "var samplerPtr=Number(HEAPU64[(entryPtr+40)/8])",
    ),
    (
        "var textureViewPtr=HEAPU32[(entryPtr+48)/4]",
        "var textureViewPtr=Number(HEAPU64[(entryPtr+48)/8])",
    ),
]


def patch(path: Path) -> int:
    src = path.read_text()
    if all(new in src for _, new in PATCHES):
        # Already patched — idempotent no-op (e.g. running the script twice
        # in a row, or in CI where the build cache survived).
        print(f"[fix-mem64-bindgroup-shim] {path} already patched; no-op.")
        return 0

    for old, new in PATCHES:
        if src.count(old) != 1:
            sys.stderr.write(
                f"[fix-mem64-bindgroup-shim] FATAL: expected exactly one occurrence "
                f"of {old!r} in {path}, found {src.count(old)}.\n"
                "Emscripten codegen layout has drifted; review _wgpuDeviceCreateBindGroup\n"
                "in the generated file before updating this patch list.\n"
            )
            return 2
        src = src.replace(old, new)

    path.write_text(src)
    print(
        f"[fix-mem64-bindgroup-shim] {path} patched ("
        f"{len(PATCHES)} bind-group entry pointer reads "
        "switched HEAPU32 → HEAPU64)."
    )
    return 0


def main() -> int:
    if len(sys.argv) != 2:
        sys.stderr.write("usage: fix-mem64-bindgroup-shim.py <path-to-webllm-wasm-mem64.js>\n")
        return 1
    return patch(Path(sys.argv[1]))


if __name__ == "__main__":
    raise SystemExit(main())
