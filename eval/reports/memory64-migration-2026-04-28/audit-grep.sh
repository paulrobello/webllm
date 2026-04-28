#!/usr/bin/env bash
# MEMORY64 migration call-site audit. Run from repo root. Output is the
# raw evidence behind PUNCH-LIST.md — re-run after each phase commit to
# confirm no callers regressed back to stdlib _malloc/_free.
set -euo pipefail

REPORT_DIR="$(dirname "$0")"
cd "$(dirname "$0")/../../.."

echo "=== JS-side _malloc / _free / wasm.malloc / wasm.free ===" >"$REPORT_DIR/audit-js.txt"
grep -rn '_malloc\b\|_free\b\|wasm\.malloc\|wasm\.free\|m\._malloc\|m\._free' \
  --include='*.ts' --include='*.js' --include='*.html' \
  src/ smoke-test/ eval/ tests/ 2>/dev/null \
  | grep -v 'build/\|build-mem64/\|node_modules\|webllm-bundle\|webllm-wasm' \
  >>"$REPORT_DIR/audit-js.txt" || true

echo "=== JS-side stack allocator (stackAlloc/Save/Restore) ===" >"$REPORT_DIR/audit-stack.txt"
grep -rn 'stackAlloc\|stackSave\|stackRestore' \
  --include='*.ts' --include='*.js' --include='*.html' \
  src/ smoke-test/ tests/ 2>/dev/null \
  | grep -v 'build/\|build-mem64/\|node_modules\|webllm-bundle\|webllm-wasm' \
  >>"$REPORT_DIR/audit-stack.txt" || true

echo "=== Bridge int32_t size/offset params ===" >"$REPORT_DIR/audit-bridge.txt"
grep -n 'int32_t.*\(size\|offset\)\|int32_t mem_size' \
  src/wasm/webgpu-bridge.cpp \
  >>"$REPORT_DIR/audit-bridge.txt" || true

echo "=== GGUF loader heap boundaries ===" >"$REPORT_DIR/audit-gguf.txt"
grep -rn 'modelPtr\|wasm\.malloc\|HEAPU8\|heapU8\.buffer' \
  --include='*.js' --include='*.ts' \
  src/models/ smoke-test/ 2>/dev/null \
  | grep -v 'webllm-bundle\|webllm-wasm' \
  >>"$REPORT_DIR/audit-gguf.txt" || true

echo "Audit complete. See:"
ls -1 "$REPORT_DIR"/audit-*.txt
