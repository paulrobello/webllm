# ENH-005 — Probe: Per-Conversation KV Multiplexing Cost for Multi-NPC Agents

> **Status**: proposed (probe phase only) · **Effort/risk**: Probe Low · Follow-on High (user-gated)
> **Policy note**: KV-per-conversation-on-shared-weights multiplexing is **deferred** in CLAUDE.md,
> not banned. Per the probe-first doctrine, this plan executes ONLY the measurement; the multi-slot
> implementation proceeds only on a failing-swap verdict AND explicit user opt-in.

## Goal

Produce the number that adjudicates the deferred lever: what does a conversation switch cost
(KV snapshot save + load) at realistic multi-NPC scales, on the canonical hardware, with an 8B
model? The verdict decides between "snapshot-swap is fine — document the batched-tick NPC
pattern" and "true multi-slot KV is justified — file the follow-on".

## Probe declaration (per probe-first doctrine — declared before running)

- **Measures**: per-switch latency (ms) = save(current) + load(next), for occupied-context sizes
  {512, 2048, 4096} tokens, across {2, 4, 8} round-robin conversations, on (a) the smallest
  registered chat model and (b) one canonical 8B model. Also: bytes moved per swap, and whether
  latency scales linearly with occupied tokens.
- **Thresholds** (from the NPC analysis's 2-4 s/decision budget, memory `npc_control_scenario_analysis`):
  - swap p50 < 150 ms @ 4K context → **PASS**: swap overhead is <10% of the decision budget;
    document the round-robin/batched-tick pattern; the deferred lever STAYS deferred (closes with
    data instead of speculation).
  - swap p50 > 500 ms @ 4K → **FAIL**: swapping burns >25% of budget; multi-slot KV is justified;
    file the follow-on for user decision.
  - between → **TIERED**: document a tiered-tick pattern (hot conversation keeps the cache;
    background NPCs tick at lower frequency); lever stays deferred.
- **Gates**: the follow-on implementation decision, and the NPC-pattern documentation.

## Current state

- `ConversationPool` (`src/core/conversation-pool.ts`) already serializes/restores per-conversation
  KV snapshots — `chatCompletionWithConversation` does snapshot load/save around each turn today,
  so the swap path EXISTS; what's unmeasured is its cost at scale.
- Load-bearing patterns the probe harness must respect (vault, `tensor-readback-async-memory-corruption-pitfall`):
  KV reads via `await downloadFromTensor()` (fresh copy), writes via sync `uploadToTensor()`; the
  existing pool code does this — the probe *drives* it, never reimplements it.
- Probe-file conventions: model on `eval/probes/probe-prefix-cache-at-scale-2026-05-01.ts` (same
  shape of concern: KV behavior at scale) — argument parsing, timing capture, and report emission
  patterns are all there to copy.

## Implementation steps

1. **Harness** `eval/probes/probe-kv-multiplex-2026-07-14.ts` (Bun-side driver following the
   prefix-cache probes' structure): it drives the browser smoke/probe page the way those probes do
   (read one of them end-to-end first and copy its transport — they encode how a probe reaches the
   GPU path). Scenario per configuration: create N conversations, prefill each to the target
   occupied size with a fixed seed prompt, then run R=24 round-robin single-turn completions with
   1-token generation (isolates swap cost from decode cost), recording per-switch save/load timing
   from marks around the snapshot calls.
2. **Timing hooks**: if the snapshot save/load spans aren't already timed, add two
   `performance.mark`/`measure` pairs around them (in the pool or the turn path) behind a
   `WEBLLM_PROBE_TIMING` env/query gate so production paths are untouched; the probe page reads
   the measures. Remove-or-keep decision goes in the report.
3. **Run matrix**: 2 models × 3 context sizes × 3 conversation counts, 3 repetitions each; dashboard
   off or `?ingest=off` (throwaway runs per CLAUDE.md).
4. **Report** `eval/reports/kv-multiplex-probe-2026-07-14/SUMMARY.md`: matrix table, p50/p95 per
   cell, bytes/swap, verdict against the declared thresholds, and the recommendation. Follow the
   existing SUMMARY.md conventions (status, headline metric, links).
5. **Close the loop**: add the verdict as a line to TODO.md under the appropriate watch-list stub;
   if PASS/TIERED, include the documented NPC pattern (which tick pattern, measured budget share).

## Files to touch

- `eval/probes/probe-kv-multiplex-2026-07-14.ts` (new); possibly a paired probe page or params on
  the existing smoke page (match whatever transport the prefix-cache probes used)
- `src/core/conversation-pool.ts` or the turn path (gated timing marks only, if absent)
- `eval/reports/kv-multiplex-probe-2026-07-14/SUMMARY.md` (new), `TODO.md` (verdict line)

## Verification

1. `make checkall` (timing gates must not affect default paths — assert the gate defaults off).
2. The probe's own sanity check: swap timings must be nonzero and scale with occupied context; a
   flat-zero column means the marks are misplaced — fix before trusting the matrix.
3. Report review: verdict follows mechanically from the declared thresholds.

## Rollback

The probe is additive tooling. If timing marks were added to production files, they are gated and
removable in one commit; the report and TODO line are documentation and stay (negative results are
results — §28 template).
