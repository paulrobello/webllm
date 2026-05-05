# Tool-format investigation 2026-05-04 — Llama-3.x + Phi-3

**Trigger:** TODO.md watch-list ticket `ab44813` queued two
investigation-first probes following the Mistral V0.3
`[AVAILABLE_TOOLS]` fix:

1. **Llama-3.x tool format** — explain the Llama-3.1-8B (0.98) vs
   Llama-3.2-3B (0.17) gap on the same code path.
2. **Phi-3 tool format** — determine whether Phi-3.5-mini's 0.17
   floor has a usable upstream format.

**Outcome:**

- **Llama-3.x (RECLASSIFIED):** not a format-dispatch issue. Pure
  capability cliff at <8B. Sub-8B Llama variants emit
  *structurally-shaped* attempts that fail the strict JSON parser.
  → file follow-up: **lenient parser for malformed tool-call
  emissions** (parser-side fix, not a chat-template fix).
- **Phi-3.5-mini (CLOSED no-fix):** no tool-call attempt at all.
  Hallucinates conversational results. Phi-3-instruct base is not
  function-calling-fine-tuned and emits no usable structural
  signal regardless of instruction format. → demote tool-calling
  expectations; close as "no usable format upstream."

---

## Investigation method

Pulled per-task `tc-001` outputs from `eval/reports/smoke-runs.db`
for the canonical greedy-baseline-2026-05-04 run set:

| Model | eval_id | Tool-calling score |
|---|---|---:|
| llama-3.1-8b-instruct-iq3m | bench-1777932978119-yvcdn5 | 0.98 |
| llama-3.2-3b-q4f16 | bench-1777931257388-r820dd | 0.17 |
| llama-3.2-1b-q4f16 | bench-1777930881942-czdia1 | 0.17 |
| hermes-3-llama-3.2-3b-q4f16 | bench-1777931337587-kicv7t | 0.17 |
| tinyllama-1.1b-chat-q4_0 | bench-1777930929933-zuasxg | 0.17 |
| phi-3.5-mini-q4km | bench-1777931420699-yyidqx | 0.17 |

All ran on the same canonical instructions
(Qwen3/Hermes-style `<tool_call>{"name":"...","arguments":{...}}</tool_call>`).

---

## Per-model emission patterns (tc-001: get_weather, Tokyo)

### Llama-3.1-8B-Instruct (0.98) — reference behavior

```
<tool_call>
{"name": "get_weather", "arguments": {"city": "Tokyo"}}
</tool_call>
```

Faithfully follows the Qwen3-format we instruct. Format-compliance
is total. Score 0.98 across 12 tool-calling tasks. **Already optimal
on the current code path.**

### Llama-3.2-3B (0.17) — structural malformation

```
<tool_call>
    <name>get_weather</name>
    <arguments>
        {"city": "Tokyo"}
    </arguments>
</tool_call>
```

Pattern: model interprets the `<tool_call>` wrapper as XML and
generalizes the inner content into XML child elements. Args remain
JSON for easy tasks; on medium-difficulty tasks (`tc-005` `send_email`)
the args degenerate further:

```
<arguments>
    <to>John</to>
    <subject>Meeting Invitation</subject>
    <body>...</body>
</arguments>
```

JSON-inside-XML is **mechanically parseable** (extract `<name>` text,
parse `<arguments>` body as JSON if it starts with `{`). Pure-XML
arguments are not — would need an XML-to-object parser.

### Llama-3.2-1B (0.17) — system-prompt parroting

```
# Tools

<tools>
  {"type":"function","name":"get_weather","parameters":{"city":"Tokyo","units":"metric"}}
</tools>

# Tool Call
<tool_call>
  {"name":"get_weather","arguments":{"city":"Tokyo","units":"metric"}
</tool_call>

# Response
{"name":"get_weather","arguments":{"city":"Tokyo","units":"metric"}
```

1B model echoes section headers (`# Tools`, `# Tool Call`, `# Response`)
back as part of its output, then omits closing braces (`{...` rather
than `{...}`). Format-instruction overload — model can't separate
"this is the format you USE" from "this is content I REPRODUCE."
Not fixable by parser leniency; deeper capability gap.

### Hermes-3-Llama-3.2-3B (0.17) — fabricated JSON keys

```
{"name": "get_weather", "arguments:
 "args-json-object":
 {"city": "Tokyo"}}
```

Mangles the JSON structure: missing closing quote on `"arguments`,
fabricates an unrequested `"args-json-object":` key, multiline
malformation. Reasoning-fine-tuned variant (Hermes-3) actually
performs *worse* than base Llama-3.2-3B on tool-calling — the
reasoning post-training appears to displace strict-format adherence.

### TinyLlama (0.17) — ignores tool concept

```
Japanese: Sesshin Tōkyō
English: Tokyo
In this question, "Japanese" refers to the Japanese language...
```

No structural attempt. Treats the tool-laden prompt as a translation
question. 1.1B at Q4_0 lacks the capacity to even register the
tool-calling intent.

### Phi-3.5-mini (0.17) — conversational hallucination

```
To provide the current weather in Tokyo, I would typically use a
weather API or website. However, as an AI, I don't have real-time
capabilities. Here's how you can find the weather in Tokyo:

1. Visit a weather forecasting website like Weather.com...
```

Or (tc-002, search_restaurants):

```
After conducting a search using the specified criteria, I have
found several Italian restaurants in New York. Here are a few
options:
1. Carbone - Located at 10 W 53rd St...
2. L'Artusi - Situated at 120 W 34th St...
```

Phi-3 either declines (refuses) or **hallucinates the tool result
as conversational text**. Never emits a `<tool_call>` wrapper, never
emits JSON, never references the tool by name. Phi-3-instruct was
trained on instruction-following but **not on a function-calling
fine-tune**; the base model has no internal representation of the
tool-call protocol.

---

## Findings

### Finding 1 — Llama-3.x ticket reclassification

**Original framing:** "Llama-3.x tool format" — implying a
format-dispatch fix akin to Mistral V0.3's `[AVAILABLE_TOOLS]`.

**Actual finding:** **NO format-dispatch fix would help.** Llama-3.1-8B
already scores 0.98 on the current Qwen3 format; switching to
`<|python_tag|>` would at best preserve that and at worst regress it.
Sub-8B variants fail not because the format is wrong but because
they can't faithfully reproduce *any* multi-token structural format.

**Right fix:** parser-side leniency for the Llama-3.2-3B malformed
XML pattern.

```typescript
// Add to tool-system.ts: third XML variant
const XML_NESTED_RE =
  /<tool_call>\s*<name>([^<]+?)<\/name>\s*<arguments>\s*(\{[\s\S]*?\})\s*<\/arguments>\s*<\/tool_call>/;
```

This handles JSON-inside-XML. Pure-XML arguments are deferred (more
complex; lift from rare medium tasks isn't worth the complexity
right now).

**Predicted impact** (no validation run yet — that's the implementation
ticket's job):
- llama-3.2-3b-q4f16: 0.17 → ~0.50-0.70 (8 of 12 tasks emit
  parseable JSON-inside-XML; 2 are no-tool-call, 2 are medium with
  pure-XML args)
- llama-3.2-1b-q4f16: minimal lift — failures are upstream of
  parser (incomplete braces, parroted headers)
- hermes-3-llama-3.2-3b-q4f16: minimal lift — fabricated JSON keys
- tinyllama-1.1b-chat-q4_0: zero lift — no structural attempt

**Action:** file follow-up ticket as **parser-side fix**, not a
chat-template family fix. Different surface area entirely.

### Finding 2 — Phi-3.5-mini tool-calling: structural absence

**Phi-3 emits no tool-call signal of any kind.** Hallucinates results
conversationally (tc-001), pretends to have invoked the tool (tc-004),
or refuses citing AI limitations. Across 10 tool-calling tasks, the
model emitted exactly zero `<tool_call>` tags, zero JSON-shaped
objects, and zero structural attempts.

**Microsoft documentation check:** Phi-3.5-mini-instruct's HF model
card (`microsoft/Phi-3.5-mini-instruct`) and Microsoft's Cookbook
sample notebooks **do not document a function-calling format**. Phi-3
was trained on instruction-following + reasoning; tool-calling was
not in the post-training mix. Microsoft's Phi-3 + tool-calling demos
in vLLM/Ollama use generic prompt-engineering (no fine-tuned
emission convention).

**No format engineering will lift Phi-3.5 above the 0.17 floor**
because there's no learned emission to elicit. The model would need
either (a) a tool-calling-specific fine-tune that doesn't exist
upstream, or (b) a much heavier system-prompt-engineering approach
(few-shot examples, structured output coercion via response-format
constraints, or grammar-guided decoding) — all out of scope for a
chat-template-level fix.

**Action:** **close as "no usable format upstream"**. Demote
tool-calling expectations for Phi-3 in dashboard / per-model
documentation. The 1.00 reasoning + 0.76 instruction-following are
the model's load-bearing strengths; tool-calling is not.

---

## Closure recommendations

### Llama-3.x ticket (reclassified)

**Status:** **investigation closed; follow-up ticket filed.** The
original "Llama-3.x format" framing is wrong — close that. File a
new ticket: **"Tool-call parser leniency for sub-8B Llama-3 family
emissions"** with scope:

1. Add `XML_NESTED_RE` regex to `src/characters/tool-system.ts`.
2. Add tests to `tests/tool-system.test.ts` covering:
   - JSON-inside-XML pattern (Llama-3.2-3B easy tasks)
   - Negative case: `<name>` without `<arguments>` (don't false-match)
   - Negative case: pure-XML arguments (don't false-match;
     graceful no-parse rather than incorrect parse)
3. Re-bench llama-3.2-3b-q4f16 + llama-3.2-1b-q4f16 +
   hermes-3-llama-3.2-3b-q4f16 to validate predicted lift.
4. Update dashboard if 3B tier moves materially.

**Predicted scope:** ~1 hour wall (1 regex + 3 tests + 1 bench).

### Phi-3.5-mini ticket (closed no-fix)

**Status:** **closed; demote tool-calling expectations.**
- No follow-up implementation ticket.
- Update dashboard / per-model documentation to flag Phi-3.5-mini
  as **strong on reasoning, weak on tool-calling**.
- The 0.17 floor is the achievable ceiling for this model on the
  current bench surface.

---

## Process notes

The investigation surfaced an under-stated lesson from the
Mistral V0.3 fix earlier in the session: **format-dispatch fixes
have a model-capability prerequisite**. The Mistral V0.3 fix worked
because the model was trained on `[AVAILABLE_TOOLS]` and could
faithfully emit the matching `[TOOL_CALLS]` structure. For Phi-3
there is no learned format to elicit; for Llama-3.2-3B there is a
learned format but the model can't reproduce it faithfully at scale
< 8B.

Going forward, before reaching for a chat-template family-dispatch
addition, **first verify the model emits *some* structural signal
on the current path** — if it emits nothing (Phi-3 case), no format
will work. If it emits *something structural-but-wrong* (Llama-3.2
case), parser leniency may be the right surface, not template
reconfiguration.

---

## Artifacts

- Per-model task outputs extracted from
  `eval/reports/smoke-runs.db` (eval_ids listed in method table
  above).
- Original ticket queue: TODO.md commit `ab44813`.
- Greedy-baseline closure that triggered the investigation:
  [`eval/reports/greedy-baseline-2026-05-04/SUMMARY.md`](../greedy-baseline-2026-05-04/SUMMARY.md).
- Mistral V0.3 fix that established the original format-dispatch
  pattern: commit `0f590a4`.
