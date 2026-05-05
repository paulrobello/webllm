// P1 tokenizer parity fixture generator. For each canonical vocab,
// reads the GGUF, builds a legacy Tokenizer, runs the 200-prompt
// corpus through Tokenizer.encode(), writes parity-fixture.json.
// See PROMPT-FIXTURE.md for corpus structure rationale.
//
// Run from repo root:
//   bun run eval/reports/p1-tokenizer-2026-05-05/generate-fixture.ts
//
// Idempotent — overwrites parity-fixture.json each run.

import { readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { Tokenizer } from "../../../src/inference/tokenizer.js";
import { ModelLoader } from "../../../src/models/model-loader.js";

interface VocabSpec {
	name: string;
	ggufPath: string;
	ggufUrl: string;
}

const VOCABS: VocabSpec[] = [
	{
		name: "spm-llama",
		ggufPath: "smoke-test/models/tinyllama-1.1b-chat-q4_0.gguf",
		ggufUrl: "/models/tinyllama-1.1b-chat-q4_0.gguf",
	},
	{
		name: "llama-bpe",
		ggufPath: "smoke-test/models/llama-3.2-1b-q4f16.gguf",
		ggufUrl: "/models/llama-3.2-1b-q4f16.gguf",
	},
	{
		name: "qwen2",
		ggufPath: "smoke-test/models/qwen2.5-1.5b-q4f16.gguf",
		ggufUrl: "/models/qwen2.5-1.5b-q4f16.gguf",
	},
	{
		name: "qwen3",
		ggufPath: "smoke-test/models/qwen3-0.6b-q4f16.gguf",
		ggufUrl: "/models/qwen3-0.6b-q4f16.gguf",
	},
	{
		name: "wordpiece-bert",
		ggufPath: "smoke-test/models/bge-small-en-v1.5-q0f16.gguf",
		ggufUrl: "/models/bge-small-en-v1.5-q0f16.gguf",
	},
];

// ---------------------------------------------------------------------------
// 200-prompt corpus. 5 categories × 40 prompts each.
// See PROMPT-FIXTURE.md for category rationale.
// ---------------------------------------------------------------------------

const CAT1_ASCII: string[] = [
	// 15 single-sentence prose snippets
	"The capital of France is Paris.",
	"Hello, world!",
	"She said, \"It's a long way to Tipperary.\"",
	"What time is the meeting tomorrow?",
	"I can't believe it's not butter.",
	"The quick brown fox jumps over the lazy dog.",
	"To be, or not to be, that is the question.",
	"All work and no play makes Jack a dull boy.",
	"A journey of a thousand miles begins with a single step.",
	"Knowledge is power, but enthusiasm pulls the switch.",
	"He didn't want to go, but he went anyway.",
	"The price is $19.99 -- a real bargain.",
	"Don't count your chickens before they hatch.",
	"Roses are red; violets are blue.",
	"Time flies like an arrow; fruit flies like a banana.",
	// 10 multi-sentence paragraphs
	"The cat sat on the mat. It was a sunny afternoon. Birds chirped outside.",
	"I went to the store. I bought milk and eggs. The total was $7.42.",
	"Climbing was hard. The rocks were slippery. We persevered. We reached the top.",
	"Server logs show three errors. Each error references a missing file. Restart the daemon.",
	"The recipe calls for two cups of flour. Sift it carefully. Add the eggs slowly.",
	"He paused. Then he spoke. His voice was calm and measured.",
	"Open the door. Close the window. Turn off the light before you leave.",
	"Tests are passing now. Coverage is at 94%. Linter reports no warnings.",
	"Boot time is critical. We measured 1.8 seconds. The target is under 2.",
	"Read the docs. Run the tests. Commit your changes. Push to origin.",
	// 5 code snippets
	"def foo(x): return x + 1",
	"const arr = [1, 2, 3].map(x => x * 2);",
	"SELECT id, name FROM users WHERE active = 1;",
	"#!/bin/bash\nfor f in *.txt; do echo \"$f\"; done",
	"{\"name\": \"alice\", \"age\": 30, \"items\": [1,2,3]}",
	// 5 punctuation-heavy
	"https://example.com/path/to/page",
	"192.168.1.100",
	"alice.bob+tag@example.co.uk",
	"/usr/local/bin/python3.11",
	"C:\\Users\\Admin\\Documents\\file.txt",
	// 5 numeric / arithmetic
	"1+1=2",
	"3.14159265358979",
	"1,000,000",
	"$100.00",
	"0xDEADBEEF",
];

const CAT2_UNICODE: string[] = [
	// 10 emoji-bearing
	"Good morning 🌞",
	"Math: 𝕏 ∈ ℝ",
	"Family: 👨‍👩‍👧",
	"Pizza 🍕 and beer 🍺 tonight",
	"🎉🎊🎈 happy birthday 🎂🎁",
	"Weather: ☀️ → 🌧️ → ⛈️",
	"❤️🧡💛💚💙💜",
	"Smiling face: 😀, winking: 😉, crying: 😢",
	"Skin tones: 👋🏻 👋🏼 👋🏽 👋🏾 👋🏿",
	"Pirate flag: 🏴‍☠️ adventure ⚓",
	// 10 CJK
	"你好世界",
	"中文测试,带标点。",
	"今天天气很好。",
	"我爱编程。",
	"机器学习是未来。",
	"こんにちは、世界。",
	"日本語のテスト",
	"漢字とひらがなとカタカナ",
	"안녕하세요 세계",
	"한국어를 사랑해요",
	// 5 RTL
	"مرحبا بالعالم",
	"اللغة العربية جميلة",
	"שלום עולם",
	"עברית מודרנית",
	"السلام عليكم ورحمة الله",
	// 5 combining-diacritics
	"café", // precomposed é
	"café", // decomposed e + combining acute
	"Nguyễn", // Vietnamese tone mark
	"phở bò",
	"/kæt/", // IPA
	// 5 ligatures and special spacing
	"em space",
	"em—dash",
	"ellipsis…",
	"oﬁce", // fi ligature
	"ﬂower", // fl ligature
	// 5 mixed-script
	"Hello 世界",
	"Project 中文-名 v1.0",
	"Café in 北京 with 🌞",
	"مرحبا hello שלום",
	"Status: 完了 ✅",
];

const CAT3_SPECIAL: string[] = [
	// 8 ChatML
	"<|im_start|>system\nYou are helpful.<|im_end|>",
	"<|im_start|>user\nWhat is 2+2?<|im_end|>",
	"<|im_start|>assistant\n4<|im_end|>",
	"<|im_start|>system\nYou are concise.<|im_end|>\n<|im_start|>user\nHi.<|im_end|>",
	"<|im_start|>user\nTell a joke.<|im_end|>\n<|im_start|>assistant\nWhy did the chicken cross the road?<|im_end|>",
	"<|im_start|>system\nReply only in JSON.<|im_end|>\n<|im_start|>user\nWhat is 1+1?<|im_end|>\n<|im_start|>assistant\n{\"answer\": 2}<|im_end|>",
	"<|im_start|>tool\n{\"result\": 42}<|im_end|>",
	"<|im_start|>user\nHello.<|im_end|>\n<|im_start|>assistant\nHi! How can I help?<|im_end|>",
	// 8 Llama-2 INST
	"<s>[INST] What is 2+2? [/INST]",
	"<s>[INST] <<SYS>>\nYou are helpful.\n<</SYS>>\n\nHello! [/INST]",
	"<s>[INST] Tell me about Paris. [/INST] Paris is the capital of France.</s>",
	"[INST] Continue this story. [/INST]",
	"<s>[INST] <<SYS>>\nBe brief.\n<</SYS>>\n\nWhat is AI? [/INST] Artificial intelligence.</s>",
	"<s>[INST] Translate to French: hello [/INST] bonjour</s><s>[INST] thank you [/INST]",
	"<s>[INST] Code a hello world in Python. [/INST]",
	"<s>[INST] What's 5 * 7? [/INST] 35</s>",
	// 6 Phi-3
	"<|user|>question<|end|><|assistant|>",
	"<|system|>You are concise.<|end|><|user|>Hi.<|end|><|assistant|>",
	"<|user|>What is 2+2?<|end|><|assistant|>4<|end|>",
	"<|system|>Be brief.<|end|><|user|>Tell a joke<|end|><|assistant|>Why did the chicken cross the road?<|end|>",
	"<|user|>Translate \"hello\" to Spanish.<|end|><|assistant|>Hola.<|end|>",
	"<|user|>Continue: Once upon a time<|end|><|assistant|>",
	// 6 Qwen3 thinking-tag
	"<think>reasoning</think>",
	"<think>Let me think step by step.</think>The answer is 4.",
	"<|im_start|>assistant\n<think>I should be careful.</think>\nHi!<|im_end|>",
	"<think>\nMulti-line\nreasoning\nblock\n</think>\nFinal.",
	"<think></think>empty think",
	"answer<think>after</think>more",
	// 6 BERT special-token
	"[CLS] question [SEP] answer [SEP]",
	"[CLS] hello world [SEP]",
	"[CLS] the capital of [MASK] is paris [SEP]",
	"[CLS] short [SEP]",
	"[CLS] who wrote hamlet [SEP] william shakespeare [SEP]",
	"[CLS] [MASK] [MASK] [MASK] [SEP]",
	// 6 mixed
	"<tool_call>{\"name\": \"search\", \"args\": {\"q\": \"hi\"}}</tool_call>",
	"<tool_response>{\"result\": \"ok\"}</tool_response>",
	"text<|endoftext|>",
	"<|im_start|>assistant\n<tool_call>{\"name\": \"x\"}</tool_call><|im_end|>",
	"<|im_end|><|endoftext|>",
	"prefix<|im_start|>system\nbe nice<|im_end|>suffix",
];

const CAT4_EDGE: string[] = [
	// 5 empty-or-whitespace
	"",
	" ",
	"\n",
	"\t",
	"                                                  ", // 50 spaces
	// 5 whitespace-run
	"\n\n\n\n\n\n\n\n",
	"\t\t\t\t",
	"   ",
	"\n \t\n ",
	"     leading whitespace",
	// 5 trailing-whitespace
	"text\n",
	"text\n\n",
	"text\n\n\n\n\n",
	"hello world   ",
	"line\t",
	// 5 single-char
	"a",
	"7",
	"!",
	"😀",
	"中",
	// 5 very-long
	"abcdefghij".repeat(26).slice(0, 256), // 256 chars no spaces
	"ab".repeat(500), // 1000 chars repetitive
	Array.from({ length: 100 }, (_, i) => `line ${i}`).join("\n"),
	"x".repeat(512),
	"the ".repeat(200).trim(),
	// 5 control-char
	" ",
	"",
	"",
	"﻿", // BOM
	"‍", // ZWJ
	// 5 byte-fallback edge
	"hello 𝕏 world",
	"a\u{1F600}b", // 4-byte codepoint between ASCII
	"\u{1D54F}\u{1D54F}\u{1D54F}",
	"text 𝓐𝓑𝓒 more",
	"\u{2070E}", // CJK Ext B
	// 5 newline-only-with-context
	"a\nb",
	"a\n\nb",
	"\na",
	"a\n",
	"\n\n",
];

const CAT5_MIXED: string[] = [
	// 8 URL strings
	"https://example.com/path?query=1&foo=bar",
	"https://example.com/page#section-2",
	"https://en.wikipedia.org/wiki/Caf%C3%A9",
	"https://api.example.com/v1/users/42?expand=profile,settings",
	"http://localhost:8080/health",
	"https://github.com/anthropics/claude-code/issues/1234",
	"https://docs.example.com/path/to/article?utm_source=twitter&utm_campaign=spring",
	"https://example.com/search?q=hello%20world&lang=en",
	// 8 JSON snippets
	"{\"name\":\"alice\",\"age\":30}",
	"{\"items\":[{\"id\":1},{\"id\":2}]}",
	"{\"text\":\"line1\\nline2\\ttab\"}",
	"{\"unicode\":\"caf\\u00e9\"}",
	"[{\"a\":1},{\"a\":2},{\"a\":3}]",
	"{\"nested\":{\"deep\":{\"deeper\":{\"deepest\":true}}}}",
	"{\"empty\":{},\"null\":null,\"bool\":false}",
	"{\"escaped\":\"a \\\"quote\\\" and \\\\backslash\"}",
	// 8 HTML / Markdown
	"<a href=\"https://example.com\">link</a>",
	"# Heading 1\n## Heading 2\n\nParagraph text.",
	"```python\nprint('hello')\n```",
	"[Café](https://example.com/café)",
	"<p>Hello <strong>world</strong>!</p>",
	"- item one\n- item two\n  - nested\n- item three",
	"`inline code` in a sentence",
	"| col1 | col2 |\n|------|------|\n| a    | b    |",
	// 8 multi-paragraph mixed prose-and-code
	"Here is a function:\n\n```js\nfunction add(a, b) { return a + b; }\n```\n\nUse it like `add(1, 2)`.",
	"To install run:\n\n```bash\nnpm install foo\n```\n\nThen import it.",
	"The error trace was:\n\n```\nTypeError: undefined is not a function\n  at line 42\n```\n\nWe fixed it.",
	"Config:\n```yaml\nport: 8080\nhost: localhost\n```\nThat's all.",
	"Step 1: read the file. Step 2:\n```\ncat file.txt | grep foo\n```\nStep 3: parse output.",
	"Use SQL: ```SELECT * FROM users WHERE id = 1```. Then in Python: ```rows = cursor.fetchall()```.",
	"Diff:\n```diff\n- old line\n+ new line\n```\nThat's the change.",
	"Header.\n\n    indented code block\n    second line\n\nFooter.",
	// 8 social-media style
	"@alice check out #typescript -- it's great! 🎉",
	"RT @bob: just shipped v2.0! 🚀🚀 link in bio",
	"GM ☀️ what's everyone working on today?",
	"lol that meme was 🔥🔥🔥 fr no cap",
	"Just released the new feature! Thanks @team for the hard work 💪 #buildinpublic",
	"hot take: tabs > spaces. fight me. 🥊",
	"trying new café in 巴黎 🥐 -- 10/10 vibes",
	"Update: I'll be at #ConfXYZ next week. DM if you want to grab coffee ☕",
];

// Sanity-check category sizes at load time so a typo can't ship silently.
function assertCount(name: string, arr: string[], expected: number): void {
	if (arr.length !== expected) {
		throw new Error(
			`${name}: expected ${expected} prompts, got ${arr.length}`,
		);
	}
}
assertCount("CAT1_ASCII", CAT1_ASCII, 40);
assertCount("CAT2_UNICODE", CAT2_UNICODE, 40);
assertCount("CAT3_SPECIAL", CAT3_SPECIAL, 40);
assertCount("CAT4_EDGE", CAT4_EDGE, 40);
assertCount("CAT5_MIXED", CAT5_MIXED, 40);

const PROMPTS: string[] = [
	...CAT1_ASCII,
	...CAT2_UNICODE,
	...CAT3_SPECIAL,
	...CAT4_EDGE,
	...CAT5_MIXED,
];
if (PROMPTS.length !== 200) {
	throw new Error(`Expected 200 prompts, got ${PROMPTS.length}`);
}

// ---------------------------------------------------------------------------
// Loader + encoder loop
// ---------------------------------------------------------------------------

function loadVocab(spec: VocabSpec): Tokenizer {
	const buf = new Uint8Array(readFileSync(spec.ggufPath));
	const parsed = ModelLoader.parseModel(buf);
	return new Tokenizer(parsed.tokenizerConfig);
}

interface FixtureEntry {
	vocab: string;
	ggufUrl: string;
	expected: { prompt: string; ids: number[] }[];
}

const fixture: FixtureEntry[] = [];
for (const spec of VOCABS) {
	console.log(`[${spec.name}] loading ${spec.ggufPath}…`);
	const tk = loadVocab(spec);
	const expected = PROMPTS.map((prompt) => ({
		prompt,
		ids: tk.encode(prompt),
	}));
	fixture.push({ vocab: spec.name, ggufUrl: spec.ggufUrl, expected });
	console.log(`[${spec.name}] encoded ${PROMPTS.length} prompts.`);
}

const outPath = join(import.meta.dir, "parity-fixture.json");
writeFileSync(outPath, JSON.stringify({ prompts: PROMPTS, fixture }, null, 2));
console.log(`Wrote ${outPath}`);
