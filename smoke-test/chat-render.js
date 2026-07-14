// chat-render.js — transcript message rendering with markdown,
// syntax highlight, and Qwen3 <think>...</think> collapse.

// Vendored deps register UMD globals (`marked`, `hljs`) — chat.html
// loads them via classic `<script>` tags before this module runs.
// (ES-module `import()` of these files runs in module scope and never
// reaches `globalThis`.)
async function ensureLibs() {
	// No-op kept for callers that previously awaited it.
}

/**
 * Split assistant text into { thinking, answer } based on Qwen3's
 * `<think>...</think>` convention. Unclosed `<think>` means the model
 * is still thinking (visible answer is empty); closed block separates.
 */
export function splitThinking(raw) {
	const open = raw.indexOf("<think>");
	if (open === -1) return { thinking: "", answer: raw };
	const close = raw.indexOf("</think>", open);
	if (close === -1) {
		return { thinking: raw.slice(open + "<think>".length), answer: "" };
	}
	const thinking = raw.slice(open + "<think>".length, close);
	const answer = raw.slice(0, open) + raw.slice(close + "</think>".length);
	return { thinking, answer };
}

/**
 * Render markdown to HTML using vendored `marked`. Falls back to
 * escape-only rendering when `marked` hasn't loaded yet (test mode).
 */
export function renderMarkdown(text) {
	const marked = globalThis.marked;
	if (!marked) return escapeHtml(text);
	// marked v12 exposes a default `marked()` function and `marked.parse`.
	const fn = typeof marked === "function" ? marked : marked.parse;
	return fn(text, { gfm: true, breaks: true });
}

function escapeHtml(s) {
	return s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

/**
 * Render an assistant bubble's content into the supplied element.
 * Splits any thinking block into a collapsible `<details>` and renders
 * the visible answer via markdown. Idempotent — call on every chunk.
 */
export async function renderAssistantInto(el, raw) {
	await ensureLibs();
	const { thinking, answer } = splitThinking(raw);
	el.innerHTML = "";
	if (thinking) {
		const details = document.createElement("details");
		details.className = "chat-think";
		const summary = document.createElement("summary");
		summary.textContent = "thinking";
		details.appendChild(summary);
		const pre = document.createElement("pre");
		pre.textContent = thinking;
		details.appendChild(pre);
		el.appendChild(details);
	}
	if (answer) {
		const body = document.createElement("div");
		body.className = "chat-md";
		body.innerHTML = renderMarkdown(answer);
		if (globalThis.hljs) {
			for (const block of body.querySelectorAll("pre code")) {
				try {
					globalThis.hljs.highlightElement(block);
				} catch (_e) {
					/* tolerate */
				}
			}
		}
		el.appendChild(body);
	}
}
