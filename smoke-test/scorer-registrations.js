/**
 * Browser-side registration of the 13 custom scorers used by the shipped
 * task packs. Imports `registerCustomScorer` from the compiled bundle, so
 * the registry instance here is the SAME one that `score()` queries — a
 * separate build step would create a second module copy and score() would
 * never see these registrations.
 *
 * Mirror of eval/tasks/scorer-registrations.ts — keep them in sync when
 * adding or changing scorers. They have to match by name.
 */

// Pick up the bundle URL from the page's query string so cache-busted
// reloads are honoured (matches the pattern real-model-page.js uses).
const q = typeof window !== "undefined" ? window.location.search || "" : "";
const { registerCustomScorer } = await import(`./webllm-bundle.js${q}`);

// ── reasoning/rs-012 — water jug problem ───────────────────────────────
registerCustomScorer("rs-012-water-jug", (output) => {
	const lower = output.toLowerCase();
	const mentionsFill5 = lower.includes("fill") && lower.includes("5");
	const mentionsPour = lower.includes("pour") || lower.includes("transfer");
	const mentionsEmpty =
		lower.includes("empty") ||
		lower.includes("dump") ||
		lower.includes("discard");
	const mentions4 = lower.includes("4");
	let score = 0;
	if (mentionsFill5) score += 0.25;
	if (mentionsPour) score += 0.25;
	if (mentionsEmpty) score += 0.25;
	if (mentions4) score += 0.25;
	return score;
});

// ── instruction/in-002 — one-sentence answer ───────────────────────────
registerCustomScorer("in-002-one-sentence", (output) => {
	const sentences = output.match(/[.!?]+/g);
	const count = sentences ? sentences.length : 0;
	if (count === 1) return 1;
	if (count === 2) return 0.5;
	return 0;
});

// ── instruction/in-006 — exactly 5 numbered items ──────────────────────
registerCustomScorer("in-006-numbered-5-items", (output) => {
	const lines = output.split("\n");
	let numberedCount = 0;
	for (const line of lines) {
		if (/^\s*\d+[.)]\s/.test(line)) numberedCount++;
	}
	if (numberedCount === 5) return 1;
	if (numberedCount >= 4) return 0.5;
	return 0;
});

// ── instruction/in-008 — avoid forbidden words ─────────────────────────
registerCustomScorer("in-008-avoid-forbidden-words", (output) => {
	const lower = output.toLowerCase();
	const forbidden = ["beautiful", "pretty", "nice"];
	const used = forbidden.filter((w) => lower.includes(w));
	if (used.length === 0) return 1;
	if (used.length === 1) return 0.5;
	return 0;
});

// ── instruction/in-009 — photosynthesis 3-bullet constraints ───────────
registerCustomScorer("in-009-photosynthesis-3-bullets", (output) => {
	let score = 0;
	const lines = output.split("\n").filter((l) => l.trim());
	const bullets = lines.filter((l) => /^\s*[-*•]\s/.test(l));
	if (bullets.length === 3) score += 0.34;
	const allCapitalized = bullets.every((b) => /^\s*[-*•]\s*[A-Z]/.test(b));
	if (allCapitalized) score += 0.33;
	if (output.toLowerCase().includes("chlorophyll")) score += 0.33;
	return Math.min(score, 1);
});

// ── instruction/in-011 — Alice JSON schema ─────────────────────────────
registerCustomScorer("in-011-alice-json", (output) => {
	try {
		const json = JSON.parse(output);
		let score = 0;
		if (typeof json.name === "string" && json.name.toLowerCase() === "alice") {
			score += 0.25;
		}
		if (typeof json.age === "number" && json.age === 30) score += 0.25;
		if (Array.isArray(json.hobbies)) score += 0.25;
		if (
			Array.isArray(json.hobbies) &&
			json.hobbies.some((h) => String(h).toLowerCase().includes("reading")) &&
			json.hobbies.some((h) => String(h).toLowerCase().includes("hiking"))
		) {
			score += 0.25;
		}
		return score;
	} catch {
		return 0;
	}
});

// ── instruction/in-012 — three questions in order ──────────────────────
registerCustomScorer("in-012-three-questions-order", (output) => {
	const idx7 = output.indexOf("7");
	const idxBlue = output.toLowerCase().indexOf("blue");
	const idxTokyo = output.toLowerCase().indexOf("tokyo");
	if (
		idx7 !== -1 &&
		idxBlue !== -1 &&
		idxTokyo !== -1 &&
		idx7 < idxBlue &&
		idxBlue < idxTokyo
	) {
		return 1;
	}
	const found = [idx7 !== -1, idxBlue !== -1, idxTokyo !== -1].filter(
		Boolean,
	).length;
	return found / 3;
});

// ── semantic-reasoning/emb-003 — "fast" synonyms ────────────────────────────────
registerCustomScorer("emb-003-fast-synonyms", (output) => {
	const synonyms = ["quick", "rapid", "swift", "speedy", "brisk", "hasty"];
	const lower = output.toLowerCase().trim();
	return synonyms.some((s) => lower.includes(s)) ? 1 : 0;
});

// ── semantic-reasoning/emb-004 — "hot" antonyms ─────────────────────────────────
registerCustomScorer("emb-004-hot-antonyms", (output) => {
	const antonyms = ["cold", "cool", "freezing", "frigid", "icy", "chilly"];
	const lower = output.toLowerCase().trim();
	return antonyms.some((a) => lower.includes(a)) ? 1 : 0;
});

// ── semantic-reasoning/emb-005 — "foot → sock" analogy ──────────────────────────
registerCustomScorer("emb-005-foot-analogy", (output) => {
	const valid = ["sock", "shoe", "boot"];
	const lower = output.toLowerCase().trim();
	return valid.some((v) => lower.includes(v)) ? 1 : 0;
});

// ── semantic-reasoning/emb-006 — fish/vegetable grouping ────────────────────────
registerCustomScorer("emb-006-fish-vegetables-grouping", (output) => {
	const lower = output.toLowerCase();
	const fish = ["salmon", "trout", "tuna"];
	const vegetables = ["carrot", "broccoli"];
	let groupedCorrectly = 0;
	for (const f of fish) if (lower.includes(f)) groupedCorrectly++;
	for (const v of vegetables) if (lower.includes(v)) groupedCorrectly++;
	if (groupedCorrectly < 5) return 0;

	const lines = output.split("\n");
	const fishLine = lines.find(
		(l) =>
			l.toLowerCase().includes("salmon") ||
			l.toLowerCase().includes("trout") ||
			l.toLowerCase().includes("tuna"),
	);
	const vegLine = lines.find(
		(l) =>
			l.toLowerCase().includes("carrot") ||
			l.toLowerCase().includes("broccoli"),
	);
	if (!fishLine || !vegLine) return 0.5;
	const fishTogether =
		fishLine.toLowerCase().includes("salmon") &&
		fishLine.toLowerCase().includes("trout") &&
		fishLine.toLowerCase().includes("tuna");
	const vegTogether =
		vegLine.toLowerCase().includes("carrot") &&
		vegLine.toLowerCase().includes("broccoli");
	if (fishTogether && vegTogether) return 1;
	if (fishTogether || vegTogether) return 0.75;
	return 0.5;
});

// ── semantic-reasoning/emb-009 — puppy/kitten analogy ───────────────────────────
registerCustomScorer("emb-009-puppy-kitten", (output) => {
	const lower = output.toLowerCase().trim();
	if (lower.includes("kitten")) return 1;
	if (lower.includes("cub")) return 0.75;
	if (lower.includes("cat") && !lower.includes("category")) return 0.5;
	return 0;
});

// ── semantic-reasoning/emb-012 — light-sense disambiguation ─────────────────────
registerCustomScorer("emb-012-light-sense-disambiguation", (output) => {
	const expectedAnswers = ["different", "same", "same"];
	const lines = output
		.split("\n")
		.map((l) => l.toLowerCase().trim())
		.filter((l) => l.length > 0);

	let correct = 0;
	for (let i = 0; i < expectedAnswers.length; i++) {
		const target = expectedAnswers[i];
		if (lines[i] && lines[i].includes(target)) {
			correct++;
		} else {
			const allAnswers = lines.join(" ");
			const sameCount = (allAnswers.match(/\bsame\b/g) || []).length;
			const diffCount = (allAnswers.match(/\bdifferent\b/g) || []).length;
			if (sameCount === 2 && diffCount === 1) return 0.75;
		}
	}
	return correct / expectedAnswers.length;
});
