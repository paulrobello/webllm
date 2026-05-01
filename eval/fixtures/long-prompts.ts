/**
 * Long prompts for §4 FA revisit prefill TTFT measurement.
 *
 * Three deterministic prompts with target token counts measured against
 * the LLaMA / Qwen / Mistral tokenizer family (BPE, ~1 token per ~4
 * characters of English prose). Counts are approximate — within ±15%
 * across the four §4-baseline tokenizers. Treat the IDs as workload
 * labels, not exact token counts.
 *
 * Each prompt ends with an open question that drives the model to
 * generate a long-form answer, exercising the long-decode workload
 * point too.
 */

const FILLER_PARAGRAPH =
	"Software engineering is the systematic application of engineering principles to the design, development, maintenance, testing, and evaluation of software. It involves understanding requirements, designing systems, writing code, performing tests, and ensuring quality through processes that are repeatable and predictable. Modern software engineering relies on tools, frameworks, and methodologies that have evolved over decades of industrial practice. Version control, automated testing, continuous integration, and code review are common pillars of disciplined development workflows.";

function repeat(p: string, times: number): string {
	return Array.from({ length: times }, () => p).join(" ");
}

// ---------------------------------------------------------------------------
// Probe 9a fixtures — NPC-shaped (stable PREFIX / variable TAIL) prompts.
// We run three of these and fit prefillMs = a·prefix + b·tail + c. The
// smoke result line carries `tokensIn=N` so the runner reads the actual
// prompt token count for each fixture rather than relying on the label.
// ---------------------------------------------------------------------------

const NPC_PREFIX_SHORT =
	"You are an NPC AI controller for a fantasy MMO. Tools: move(x, y), speak(text), attack(target), use_item(item), trade(player). Each NPC has stats: hp, mp, level, position, inventory. Pick exactly one tool call per tick.";

const NPC_PREFIX_LONG = `${NPC_PREFIX_SHORT} Detailed tool reference. move(x, y): walk the NPC to grid coordinates (x, y); fails if path is blocked, slowed by terrain. speak(text): emit a short utterance audible to NPCs and players within 12 tiles; logs to chat. attack(target): initiate combat with target NPC or player id; honors faction rules and aggro tables. use_item(item): consume from inventory; potions restore hp/mp, scrolls cast spells, food triggers regen ticks. trade(player): open trade window with target player id; both parties must accept. Stat semantics: hp is current health out of max_hp, depletes from damage and regenerates outside combat; mp is mana for spells, regenerates faster than hp; level scales damage and resists; position is current grid cell as (x, y); inventory is a list of item ids. Decision rules: prefer survival over aggression below 30% hp, prefer engagement above 70% hp, fall back to flee if outnumbered three to one or more, never break neutrality with same-faction NPCs.`;

const NPC_TAIL_SHORT =
	"Goblin sees Hero approaching at distance 8. hp 22/40. Decide:";

const NPC_TAIL_LONG =
	"Goblin sees Hero approaching at distance 8, with two friendly Wolves at distance 4 and 6 respectively. Currently the Goblin is at (15, 22), Hero at (15, 30). Goblin hp is 22/40, mp is 8/15, level 5, inventory contains a healing potion and a rusted dagger. Recent chat log: Hero said 'easy xp', Wolf 1 howled, Wolf 2 is busy fighting another player at (8, 18). Faction stance toward Hero is hostile, toward Wolves is allied. Decide:";

export const LONG_PROMPTS: Record<string, string> = {
	// ~256 tokens
	"prefill-256": `${repeat(FILLER_PARAGRAPH, 1)} Given that context, briefly explain what makes a software engineering team effective.`,
	// ~512 tokens
	"prefill-512": `${repeat(FILLER_PARAGRAPH, 2)} Given that context, briefly explain what makes a software engineering team effective.`,
	// ~1024 tokens
	"prefill-1024": `${repeat(FILLER_PARAGRAPH, 4)} Given that context, briefly explain what makes a software engineering team effective.`,

	// Probe 9a: 3 prompts × known (PREFIX, TAIL) split. The runner pulls
	// each prompt's actual `tokensIn` from the smoke result line and
	// solves for a, b, c.
	"probe9a-Pshort-Tshort": `${NPC_PREFIX_SHORT} ${NPC_TAIL_SHORT}`,
	"probe9a-Pshort-Tlong": `${NPC_PREFIX_SHORT} ${NPC_TAIL_LONG}`,
	"probe9a-Plong-Tshort": `${NPC_PREFIX_LONG} ${NPC_TAIL_SHORT}`,
};
