import { expect, test } from "bun:test";
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import vm from "node:vm";

const reportTemplatePath = resolve(
	import.meta.dir,
	"../eval/report-template.html",
);

class FakeClassList {
	private classes = new Set<string>();

	constructor(initial?: string) {
		if (initial) {
			for (const value of initial.split(/\s+/).filter(Boolean)) {
				this.classes.add(value);
			}
		}
	}

	add(...values: string[]): void {
		for (const value of values) this.classes.add(value);
	}

	remove(...values: string[]): void {
		for (const value of values) this.classes.delete(value);
	}

	contains(value: string): boolean {
		return this.classes.has(value);
	}

	toggle(value: string, force?: boolean): boolean {
		if (force === true) {
			this.classes.add(value);
			return true;
		}
		if (force === false) {
			this.classes.delete(value);
			return false;
		}
		if (this.classes.has(value)) {
			this.classes.delete(value);
			return false;
		}
		this.classes.add(value);
		return true;
	}
}

class FakeElement {
	id = "";
	className = "";
	textContent = "";
	value = "";
	disabled = false;
	tabIndex = 0;
	dataset: Record<string, string> = {};
	children: FakeElement[] = [];
	parent: FakeElement | null = null;
	classList = new FakeClassList();
	private listeners = new Map<
		string,
		Array<(event?: Record<string, unknown>) => void>
	>();
	private attributes = new Map<string, string>();
	private innerHtmlValue = "";
	scrollCalls = 0;

	constructor(
		readonly tagName: string,
		options: {
			id?: string;
			className?: string;
			value?: string;
			dataset?: Record<string, string>;
		} = {},
	) {
		if (options.id) this.id = options.id;
		if (options.className) {
			this.className = options.className;
			this.classList = new FakeClassList(options.className);
		}
		if (options.value) this.value = options.value;
		if (options.dataset) this.dataset = { ...options.dataset };
	}

	appendChild(child: FakeElement): FakeElement {
		child.parent = this;
		this.children.push(child);
		return child;
	}

	get innerHTML(): string {
		return this.innerHtmlValue;
	}

	set innerHTML(value: string) {
		this.innerHtmlValue = value;
		if (value === "") {
			this.children = [];
		}
	}

	addEventListener(
		type: string,
		handler: (event?: Record<string, unknown>) => void,
	): void {
		const handlers = this.listeners.get(type) ?? [];
		handlers.push(handler);
		this.listeners.set(type, handlers);
	}

	click(): void {
		for (const handler of this.listeners.get("click") ?? []) {
			handler({ currentTarget: this, target: this });
		}
	}

	dispatchKey(key: string): void {
		for (const handler of this.listeners.get("keydown") ?? []) {
			let prevented = false;
			handler({
				key,
				currentTarget: this,
				target: this,
				preventDefault: () => {
					prevented = true;
				},
			});
			if (prevented) break;
		}
	}

	setAttribute(name: string, value: string): void {
		this.attributes.set(name, value);
		if (name === "id") this.id = value;
	}

	getAttribute(name: string): string | null {
		return this.attributes.get(name) ?? null;
	}

	get options(): FakeElement[] {
		return this.children;
	}

	scrollIntoView(): void {
		this.scrollCalls += 1;
	}
}

function buildTemplateHarness() {
	const elements = new Map<string, FakeElement>();
	const make = (
		tagName: string,
		id: string,
		options: {
			className?: string;
			value?: string;
			dataset?: Record<string, string>;
		} = {},
	): FakeElement => {
		const element = new FakeElement(tagName, { id, ...options });
		elements.set(id, element);
		return element;
	};

	const filterDimension = make("select", "filter-dimension", { value: "all" });
	filterDimension.appendChild(new FakeElement("option", { value: "all" }));
	const filterDifficulty = make("select", "filter-difficulty", {
		value: "all",
	});
	filterDifficulty.appendChild(new FakeElement("option", { value: "all" }));
	filterDifficulty.appendChild(new FakeElement("option", { value: "easy" }));
	filterDifficulty.appendChild(new FakeElement("option", { value: "medium" }));
	filterDifficulty.appendChild(new FakeElement("option", { value: "hard" }));

	const sortId = new FakeElement("button", {
		className: "active",
		dataset: { sort: "id" },
	});
	const sortScore = new FakeElement("button", { dataset: { sort: "score" } });
	const sortLatency = new FakeElement("button", {
		dataset: { sort: "latency" },
	});
	const sortableHeaders = [
		new FakeElement("th", { dataset: { col: "id" } }),
		new FakeElement("th", { dataset: { col: "dimension" } }),
		new FakeElement("th", { dataset: { col: "difficulty" } }),
		new FakeElement("th", { dataset: { col: "score" } }),
		new FakeElement("th", { dataset: { col: "latency" } }),
		new FakeElement("th", { dataset: { col: "tps" } }),
	];

	const dimensionGrid = make("div", "dimension-grid");
	const resultsSection = new FakeElement("section", {
		className: "results-section",
	});
	const tbody = make("tbody", "results-tbody");

	const document = {
		createElement(tagName: string) {
			return new FakeElement(tagName);
		},
		getElementById(id: string) {
			const element = elements.get(id);
			if (!element) {
				throw new Error(`Missing element: ${id}`);
			}
			return element;
		},
		querySelector(selector: string) {
			if (selector === ".results-section") return resultsSection;
			return null;
		},
		querySelectorAll(selector: string) {
			if (selector === ".results-controls button[data-sort]") {
				return [sortId, sortScore, sortLatency];
			}
			if (selector === ".results-table thead th[data-col]") {
				return sortableHeaders;
			}
			if (selector === ".results-table thead th") {
				return sortableHeaders;
			}
			if (selector === ".col-output") {
				return [];
			}
			return [];
		},
		addEventListener() {},
	};

	for (const id of [
		"model-name",
		"report-timestamp",
		"report-total-meta",
		"footer-timestamp",
		"summary-score",
		"summary-score-sub",
		"summary-tasks",
		"summary-passrate",
		"summary-pass-sub",
		"summary-latency",
		"detail-modal",
		"modal-close",
		"modal-title",
		"modal-body",
		"compare-file",
		"compare-label",
		"compare-model",
		"compare-section",
		"compare-grid",
		"clear-dimension-filter",
	]) {
		if (!elements.has(id)) {
			make("div", id);
		}
	}

	const clearDimensionFilter = elements.get("clear-dimension-filter");
	if (!clearDimensionFilter) {
		throw new Error("Missing clear-dimension-filter element");
	}

	return {
		context: {
			console,
			window: {},
			document,
		} as {
			console: Console;
			window: Record<string, unknown>;
			document: unknown;
			render?: (report: unknown, compare: boolean) => void;
			setupControls?: (report: unknown) => void;
		},
		elements: {
			dimensionGrid,
			filterDimension,
			filterDifficulty,
			resultsSection,
			tbody,
			clearDimensionFilter,
		},
	};
}

function loadTemplateScript() {
	const template = readFileSync(reportTemplatePath, "utf-8");
	const match = template.match(/<script>([\s\S]*)<\/script>\s*<\/body>/);
	if (!match) {
		throw new Error("Could not extract report template script");
	}
	return match[1];
}

test("eval report dimension cards filter results and reset back to all", () => {
	const script = loadTemplateScript();
	const { context, elements } = buildTemplateHarness();
	vm.runInNewContext(script, context);

	const report = {
		modelId: "demo-model",
		timestamp: "2026-04-23T23:59:00.000Z",
		totalTasks: 2,
		overall: 0.75,
		dimensions: {
			reasoning: { score: 1, passed: 1, total: 1, avgLatencyMs: 1200 },
			"tool-calling": { score: 0.5, passed: 0, total: 1, avgLatencyMs: 1500 },
		},
		results: [
			{
				taskId: "r1",
				dimension: "reasoning",
				difficulty: "easy",
				score: 1,
				latencyMs: 1200,
				tokensPerSecond: 25,
				modelOutput: "ok",
			},
			{
				taskId: "t1",
				dimension: "tool-calling",
				difficulty: "hard",
				score: 0.5,
				latencyMs: 1500,
				tokensPerSecond: 20,
				modelOutput: "tool",
			},
		],
	};

	// `vm.runInNewContext` injects `render` and `setupControls` into the
	// sandbox context as side effects of the script body.
	if (!context.render || !context.setupControls) {
		throw new Error("template script did not inject render/setupControls");
	}
	context.render(report, false);
	context.setupControls(report);

	expect(elements.dimensionGrid.children).toHaveLength(2);
	expect(elements.tbody.children).toHaveLength(2);
	expect(elements.clearDimensionFilter.disabled).toBe(true);

	const reasoningCard = elements.dimensionGrid.children[0];
	reasoningCard.click();

	expect(elements.filterDimension.value).toBe("reasoning");
	expect(elements.tbody.children).toHaveLength(1);
	expect(elements.resultsSection.scrollCalls).toBe(1);
	expect(elements.clearDimensionFilter.disabled).toBe(false);
	expect(elements.dimensionGrid.children[0].classList.contains("active")).toBe(
		true,
	);
	expect(elements.dimensionGrid.children[1].classList.contains("active")).toBe(
		false,
	);
	expect(elements.dimensionGrid.children[0].getAttribute("aria-pressed")).toBe(
		"true",
	);

	elements.clearDimensionFilter.click();

	expect(elements.filterDimension.value).toBe("all");
	expect(elements.tbody.children).toHaveLength(2);
	expect(elements.clearDimensionFilter.disabled).toBe(true);
	expect(elements.dimensionGrid.children[0].classList.contains("active")).toBe(
		false,
	);
	expect(elements.dimensionGrid.children[1].classList.contains("active")).toBe(
		false,
	);
});
