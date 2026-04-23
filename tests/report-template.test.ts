import { expect, test } from "bun:test";
import { readFileSync } from "node:fs";
import { resolve } from "node:path";

const reportTemplatePath = resolve(
	import.meta.dir,
	"../eval/report-template.html",
);

test("eval report dimension cards are clickable filters", () => {
	const template = readFileSync(reportTemplatePath, "utf-8");

	expect(template).toContain("card.addEventListener('click'");
	expect(template).toContain("sel.value = dim");
	expect(template).toContain("renderTable(data)");
});
