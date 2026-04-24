import { expect, test } from "bun:test";
import {
	computeSystemId,
	type SystemProfileInput,
} from "../src/evaluation/system-profile.js";

const SAMPLE: SystemProfileInput = {
	userAgent:
		"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36",
	chromeVersion: "130.0.0.0",
	platform: "MacIntel",
	hardwareConcurrency: 16,
	deviceMemoryGb: 8,
	gpuVendor: "Apple",
	gpuArchitecture: "metal-3",
	gpuDevice: "Apple M5 Max",
	gpuDescription: "Apple M5 Max",
	gpuMaxBufferSize: 268435456,
	gpuMaxStorageBufferBindingSize: 134217728,
};

test("computeSystemId is stable for the same core fields", async () => {
	const id1 = await computeSystemId(SAMPLE);
	const id2 = await computeSystemId({ ...SAMPLE });
	expect(id1).toBe(id2);
	expect(id1).toMatch(/^[0-9a-f]{16}$/);
});

test("dynamic fields don't affect the system id", async () => {
	const a = await computeSystemId({
		...SAMPLE,
		screenWidth: 1920,
		screenHeight: 1080,
		devicePixelRatio: 2,
		gpuFeatures: ["timestamp-query"],
	});
	const b = await computeSystemId({
		...SAMPLE,
		screenWidth: 3840,
		screenHeight: 2160,
		devicePixelRatio: 1,
		gpuFeatures: ["bgra8unorm-storage"],
	});
	expect(a).toBe(b);
});

test("changing a core field produces a different id", async () => {
	const base = await computeSystemId(SAMPLE);
	const diffGpu = await computeSystemId({
		...SAMPLE,
		gpuArchitecture: "ampere",
	});
	const diffChrome = await computeSystemId({ ...SAMPLE, chromeVersion: "131" });
	const diffCpu = await computeSystemId({ ...SAMPLE, hardwareConcurrency: 8 });
	expect(diffGpu).not.toBe(base);
	expect(diffChrome).not.toBe(base);
	expect(diffCpu).not.toBe(base);
});

test("undefined and empty-string core fields are treated as missing", async () => {
	const a = await computeSystemId({
		...SAMPLE,
		gpuDescription: undefined,
		gpuArchitecture: "",
	});
	const b = await computeSystemId({ ...SAMPLE, gpuDescription: undefined });
	// Both omit the empty fields → same id.
	const stripped: SystemProfileInput = { ...SAMPLE };
	delete (stripped as { gpuArchitecture?: string }).gpuArchitecture;
	delete (stripped as { gpuDescription?: string }).gpuDescription;
	const c = await computeSystemId(stripped);
	expect(a).toBe(c);
	// b only drops description, so its id depends on whether
	// gpuArchitecture is part of the canonical hash. It is, so b differs
	// from a (which dropped both).
	expect(b).not.toBe(a);
});
