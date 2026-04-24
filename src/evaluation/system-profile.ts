/**
 * System-profile collection. Captures the (approximate) hardware + browser
 * fingerprint a benchmark run executed against, so dashboard rows don't
 * have to repeat OS/GPU/UA on every record. Each unique system gets a
 * stable id (SHA-256 of the canonical-core fields) and is stored once.
 *
 * Pure runtime-agnostic helpers — collection happens in the browser
 * (where WebGPU lives), but hashing + types are usable from anywhere.
 */

export interface SystemProfile {
	systemId: string;
	collectedAt: string;

	// Browser
	userAgent: string;
	chromeVersion?: string;
	platform?: string;

	// Approximate hardware (browser-exposed; deliberately coarse for
	// fingerprinting reasons in modern Chrome)
	hardwareConcurrency?: number;
	deviceMemoryGb?: number;

	// GPU (WebGPU adapter info + relevant limits)
	gpuVendor?: string;
	gpuArchitecture?: string;
	gpuDevice?: string;
	gpuDescription?: string;
	gpuMaxBufferSize?: number;
	gpuMaxStorageBufferBindingSize?: number;
	gpuFeatures?: string[];

	// Display
	screenWidth?: number;
	screenHeight?: number;
	devicePixelRatio?: number;
}

/**
 * Fields that go into the system-id hash. Two systems are considered the
 * "same" if these match — UA Chrome version + OS, GPU identity, and the
 * coarse hardware shape. Dynamic fields (timestamp, screen size, etc.)
 * are intentionally excluded so id is stable across runs.
 */
const CORE_FIELDS = [
	"userAgent",
	"chromeVersion",
	"platform",
	"hardwareConcurrency",
	"deviceMemoryGb",
	"gpuVendor",
	"gpuArchitecture",
	"gpuDevice",
	"gpuDescription",
	"gpuMaxBufferSize",
	"gpuMaxStorageBufferBindingSize",
] as const;

export type SystemProfileInput = Omit<
	SystemProfile,
	"systemId" | "collectedAt"
>;

/**
 * Compute the canonical system id for a profile. Uses SHA-256 over a
 * stable JSON serialisation of the core fields (sorted keys, undefined
 * dropped). Works in any environment with a WebCrypto-compatible
 * `crypto.subtle` (browsers + Bun + modern Node).
 */
export async function computeSystemId(
	input: SystemProfileInput,
): Promise<string> {
	const canonical: Record<string, unknown> = {};
	for (const field of CORE_FIELDS) {
		const value = (input as Record<string, unknown>)[field];
		if (value !== undefined && value !== null && value !== "") {
			canonical[field] = value;
		}
	}
	const ordered = Object.fromEntries(
		Object.keys(canonical)
			.sort()
			.map((k) => [k, canonical[k]]),
	);
	const data = new TextEncoder().encode(JSON.stringify(ordered));
	const digest = await crypto.subtle.digest("SHA-256", data);
	const bytes = new Uint8Array(digest);
	let hex = "";
	for (const b of bytes) hex += b.toString(16).padStart(2, "0");
	return hex.slice(0, 16); // 64 bits is plenty for dedup; keeps URLs short
}

function parseChromeVersion(userAgent: string): string | undefined {
	const m = userAgent.match(/Chrome\/(\d+(?:\.\d+)*)/);
	return m?.[1];
}

/**
 * Collect a system profile from the browser. Pass an already-acquired
 * WebGPU adapter — we read its info + limits without requesting a device.
 * Falls back gracefully when fields aren't exposed.
 */
export async function collectBrowserSystemProfile(
	adapter: GPUAdapter,
): Promise<SystemProfile> {
	const ua = typeof navigator !== "undefined" ? navigator.userAgent : "";
	const platform =
		typeof navigator !== "undefined" && "platform" in navigator
			? (navigator as Navigator & { platform?: string }).platform
			: undefined;
	const chromeVersion = parseChromeVersion(ua);
	const hardwareConcurrency =
		typeof navigator !== "undefined"
			? navigator.hardwareConcurrency
			: undefined;
	const deviceMemoryGb =
		typeof navigator !== "undefined" && "deviceMemory" in navigator
			? (navigator as Navigator & { deviceMemory?: number }).deviceMemory
			: undefined;

	let gpuInfo: GPUAdapterInfo | undefined;
	try {
		// `info` is the modern accessor; older builds expose
		// `requestAdapterInfo()`. Both yield the same shape.
		gpuInfo = adapter.info;
		if (!gpuInfo && "requestAdapterInfo" in adapter) {
			gpuInfo = await (
				adapter as GPUAdapter & {
					requestAdapterInfo(): Promise<GPUAdapterInfo>;
				}
			).requestAdapterInfo();
		}
	} catch {
		gpuInfo = undefined;
	}

	const features: string[] = [];
	for (const f of adapter.features) features.push(String(f));

	const screenWidth = typeof screen !== "undefined" ? screen.width : undefined;
	const screenHeight =
		typeof screen !== "undefined" ? screen.height : undefined;
	const devicePixelRatio =
		typeof window !== "undefined" ? window.devicePixelRatio : undefined;

	const input: SystemProfileInput = {
		userAgent: ua,
		chromeVersion,
		platform,
		hardwareConcurrency,
		deviceMemoryGb,
		gpuVendor: gpuInfo?.vendor || undefined,
		gpuArchitecture: gpuInfo?.architecture || undefined,
		gpuDevice: gpuInfo?.device || undefined,
		gpuDescription: gpuInfo?.description || undefined,
		gpuMaxBufferSize: adapter.limits.maxBufferSize,
		gpuMaxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
		gpuFeatures: features.length > 0 ? features : undefined,
		screenWidth,
		screenHeight,
		devicePixelRatio,
	};
	const systemId = await computeSystemId(input);
	return {
		systemId,
		collectedAt: new Date().toISOString(),
		...input,
	};
}
