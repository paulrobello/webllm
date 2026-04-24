/**
 * Re-export of the library-facing evaluation types. Kept as a shim so
 * existing bench-harness imports (`./types.js`) keep working; new code
 * should import directly from `src/evaluation/types.js` or from the
 * package root.
 */
export type {
	DimensionScore,
	EvalDimension,
	EvalReport,
	EvalResult,
	EvalTask,
	EvalToolDef,
	ScoringMethod,
	ToolCallRecord,
} from "../src/evaluation/types.js";
