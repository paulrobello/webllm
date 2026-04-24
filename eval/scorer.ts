/**
 * Re-export of the library-facing scorer. See `src/evaluation/scorer.ts`
 * for the implementation. Kept as a shim so existing harness imports
 * (`./scorer.js`) keep working.
 */
export { score } from "../src/evaluation/scorer.js";
