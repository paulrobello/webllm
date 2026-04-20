import { bench, group } from "mitata";
import { GgufParser } from "../src/models/gguf-parser.js";
import { makeMinimalGgufBuffer } from "./helpers.js";

const minimalBuf = makeMinimalGgufBuffer(2, 1);
const smallBuf = makeMinimalGgufBuffer(5, 3);
const mediumBuf = makeMinimalGgufBuffer(10, 10);
const largeBuf = makeMinimalGgufBuffer(20, 30);

group("GgufParser.parse", () => {
  bench("minimal (2 KV, 1 tensor)", () => {
    GgufParser.parse(minimalBuf);
  });

  bench("small (5 KV, 3 tensors)", () => {
    GgufParser.parse(smallBuf);
  });

  bench("medium (10 KV, 10 tensors)", () => {
    GgufParser.parse(mediumBuf);
  });

  bench("large (20 KV, 30 tensors)", () => {
    GgufParser.parse(largeBuf);
  });
});
