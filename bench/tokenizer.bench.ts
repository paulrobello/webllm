import { bench, group } from "mitata";
import { Tokenizer } from "../src/inference/tokenizer.js";
import { makeBpeTokenData, makeSpmTokenData } from "./helpers.js";

const bpe = makeBpeTokenData();
const bpeTokenizer = new Tokenizer(bpe.config);

const spm = makeSpmTokenData();
const spmTokenizer = new Tokenizer(spm.config);

const tokenIds = Array.from({ length: 50 }, (_, i) => i + 3);

group("BPE encode", () => {
  bench("short text", () => {
    bpeTokenizer.encode(bpe.short);
  });

  bench("medium text", () => {
    bpeTokenizer.encode(bpe.medium);
  });

  bench("long text", () => {
    bpeTokenizer.encode(bpe.long);
  });
});

group("SPM encode", () => {
  bench("short text", () => {
    spmTokenizer.encode(spm.short);
  });

  bench("medium text", () => {
    spmTokenizer.encode(spm.medium);
  });
});

group("Decode", () => {
  bench("50 tokens", () => {
    bpeTokenizer.decode(tokenIds);
  });
});
