import type { KVCacheConfig } from "../src/models/kv-cache.js";
import { KVCache } from "../src/models/kv-cache.js";
import type { GgufValueType } from "../src/models/gguf-types.js";
import { GGUF_MAGIC, GGUF_VERSION } from "../src/models/gguf-types.js";
import {
  type TokenData,
  type TokenizerConfig,
  TokenAttribute,
  TokenizerType,
} from "../src/inference/tokenizer.js";

export function makeRandomLogits(vocabSize: number): Float32Array {
  const logits = new Float32Array(vocabSize);
  for (let i = 0; i < vocabSize; i++) {
    logits[i] = Math.random() * 20 - 10;
  }
  return logits;
}

export function makeBpeTokenData(): {
  config: TokenizerConfig;
  short: string;
  medium: string;
  long: string;
} {
  const tokens: TokenData[] = [
    { text: "<pad>", score: 0, attr: TokenAttribute.CONTROL },
    { text: "<s>", score: 0, attr: TokenAttribute.CONTROL },
    { text: "</s>", score: 0, attr: TokenAttribute.CONTROL },
  ];

  const chars = "abcdefghijklmnopqrstuvwxyz ";
  for (let i = 0; i < chars.length; i++) {
    tokens.push({
      text: chars[i],
      score: -(i + 1),
      attr: TokenAttribute.NORMAL,
    });
  }

  const bigrams = [
    "th",
    "he",
    "in",
    "er",
    "an",
    "on",
    "re",
    "ti",
    "es",
    "ng",
  ];
  for (const bg of bigrams) {
    tokens.push({ text: bg, score: -0.5, attr: TokenAttribute.NORMAL });
  }

  const words = [
    "the",
    "and",
    "for",
    "are",
    "but",
    "not",
    "you",
    "all",
    "can",
    "had",
    "her",
    "was",
    "one",
    "our",
    "out",
    "day",
    "get",
    "has",
    "him",
    "his",
    "how",
    "its",
    "may",
    "new",
    "now",
    "old",
    "see",
    "way",
    "who",
    "boy",
  ];
  for (const w of words) {
    tokens.push({ text: w, score: -0.1, attr: TokenAttribute.NORMAL });
  }

  const moreWords = [
    "that",
    "with",
    "have",
    "this",
    "will",
    "your",
    "from",
    "they",
    "been",
    "call",
    "what",
    "when",
    "some",
    "into",
    "time",
    "very",
    "just",
    "know",
    "take",
    "people",
  ];
  for (const w of moreWords) {
    tokens.push({ text: w, score: -0.05, attr: TokenAttribute.NORMAL });
  }

  const bpeRanks = new Map<string, number>();
  for (let i = 0; i < bigrams.length; i++) {
    bpeRanks.set(bigrams[i][0] + " " + bigrams[i][1], i);
  }

  const config: TokenizerConfig = {
    type: TokenizerType.BPE,
    tokens,
    bpeRanks,
    addedTokens: new Map(),
    eosTokenId: 2,
    bosTokenId: 1,
    padTokenId: 0,
    vocabSize: tokens.length,
  };

  return {
    config,
    short: "hello",
    medium: "the people can see the way for all the boys and girls in the new day",
    long:
      "the people can see the way for all the boys and girls in the new day " +
      "and they know that with time you will have some very good things to say " +
      "about how her new call was just not what she had been into from one old way " +
      "when your boys get their day out may see him take his time now".repeat(3),
  };
}

export function makeSpmTokenData(): {
  config: TokenizerConfig;
  short: string;
  medium: string;
} {
  const tokens: TokenData[] = [
    { text: "<pad>", score: 0, attr: TokenAttribute.CONTROL },
    { text: "<s>", score: 0, attr: TokenAttribute.CONTROL },
    { text: "</s>", score: 0, attr: TokenAttribute.CONTROL },
  ];

  const asciiChars =
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,!?;:'-";
  for (let i = 0; i < asciiChars.length; i++) {
    tokens.push({
      text: asciiChars[i],
      score: -(i + 1),
      attr: TokenAttribute.NORMAL,
    });
  }

  const pairs = ["th", "he", "in", "er", "an", "on", "re", "ti", "es", "ng"];
  for (const p of pairs) {
    tokens.push({ text: p, score: -0.1, attr: TokenAttribute.NORMAL });
  }

  const words = [
    "the",
    "and",
    "for",
    "are",
    "but",
    "not",
    "you",
    "all",
    "can",
    "had",
  ];
  for (const w of words) {
    tokens.push({ text: w, score: -0.05, attr: TokenAttribute.NORMAL });
  }

  const config: TokenizerConfig = {
    type: TokenizerType.SPM,
    tokens,
    bpeRanks: new Map(),
    addedTokens: new Map(),
    eosTokenId: 2,
    bosTokenId: 1,
    padTokenId: 0,
    vocabSize: tokens.length,
  };

  return {
    config,
    short: "the",
    medium: "the people can see the way for all the boys and girls",
  };
}

export function makeKVCacheConfig(
  nLayers = 4,
  nCells = 256,
): KVCacheConfig {
  return {
    nLayers,
    nEmbdHeadK: 64,
    nEmbdHeadV: 64,
    nKvHead: 8,
    maxContextLength: nCells,
    dataType: "f32",
  };
}

export function makePopulatedKVCache(
  nCells = 256,
  nSequences = 2,
): KVCache {
  const config = makeKVCacheConfig(4, nCells);
  const cache = new KVCache(config);
  const cellsPerSeq = Math.floor(nCells / (nSequences + 1));
  for (let seq = 0; seq < nSequences; seq++) {
    const slots = cache.findSlots(cellsPerSeq, seq);
    cache.updateSlots(
      slots,
      Array.from({ length: cellsPerSeq }, (_, i) => i),
      seq,
    );
  }
  return cache;
}

function writeString(buf: DataView, offset: number, str: string): number {
  buf.setBigUint64(offset, BigInt(str.length), true);
  offset += 8;
  for (let i = 0; i < str.length; i++) {
    buf.setUint8(offset++, str.charCodeAt(i));
  }
  return offset;
}

export function makeMinimalGgufBuffer(
  metadataCount = 2,
  tensorCount = 1,
): ArrayBuffer {
  const headerSize = 24;

  const metadataEntries: Array<{
    key: string;
    type: GgufValueType;
    value: unknown;
  }> = [
    {
      key: "general.architecture",
      type: 8 as GgufValueType,
      value: "llama",
    },
    {
      key: "llama.context_length",
      type: 4 as GgufValueType,
      value: 4096,
    },
  ].slice(0, metadataCount);

  let kvSize = 0;
  for (const entry of metadataEntries) {
    kvSize += 8 + entry.key.length + 4;
    if ((entry.type as GgufValueType) === 8)
      kvSize += 8 + (entry.value as string).length;
    else if ((entry.type as GgufValueType) === 4) kvSize += 4;
  }

  let tensorInfoSize = 0;
  const tensorNames: string[] = [];
  for (let i = 0; i < tensorCount; i++) {
    const name = `tensor_${i}.weight`;
    tensorNames.push(name);
    tensorInfoSize += 8 + name.length + 4 + 8 + 4 + 8;
  }

  const totalSize = headerSize + kvSize + tensorInfoSize + 32 + 64;
  const buf = new ArrayBuffer(totalSize);
  const view = new DataView(buf);
  let offset = 0;

  view.setUint32(offset, GGUF_MAGIC, true);
  offset += 4;
  view.setUint32(offset, GGUF_VERSION, true);
  offset += 4;
  view.setBigUint64(offset, BigInt(tensorCount), true);
  offset += 8;
  view.setBigUint64(offset, BigInt(metadataEntries.length), true);
  offset += 8;

  for (const entry of metadataEntries) {
    offset = writeString(view, offset, entry.key);
    view.setUint32(offset, entry.type as number, true);
    offset += 4;
    if ((entry.type as GgufValueType) === 8)
      offset = writeString(view, offset, entry.value as string);
    else if ((entry.type as GgufValueType) === 4) {
      view.setUint32(offset, entry.value as number, true);
      offset += 4;
    }
  }

  for (let i = 0; i < tensorCount; i++) {
    offset = writeString(view, offset, tensorNames[i]);
    view.setUint32(offset, 1, true);
    offset += 4;
    view.setBigUint64(offset, BigInt(10 + i), true);
    offset += 8;
    view.setUint32(offset, 0, true);
    offset += 4;
    view.setBigUint64(offset, BigInt(0), true);
    offset += 8;
  }

  return buf.slice(0, offset);
}
