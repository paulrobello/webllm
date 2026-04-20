import { bench, group } from "mitata";
import { Sampler } from "../src/inference/sampler.js";
import { makeRandomLogits } from "./helpers.js";

const sampler = new Sampler({ temperature: 1.0, topK: 40, topP: 0.9 });
const greedySampler = new Sampler({ temperature: 0 });

const logits1k = makeRandomLogits(1_000);
const logits32k = makeRandomLogits(32_000);
const logits128k = makeRandomLogits(128_000);

const recentTokens = Array.from({ length: 64 }, (_, i) => i);

group("sample()", () => {
  bench("1K logits (greedy)", () => {
    greedySampler.sample(logits1k);
  });

  bench("1K logits", () => {
    sampler.sample(new Float32Array(logits1k));
  });

  bench("32K logits", () => {
    sampler.sample(new Float32Array(logits32k));
  });

  bench("128K logits", () => {
    sampler.sample(new Float32Array(logits128k));
  });
});

group("individual transforms", () => {
  bench("applyTemperature 32K", () => {
    sampler.applyTemperature(logits32k);
  });

  bench("applyTopK 32K", () => {
    sampler.applyTopK(logits32k);
  });

  bench("applyTopP 32K", () => {
    sampler.applyTopP(logits32k);
  });

  bench("applyRepetitionPenalty 32K", () => {
    const copy = new Float32Array(logits32k);
    sampler.applyRepetitionPenalty(copy, recentTokens);
  });
});
