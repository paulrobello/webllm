import { bench, group } from "mitata";
import { Generator } from "../src/inference/generation.js";
import type { GenerationConfig } from "../src/inference/generation.js";
import { Sampler } from "../src/inference/sampler.js";
import {
  InferenceSession,
  type InferenceSessionConfig,
} from "../src/models/inference-session.js";

const SESSION_CONFIG: InferenceSessionConfig = {
  maxTokens: 500,
  temperature: 0,
  topK: 40,
  topP: 1,
  repetitionPenalty: 1,
  contextOverflowPolicy: "stop",
};

function mockForwardPass(
  _tokenIds: number[],
  _positions: number[],
): Promise<Float32Array> {
  const logits = new Float32Array(100);
  logits[3] = 10.0;
  logits[2] = -100;
  return Promise.resolve(logits);
}

function makeConfig(maxTokens: number): GenerationConfig {
  return {
    prompt: "bench",
    maxTokens,
    temperature: 0,
    topK: 40,
    topP: 1,
    repetitionPenalty: 1,
  };
}

group("generate (mock forward pass)", () => {
  bench("50 tokens", async () => {
    const sampler = new Sampler({ temperature: 0 });
    const session = new InferenceSession(
      { ...SESSION_CONFIG, maxTokens: 200 },
      0,
    );
    const gen = Generator.generate(
      [1],
      sampler,
      session,
      2,
      mockForwardPass,
      makeConfig(50),
    );
    for await (const _ of gen) {
      /* consume */
    }
  });

  bench("200 tokens", async () => {
    const sampler = new Sampler({ temperature: 0 });
    const session = new InferenceSession(
      { ...SESSION_CONFIG, maxTokens: 500 },
      0,
    );
    const gen = Generator.generate(
      [1],
      sampler,
      session,
      2,
      mockForwardPass,
      makeConfig(200),
    );
    for await (const _ of gen) {
      /* consume */
    }
  });
});
