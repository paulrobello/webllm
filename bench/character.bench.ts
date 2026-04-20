import { bench, group } from "mitata";
import { Character } from "../src/characters/character.js";
import type { ToolDefinition } from "../src/characters/tool-system.js";

const tools: ToolDefinition[] = Array.from({ length: 3 }, (_, i) => ({
  name: `bench_tool_${i}`,
  description: `Benchmark tool ${i}`,
  parameters: {
    input: {
      type: "string" as const,
      description: "Input value",
      required: true,
    },
  },
  handler: async (args: Record<string, unknown>) => ({
    output: args.input,
  }),
}));

group("construction", () => {
  bench("simple character", () => {
    new Character({
      modelId: "bench-model",
      systemPrompt: "You are a benchmark assistant.",
    });
  });

  bench("character with tools", () => {
    new Character({
      modelId: "bench-model",
      systemPrompt: "You are a benchmark assistant.",
      tools,
    });
  });
});

group("chat (stub)", () => {
  bench("chat yields response", async () => {
    const char = new Character({
      modelId: "bench-model",
      systemPrompt: "You are a benchmark assistant.",
    });
    for await (const _ of char.chat("Hello")) {
      /* consume */
    }
  });
});

group("getHistory", () => {
  bench("after 3 messages", async () => {
    const char = new Character({
      modelId: "bench-model",
      systemPrompt: "You are a benchmark assistant.",
    });
    for await (const _ of char.chat("First")) {
      /* consume */
    }
    for await (const _ of char.chat("Second")) {
      /* consume */
    }
    for await (const _ of char.chat("Third")) {
      /* consume */
    }
    char.getHistory();
  });
});

group("clearHistory", () => {
  bench("clear after 5 messages", async () => {
    const char = new Character({
      modelId: "bench-model",
      systemPrompt: "You are a benchmark assistant.",
    });
    for (let i = 0; i < 5; i++) {
      for await (const _ of char.chat(`Msg ${i}`)) {
        /* consume */
      }
    }
    char.clearHistory();
  });
});
