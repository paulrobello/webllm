import { bench, group } from "mitata";
import {
  type ToolCall,
  type ToolDefinition,
  ToolSystem,
} from "../src/characters/tool-system.js";

function makeTools(count: number): ToolDefinition[] {
  return Array.from({ length: count }, (_, i) => ({
    name: `tool_${i}`,
    description: `Description for tool ${i}`,
    parameters: {
      param_a: {
        type: "string" as const,
        description: "First parameter",
        required: true,
      },
      param_b: {
        type: "number" as const,
        description: "Second parameter",
      },
    },
    handler: async (args: Record<string, unknown>) => ({
      result: args.param_a,
    }),
  }));
}

const system = new ToolSystem(makeTools(5));
const emptySystem = new ToolSystem([]);

const xmlInput =
  '<tool_call={"name": "tool_0", "arguments": {"param_a": "hello", "param_b": 42}}>';
const jsonInput =
  '{"name": "tool_1", "arguments": {"param_a": "world", "param_b": 99}}';
const negativeInput =
  "This is just regular text with no tool calls at all.";

const sampleCall: ToolCall = {
  name: "tool_0",
  arguments: { param_a: "test", param_b: 7 },
};

group("parseToolCall", () => {
  bench("XML format", () => {
    system.parseToolCall(xmlInput);
  });

  bench("JSON format", () => {
    system.parseToolCall(jsonInput);
  });

  bench("negative (no tool call)", () => {
    system.parseToolCall(negativeInput);
  });
});

group("execute", () => {
  bench("known tool", async () => {
    await system.execute(sampleCall);
  });
});

group("formatForPrompt", () => {
  bench("5 tools", () => {
    system.formatForPrompt();
  });

  bench("empty system", () => {
    emptySystem.formatForPrompt();
  });
});
