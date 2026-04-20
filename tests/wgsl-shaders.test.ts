import { describe, expect, test } from "bun:test";
import {
	ALL_SHADERS,
	SHADER_EMBEDDING_LOOKUP,
	SHADER_GELU,
	SHADER_LAYER_NORM,
	SHADER_MATMUL_F32,
	SHADER_RMS_NORM,
	SHADER_SILU,
	SHADER_SOFTMAX,
} from "../src/inference/wgsl-shaders.js";

describe("WGSL Shaders", () => {
	test("all shaders are non-empty strings", () => {
		for (const [name, source] of Object.entries(ALL_SHADERS)) {
			expect(
				source.length,
				`Shader ${name} should be non-empty`,
			).toBeGreaterThan(0);
		}
	});

	test("all shaders contain @compute entry point", () => {
		for (const [name, source] of Object.entries(ALL_SHADERS)) {
			expect(source, `Shader ${name} should have @compute`).toContain(
				"@compute",
			);
		}
	});

	test("all shaders define storage bindings", () => {
		for (const [name, source] of Object.entries(ALL_SHADERS)) {
			expect(source, `Shader ${name} should have storage bindings`).toContain(
				"var<storage",
			);
		}
	});

	test("matmul shader has workgroup_size 8x8", () => {
		expect(SHADER_MATMUL_F32).toContain("@workgroup_size(8, 8)");
	});

	test("rms_norm uses inversesqrt", () => {
		expect(SHADER_RMS_NORM).toContain("inversesqrt");
	});

	test("layer_norm has bias binding", () => {
		expect(SHADER_LAYER_NORM).toContain("bias: array<f32>");
	});

	test("softmax uses exp for numerically stable computation", () => {
		expect(SHADER_SOFTMAX).toContain("exp(");
	});

	test("gelu uses tanh approximation", () => {
		expect(SHADER_GELU).toContain("tanh");
	});

	test("silu uses sigmoid formula", () => {
		expect(SHADER_SILU).toContain("exp(-");
	});

	test("embedding lookup reads from weights and token_ids", () => {
		expect(SHADER_EMBEDDING_LOOKUP).toContain("weights: array<f32>");
		expect(SHADER_EMBEDDING_LOOKUP).toContain("token_ids: array<u32>");
	});

	test("ALL_SHADERS contains exactly 7 shaders", () => {
		expect(Object.keys(ALL_SHADERS)).toHaveLength(7);
	});
});
