.PHONY: build test lint lint-fix fmt typecheck checkall clean install deps \
        wasm-build wasm-clean \
        bench bench-perf bench-eval bench-eval-interactive bench-eval-list \
        bench-eval-models bench-inference bench-inference-save bench-all \
        smoke-test smoke-serve smoke-open smoke-test-full \
        run-all help

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MODEL       ?= hermes-3-llama-3.2-3b-q4f16
PERF_MODEL  ?= tinyllama-1.1b-chat-q4_0
PERF_RUNS   ?= 3
SMOKE_PORT  ?= 8031

# ---------------------------------------------------------------------------
# help — list all targets with descriptions
# ---------------------------------------------------------------------------
help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-22s\033[0m %s\n", $$1, $$2}'

# ---------------------------------------------------------------------------
# Install & Dependencies
# ---------------------------------------------------------------------------
install: ## Install npm dependencies
	bun install

deps: install ## Alias for install

# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------
build: ## Build browser bundle (dist/)
	bun run build

dev: ## Build and watch for changes
	bun run dev

# ---------------------------------------------------------------------------
# Testing
# ---------------------------------------------------------------------------
test: ## Run all tests
	bun run test

# ---------------------------------------------------------------------------
# Code Quality
# ---------------------------------------------------------------------------
fmt: ## Format source with Biome
	bun run fmt

lint: ## Lint source with Biome
	bun run lint

lint-fix: ## Lint and auto-fix
	bun run lint:fix

typecheck: ## Run TypeScript type checking
	bun run typecheck

checkall: fmt lint typecheck test ## Format, lint, typecheck, and test

# ---------------------------------------------------------------------------
# WASM Build (Emscripten / ggml-webgpu)
# ---------------------------------------------------------------------------
wasm-build: ## Build ggml-webgpu WASM via Emscripten
	cd src/wasm && mkdir -p build && cd build && \
	source ~/emsdk/emsdk_env.sh 2>/dev/null; \
	emcmake cmake .. \
		-DGGML_WEBGPU=ON \
		-DGGML_WEBGPU_JSPI=OFF \
		-DCMAKE_BUILD_TYPE=Release \
		-DGGML_CPU=OFF \
		-DGGML_BLAS=OFF \
		-DGGML_METAL=OFF \
		-DGGML_ACCELERATE=OFF \
		-DGGML_CUDA=OFF \
		-DGGML_OPENMP=OFF \
		-DGGML_NATIVE=OFF \
		-DGGML_LLAMAFILE=OFF \
		-DGGML_BUILD_TESTS=OFF \
		-DGGML_BUILD_EXAMPLES=OFF \
		-DBUILD_SHARED_LIBS=OFF \
		-DGGML_BACKEND_DL=OFF && \
	cmake --build . --config Release -j

wasm-clean: ## Remove WASM build artifacts
	rm -rf src/wasm/build

# ---------------------------------------------------------------------------
# Smoke Test (browser end-to-end)
# ---------------------------------------------------------------------------
smoke-test: wasm-build ## Bundle + copy WASM artifacts into smoke-test/
	bun build src/index.ts --outfile smoke-test/webllm-bundle.js --target browser
	cp src/wasm/build/webllm-wasm.js src/wasm/build/webllm-wasm.wasm smoke-test/

smoke-serve: ## Serve smoke-test/ on http://localhost:$(SMOKE_PORT)
	cd smoke-test && python3 -m http.server $(SMOKE_PORT)

smoke-open: ## Open smoke-test in default browser
	open http://localhost:$(SMOKE_PORT)/real-model.html

smoke-test-full: smoke-test ## Build, serve, and open smoke test
	@echo "Smoke test built. Run 'make smoke-serve' in another terminal, then 'make smoke-open'."

# ---------------------------------------------------------------------------
# Benchmarks
# ---------------------------------------------------------------------------
bench: bench-perf ## Run default benchmark suite

bench-perf: ## Run micro-benchmarks
	bun run bench

bench-eval: ## Run model eval benchmark (generates HTML report)
	bun run bench:eval -m $(MODEL) --html

bench-eval-interactive: ## Run model eval interactively
	bun run bench:eval -m $(MODEL) -i --html

bench-eval-list: ## List available eval tasks
	bun run bench:eval --list

bench-eval-models: ## List available eval models
	bun run bench:eval --models

bench-inference: ## Run end-to-end inference perf (needs smoke-serve + Chrome)
	bun run eval/perf.ts --model $(PERF_MODEL) --runs $(PERF_RUNS)

bench-inference-save: ## Run inference perf and save baseline
	bun run eval/perf.ts --model $(PERF_MODEL) --runs $(PERF_RUNS) --save

bench-all: bench-perf bench-eval ## Run all benchmarks

# ---------------------------------------------------------------------------
# Combined targets
# ---------------------------------------------------------------------------
run-all: checkall bench-all ## Run all quality checks then all benchmarks

# ---------------------------------------------------------------------------
# Clean
# ---------------------------------------------------------------------------
clean: wasm-clean ## Remove all build artifacts
	rm -rf dist node_modules
