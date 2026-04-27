.PHONY: build test lint lint-fix fmt typecheck checkall clean install deps \
        wasm-build wasm-clean \
        bench bench-perf bench-eval bench-eval-interactive bench-eval-list \
        bench-eval-models bench-inference bench-inference-save embed-perf embed-perf-baseline bench-chat-smoke bench-chat-smoke-matrix bench-chat-smoke-matrix-full bench-profile bench-browser-eval bench-full bench-all \
        smoke-test smoke-serve smoke-stop smoke-restart smoke-open smoke-run smoke-bench \
        dashboard-serve dashboard-stop dashboard-db-reset agentchrome-stop stop-all \
        run-all help

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MODEL          ?= hermes-3-llama-3.2-3b-q4f16
PERF_MODEL     ?= tinyllama-1.1b-chat-q4_0
PERF_RUNS      ?= 3
SMOKE_PORT     ?= 8031
DASHBOARD_PORT ?= 8033
DASHBOARD_HOST ?= 0.0.0.0
DASHBOARD_DB   ?= eval/reports/smoke-runs.db

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
# Set WEBLLM_ASSERTIONS=1 on the make invocation to build the WASM with
# Emscripten -sASSERTIONS=1, preserving GGML_ASSERT messages in the browser
# console (useful while chasing WASM aborts). Default is off (production speed).
WEBLLM_ASSERTIONS ?= 0

wasm-build: ## Build ggml-webgpu WASM via Emscripten (pass WEBLLM_ASSERTIONS=1 for diagnostic build)
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
		-DGGML_BACKEND_DL=OFF \
		-DWEBLLM_ASSERTIONS=$(WEBLLM_ASSERTIONS) && \
	cmake --build . --config Release -j

wasm-build-debug: WEBLLM_ASSERTIONS=1 ## Build WASM with -sASSERTIONS=1 (slower, preserves abort messages)
wasm-build-debug: wasm-clean wasm-build

wasm-clean: ## Remove WASM build artifacts
	rm -rf src/wasm/build

# ---------------------------------------------------------------------------
# Vendored browser libraries
# ---------------------------------------------------------------------------
vendor-refresh: ## Refresh smoke-test/vendor/ from node_modules after bumping chart.js
	@mkdir -p smoke-test/vendor
	@cp node_modules/chart.js/dist/chart.umd.min.js smoke-test/vendor/
	@echo "smoke-test/vendor/chart.umd.min.js ← node_modules/chart.js"

# ---------------------------------------------------------------------------
# Smoke Test (browser end-to-end)
# ---------------------------------------------------------------------------
smoke-test: wasm-build ## Bundle + copy WASM artifacts into smoke-test/
	bun build src/index.ts --outfile smoke-test/webllm-bundle.js --target browser
	cp src/wasm/build/webllm-wasm.js src/wasm/build/webllm-wasm.wasm smoke-test/

smoke-serve: smoke-test ## Serve smoke-test/ on http://localhost:$(SMOKE_PORT)
	bun run eval/smoke-serve.ts --port $(SMOKE_PORT)

smoke-stop: ## Kill the smoke-test HTTP server
	lsof -ti:$(SMOKE_PORT) | xargs kill -9 2>/dev/null || true

smoke-restart: smoke-test ## Kill any server on SMOKE_PORT and start a fresh one in background
	@lsof -ti:$(SMOKE_PORT) | xargs kill -9 2>/dev/null || true
	@bun run eval/smoke-serve.ts --port $(SMOKE_PORT) >/dev/null 2>&1 &
	@sleep 1
	@echo "smoke server running on http://localhost:$(SMOKE_PORT)"

smoke-open: smoke-test ## Build + open smoke-test in default browser
	open http://localhost:$(SMOKE_PORT)/real-model.html

smoke-run: smoke-test ## Build, serve in background, open browser (Ctrl-C to stop)
	@echo "Serving smoke-test on http://localhost:$(SMOKE_PORT) ..."
	@lsof -ti:$(SMOKE_PORT) | xargs kill -9 2>/dev/null || true
	@bun run eval/smoke-serve.ts --port $(SMOKE_PORT) &
	@sleep 1 && open http://localhost:$(SMOKE_PORT)/real-model.html
	@echo "Press Ctrl-C to stop the server."
	@wait

dashboard-serve: ## Run live benchmark dashboard (SSE backend, SQLite-persisted, LAN-bound)
	@lsof -ti:$(DASHBOARD_PORT) | xargs kill -9 2>/dev/null || true
	@echo "dashboard → http://localhost:$(DASHBOARD_PORT)/ (bound to $(DASHBOARD_HOST), db=$(DASHBOARD_DB))"
	bun run eval/live-server.ts --port $(DASHBOARD_PORT) --host $(DASHBOARD_HOST) --db $(DASHBOARD_DB)

dashboard-stop: ## Kill the dashboard server
	lsof -ti:$(DASHBOARD_PORT) | xargs kill -9 2>/dev/null || true

dashboard-db-reset: ## Stop the dashboard, delete its SQLite file (+WAL/SHM), next start is empty
	@lsof -ti:$(DASHBOARD_PORT) | xargs kill 2>/dev/null || true
	@sleep 1
	@rm -f $(DASHBOARD_DB) $(DASHBOARD_DB)-wal $(DASHBOARD_DB)-shm
	@echo "removed $(DASHBOARD_DB) (and -wal/-shm sidecars if any)"

stop-all: smoke-stop dashboard-stop agentchrome-stop ## Stop smoke server, dashboard server, and agentchrome session

agentchrome-stop: ## Stop all agentchrome sessions (kill launched Chrome + clear session file)
	@PORT=$$(agentchrome connect --status 2>/dev/null | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('port','')) if d.get('active') else print('')" 2>/dev/null); \
	if [ -n "$$PORT" ]; then \
		lsof -ti:$$PORT | xargs kill 2>/dev/null || true; \
		echo "stopped Chrome on CDP port $$PORT"; \
	else \
		echo "no active agentchrome session"; \
	fi
	@agentchrome connect --disconnect >/dev/null 2>&1 || true
	@echo "agentchrome session cleared"

smoke-bench: smoke-restart ## End-to-end inference benchmark (auto-launches agentchrome if needed)
	@echo "=== smoke-bench: $(PERF_MODEL)$(if $(PERF_DRAFTER), drafter=$(PERF_DRAFTER)), $(PERF_RUNS) runs ==="
	bun run eval/perf.ts --model $(PERF_MODEL) --runs $(PERF_RUNS) --profile $(if $(PERF_DRAFTER),--drafter $(PERF_DRAFTER))

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

bench-inference: smoke-restart ## Run end-to-end inference perf (needs Chrome)
	bun run eval/perf.ts --model $(PERF_MODEL) --runs $(PERF_RUNS) $(if $(PERF_DRAFTER),--drafter $(PERF_DRAFTER))

bench-inference-save: smoke-restart ## Run inference perf and save baseline
	bun run eval/perf.ts --model $(PERF_MODEL) --runs $(PERF_RUNS) --save

embed-perf: ## Run §D encoder perf harness; pass extra args via EMBED_PERF_ARGS
	@bun run eval/embed-perf.ts $(EMBED_PERF_ARGS)

embed-perf-baseline: ## Run §D encoder perf harness with timestamped baseline output dir
	@bun run eval/embed-perf.ts --out eval/reports/embed-perf-baseline-$(shell date +%Y%m%d-%H%M%S)/

bench-chat-smoke: smoke-restart ## Run browser-driven interactive chat smoke regression
	bun run eval/chat-smoke.ts --model $(PERF_MODEL)

bench-chat-smoke-matrix: smoke-restart ## Run browser-driven chat smoke matrix across default pages/models
	bun run eval/chat-smoke-matrix.ts

bench-chat-smoke-matrix-full: smoke-restart ## Run full chat smoke matrix (adds thinking-on for Qwen3 models)
	bun run eval/chat-smoke-matrix.ts --preset full

bench-profile: smoke-restart ## Combined speed + accuracy bench for a profile (PROFILES=<names or set>)
	@test -n "$(PROFILES)" || (echo "ERROR: set PROFILES=<profile-or-set>. e.g. make bench-profile PROFILES=llama-vs-qwen"; exit 1)
	bun run eval/bench.ts --profiles "$(PROFILES)"

bench-browser-eval: smoke-restart ## Browser-only accuracy eval (real WebGPU per task) — needs dashboard running
	@test -n "$(PROFILE)" || (echo "ERROR: set PROFILE=<profile>. e.g. make bench-browser-eval PROFILE=qwen3-0.6b-off-warm"; exit 1)
	@test -n "$(WEBLLM_LIVE_BENCH_URL)" || (echo "ERROR: set WEBLLM_LIVE_BENCH_URL=http://localhost:$(DASHBOARD_PORT). Dashboard must be running (make dashboard-serve)."; exit 1)
	bun run eval/browser-eval.ts --profile $(PROFILE) --live-bench-url $(WEBLLM_LIVE_BENCH_URL)

bench-full: smoke-restart ## Speed + accuracy for every configured profile, streamed to the dashboard
	@curl -sf http://localhost:$(DASHBOARD_PORT)/health >/dev/null 2>&1 || \
		(echo "ERROR: dashboard not reachable on port $(DASHBOARD_PORT). Run 'make dashboard-serve' in another terminal first."; exit 1)
	@WEBLLM_LIVE_BENCH_URL=http://localhost:$(DASHBOARD_PORT) \
		bun run eval/bench.ts --profiles full --fail-fast

bench-all: bench-perf bench-eval ## Run all benchmarks

.PHONY: bench-prefill-tiling
bench-prefill-tiling: ## Run the §22 prefill-tile measurement matrix into eval/reports/prefill-tiling-2026-04-27/
	@echo "==> see docs/superpowers/plans/2026-04-27-prefill-tiling.md Task 5"
	@echo "==> matrix is captured manually via agentchrome; this target is a placeholder"
	@echo "==> for reproducibility — re-run the cells documented in"
	@echo "==> eval/reports/prefill-tiling-2026-04-27/SUMMARY.md."

# ---------------------------------------------------------------------------
# Combined targets
# ---------------------------------------------------------------------------
run-all: checkall bench-all ## Run all quality checks then all benchmarks

# ---------------------------------------------------------------------------
# Clean
# ---------------------------------------------------------------------------
clean: wasm-clean ## Remove all build artifacts
	rm -rf dist node_modules
