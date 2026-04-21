.PHONY: build test lint fmt typecheck checkall clean install deps wasm-build \
        bench bench-perf bench-eval bench-eval-interactive bench-eval-list \
        bench-eval-models bench-all run-all smoke-test smoke-serve

install:
	bun install

deps: install

build:
	bun run build

test:
	bun run test

lint:
	bun run lint

lint-fix:
	bun run lint:fix

fmt:
	bun run fmt

typecheck:
	bun run typecheck

checkall:
	bun run checkall

bench: bench-perf

bench-perf:
	bun run bench

MODEL ?= hermes-3-llama-3.2-3b-q4f16

bench-eval:
	bun run bench:eval -m $(MODEL) --html

bench-eval-interactive:
	bun run bench:eval -m $(MODEL) -i --html

bench-eval-list:
	bun run bench:eval --list

bench-eval-models:
	bun run bench:eval --models

bench-all: bench-perf bench-eval

run-all: checkall bench-all

wasm-build:
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

wasm-clean:
	rm -rf src/wasm/build

clean:
	rm -rf dist node_modules src/wasm/build

smoke-test: wasm-build
	bun build src/index.ts --outfile smoke-test/webllm-bundle.js --target browser
	cp src/wasm/build/webllm-wasm.js src/wasm/build/webllm-wasm.wasm smoke-test/

smoke-serve:
	cd smoke-test && python3 -m http.server 8031
