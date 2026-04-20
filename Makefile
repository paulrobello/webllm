.PHONY: build test lint fmt typecheck checkall clean install deps wasm-build \
        bench bench-perf bench-eval bench-eval-interactive bench-all

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

bench-eval:
	bun run bench:eval

bench-eval-interactive:
	bun run bench:eval -i

bench-eval-list:
	bun run bench:eval --list

bench-all: bench-perf bench-eval

wasm-build:
	cd src/wasm && mkdir -p build && cd build && \
	emcmake cmake .. \
		-DGGML_WEBGPU=ON \
		-DGGML_WEBGPU_JSPI=ON \
		-DCMAKE_BUILD_TYPE=Release && \
	cmake --build . --config Release -j

clean:
	rm -rf dist node_modules src/wasm/build
