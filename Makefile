.PHONY: build test lint fmt typecheck checkall clean install deps wasm-build bench bench-eval

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

bench:
	bun run bench

bench-eval:
	bun run bench:eval

wasm-build:
	cd src/wasm && mkdir -p build && cd build && \
	emcmake cmake .. \
		-DGGML_WEBGPU=ON \
		-DGGML_WEBGPU_JSPI=ON \
		-DCMAKE_BUILD_TYPE=Release && \
	cmake --build . --config Release -j

clean:
	rm -rf dist node_modules src/wasm/build
