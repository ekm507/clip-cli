# Repository Guidelines

## Project Structure & Module Organization
This repository is a Go CLI for CLIP embedding and search.

- `cmd/clip-cli/main.go`: executable entrypoint.
- `internal/app/`: command orchestration (`add`, `search`).
- `internal/clip/`: ONNX runtime setup, model execution, image/text preprocessing.
- `internal/storage/`: SQLite persistence and embedding serialization.
- `internal/config/`: default runtime paths and model I/O names.
- `internal/vector/`: vector math helpers.
- `models/`: required model artifacts (`vision.onnx`, `text.onnx`, `tokenizer.json`).
- `Makefile`: build automation.

Keep new logic under `internal/` by concern; keep `cmd/` minimal.

## Build, Test, and Development Commands
- `make build`: builds `./clip-cli` from `./cmd/clip-cli` using a local Go cache.
- `go build ./...`: compile-check all packages.
- `go test ./...`: run all tests.
- `go run ./cmd/clip-cli search --text "cat" --limit 5`: run without creating a binary.

Runtime note: ONNX shared library is required (`libonnxruntime.so`), currently configured at `/usr/lib/libonnxruntime.so` in `internal/config/config.go`.

## Coding Style & Naming Conventions
- Follow standard Go formatting: run `gofmt -w` on changed `.go` files.
- Package names: short, lowercase, no underscores.
- Exported identifiers: `CamelCase`; unexported: `camelCase`.
- Prefer small, focused files by responsibility (runtime, preprocess, storage, etc.).
- Return wrapped errors with context (`fmt.Errorf("...: %w", err)`).

## Testing Guidelines
Use Go’s built-in `testing` package.

- Place tests next to implementation as `*_test.go`.
- Name tests as `Test<FunctionOrBehavior>` (e.g., `TestL2Normalize`).
- Prefer table-driven tests for preprocess/vector/storage logic.
- For DB tests, use temporary SQLite files (avoid committing `*.db` fixtures).

## Commit & Pull Request Guidelines
Recent history uses short, imperative commit messages (e.g., `refactor the project`, `ignore cache`). Continue this style and keep scope focused.

For PRs include:
- What changed and why.
- Key commands run (`make build`, `go test ./...`).
- Any config/path changes (especially ONNX shared library or model paths).
- Sample CLI usage/output when behavior changes.
