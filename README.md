# clip_cli

A Go CLI for CLIP-based image indexing and text-to-image search using ONNX Runtime.

## Prerequisites

- Go (1.25+)
- SQLite build tooling for `github.com/mattn/go-sqlite3` (CGO-enabled environment)
- ONNX Runtime shared library (`libonnxruntime.so`)

### ONNX Runtime shared library

This app initializes ONNX Runtime from:

`/usr/lib/libonnxruntime.so`

So you must either:

1. Place `libonnxruntime.so` at `/usr/lib/libonnxruntime.so`, or
2. Change `ORTSharedLibPath` in [internal/config/config.go](/home/erfan/works/clip_cli/internal/config/config.go)

If your `.so` is in a custom location, ensure the dynamic linker can resolve it (for example via `LD_LIBRARY_PATH`).

## Models

Download models from:

https://huggingface.co/ajaleksa/clip-onnx-models

Put files at:

- `models/vision.onnx`
- `models/text.onnx`
- `models/tokenizer.json`

## Build

```bash
make build
```

This builds the binary as `./clip-cli`.

You can also build directly:

```bash
go build -o clip-cli ./cmd/clip-cli
```

## Usage

Add an image:

```bash
./clip-cli add --image path/to/image.jpg
```

Search images by text:

```bash
./clip-cli search --text "a cat sitting on a chair" --limit 5
```

Output behavior:

- `stdout`: result filenames only (one per line in text mode)
- `stderr`: progress/log messages

JSON output:

```bash
./clip-cli search --text "a cat sitting on a chair" --limit 5 --json
```

Example JSON response:

```json
{"files":["/path/img1.jpg","/path/img2.jpg"]}
```

Embeddings are stored in `smart_gallery.db`.
