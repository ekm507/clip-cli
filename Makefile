BINARY_NAME := clip-cli
CMD_PATH := ./cmd/clip-cli
GOCACHE_DIR := $(CURDIR)/.cache/go-build

.PHONY: build
build:
	mkdir -p $(GOCACHE_DIR)
	GOCACHE=$(GOCACHE_DIR) go build -o $(BINARY_NAME) $(CMD_PATH)
