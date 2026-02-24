BINARY := nous
VERSION := $(shell git describe --tags --always --dirty 2>/dev/null || echo "dev")
COMMIT := $(shell git rev-parse --short HEAD 2>/dev/null || echo "unknown")
LDFLAGS := -ldflags "-s -w -X main.version=$(VERSION) -X main.commit=$(COMMIT)"

.PHONY: build run test clean

build:
	CGO_ENABLED=0 go build $(LDFLAGS) -o bin/$(BINARY) ./cmd/nous

run: build
	./bin/$(BINARY) -brain ../brain

test:
	go test ./...

clean:
	rm -rf bin/

# Docker build (for container deployment)
docker:
	docker build -t nous-daemon:$(VERSION) .

# Development: run with brain pointing to workspace brain
dev: build
	NOUS_BRAIN_PATH=../brain ./bin/$(BINARY)

# Tidy dependencies
tidy:
	go mod tidy
