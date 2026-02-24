# Stage 1: Build
FROM golang:1.25-alpine AS builder

# Pure Go — no C dependencies needed

WORKDIR /src
COPY go.mod go.sum ./
RUN go mod download

COPY . .
RUN CGO_ENABLED=0 go build -ldflags "-s -w" -o /nous ./cmd/nous

# Stage 2: Runtime
FROM alpine:3.21

RUN apk add --no-cache ca-certificates tzdata

# Create non-root user
RUN adduser -D -h /home/nous nous
USER nous

COPY --from=builder /nous /usr/local/bin/nous

# Brain is mounted as a volume
VOLUME ["/brain"]
# Channel state (Matrix credentials, etc.)
VOLUME ["/data"]

ENV NOUS_BRAIN_PATH=/brain
ENV NOUS_DATA_DIR=/data
ENV TZ=Europe/Bucharest

ENTRYPOINT ["nous"]
