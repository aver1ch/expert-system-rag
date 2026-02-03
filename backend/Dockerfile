FROM golang:1.23-alpine AS builder

WORKDIR /app

COPY go.mod ./
COPY cmd ./cmd

RUN go build -o server ./cmd/server

FROM alpine:3.20

WORKDIR /app

COPY --from=builder /app/server ./server

EXPOSE 8080

ENV CORE_SERVICE_URL=http://core:8000

CMD ["./server"]

