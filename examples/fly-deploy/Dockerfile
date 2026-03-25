FROM node:24-slim AS dashboard

RUN corepack enable pnpm

WORKDIR /build

COPY package.json pnpm-lock.yaml pnpm-workspace.yaml ./
COPY packages/sdk/package.json packages/sdk/
COPY packages/dashboard/package.json packages/dashboard/

RUN pnpm install --frozen-lockfile

COPY packages/sdk packages/sdk
COPY packages/dashboard packages/dashboard

RUN pnpm --filter @substructure.ai/sdk build && pnpm --filter @substructure.ai/dashboard build

# ── Rust ──────────────────────────────────────────────────────────────────────

FROM rust:1.93-slim AS builder

RUN apt-get update && apt-get install -y pkg-config libssl-dev && rm -rf /var/lib/apt/lists/*

WORKDIR /build

COPY Cargo.toml Cargo.toml
COPY crates crates
COPY --from=dashboard /build/packages/dashboard/dist packages/dashboard/dist

RUN cargo build --release --package substructure-core

# ── Runtime ───────────────────────────────────────────────────────────────────

FROM debian:trixie-slim

RUN apt-get update && apt-get install -y ca-certificates && rm -rf /var/lib/apt/lists/*

COPY --from=builder /build/target/release/substructure /usr/local/bin/substructure

EXPOSE 8080

ENV RUST_LOG=info

ENTRYPOINT ["substructure", "serve", "--host", "0.0.0.0", "--port", "8080", "--db", "/data/substructure.db"]
